"""Room segmentation for scenes with no region annotation.

Watershed over the navmesh free-space map: rooms are basins of the distance
transform, doorways are the pinch points between them. This is the fallback the
HM3D room-graph builder's docstring names and skips ("no watershed pseudo-rooms
needed") because HM3DSem ships regions -- the 729 unannotated scenes do not.

Merging is the whole game. A raw watershed over-segments badly (every alcove is
a basin), so basins are merged wherever the boundary between them is wide
relative to a door: a real doorway is a narrow pinch (<= ~1.1 m of shared
frontier), an alcove boundary is wide open.

Validation: --validate scores the segmentation against the shipped regions of an
annotated scene (adjusted Rand index over shared nav cells), so pseudo-room
quality is a measured number exactly like the object pipeline's.

  python annot_rooms.py --dataset hm3d --stem X --scene-rel train/000xx-X --out rooms.json
  python annot_rooms.py --dataset hm3d --stem X --scene-rel ... --validate
"""
import argparse
import collections
import heapq
import json
import math
import os

import numpy as np
import habitat_sim
from scipy import ndimage

from c4_collect_expert import HM3D, MP3D, CAM_H
from build_room_graphs import CELL, nav_grid

MIN_ROOM_M2 = 2.0        # basins smaller than this always merge into a neighbour
DOOR_W_M = 1.15          # shared frontier wider than this is not a door -> merge
PEAK_MIN_DIST = 5        # cells between watershed seeds (1.25 m)


def make_sim(glb, gpu):
    # renderer-less: no sensors -> no EGL, runs on any node
    b = habitat_sim.SimulatorConfiguration()
    b.scene_id = glb; b.gpu_device_id = gpu; b.enable_physics = False
    b.create_renderer = False
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = []
    sim = habitat_sim.Simulator(habitat_sim.Configuration(b, [ag]))
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
    ns.agent_radius = 0.18; ns.agent_height = CAM_H
    sim.recompute_navmesh(sim.pathfinder, ns)
    return sim


def levels_of(pf, n=6000, frac=0.05):
    """Floor heights as HISTOGRAM PEAKS of navigable-point y's.

    Gap-clustering fails on multi-storey houses: staircase points form a
    continuous ramp of y values between floors, so no gap ever appears and the
    whole house collapses into one level. Floors are where the MASS is: bins
    holding >= frac of samples, merged when adjacent, ramps ignored.
    """
    ys = []
    for _ in range(n):
        p = pf.get_random_navigable_point()
        if np.all(np.isfinite(p)):
            ys.append(float(p[1]))
    if not ys:
        return []
    hist, edges = np.histogram(ys, bins=int((max(ys) - min(ys)) / 0.2) + 1)
    hot = hist >= frac * len(ys)
    lv, i = [], 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j + 1 < len(hot) and hot[j + 1]:
                j += 1
            lo, hi = edges[i], edges[j + 1]
            pts = [y for y in ys if lo <= y <= hi]
            lv.append(float(np.median(pts)))
            i = j + 1
        else:
            i += 1
    return lv


def watershed(dist, nav):
    """seeded watershed by descending distance; pure numpy+heapq (no skimage)."""
    # seeds: local maxima of the smoothed distance transform
    sm = ndimage.gaussian_filter(dist, 1.2)
    mx = ndimage.maximum_filter(sm, size=2 * PEAK_MIN_DIST + 1)
    seeds = (sm == mx) & nav & (dist > 1.2)          # peak at least 30 cm of clearance... in cells
    lab, nseed = ndimage.label(seeds)
    if nseed == 0:
        lab[nav] = 1
        return lab
    out = np.zeros_like(lab)
    pq = []
    for (i, j) in np.argwhere(lab > 0):
        out[i, j] = lab[i, j]
        heapq.heappush(pq, (-dist[i, j], int(i), int(j)))
    while pq:
        _, i, j = heapq.heappop(pq)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < out.shape[0] and 0 <= b < out.shape[1] \
                    and nav[a, b] and out[a, b] == 0:
                out[a, b] = out[i, j]
                heapq.heappush(pq, (-dist[a, b], a, b))
    return out


def merge_basins(lab, dist):
    """merge across wide boundaries and dissolve tiny basins."""
    def frontier():
        fr = collections.defaultdict(set)
        h, w = lab.shape
        for i in range(h - 1):
            for j in range(w - 1):
                a = lab[i, j]
                for b in (lab[i + 1, j], lab[i, j + 1]):
                    if a and b and a != b:
                        fr[frozenset((a, b))].add((i, j))
        return fr

    changed = True
    while changed:
        changed = False
        sizes = collections.Counter(lab[lab > 0].ravel())
        fr = frontier()
        # widest boundary first, so big open-plan splits heal before doors decide
        for pair, cells in sorted(fr.items(), key=lambda kv: -len(kv[1])):
            a, b = tuple(pair)
            width_m = len(cells) * CELL
            tiny = min(sizes[a], sizes[b]) * CELL * CELL < MIN_ROOM_M2
            if width_m > DOOR_W_M or tiny:
                lab[lab == b] = a
                changed = True
                break
    return lab


def segment(pf, y):
    nav, x0, z0 = nav_grid(pf, y)
    if nav.sum() < 40:
        return None
    dist = ndimage.distance_transform_edt(nav)
    lab = watershed(dist, nav)
    lab = merge_basins(lab, dist)
    return nav, x0, z0, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mp3d', 'hm3d'], required=True)
    ap.add_argument('--stem', required=True)
    ap.add_argument('--scene-rel', default='')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', default='')
    ap.add_argument('--validate', action='store_true',
                    help='score against the shipped regions instead of writing output')
    a = ap.parse_args()

    if a.dataset == 'mp3d':
        glb = f"{MP3D}/{a.stem}/{a.stem}.glb"
    else:
        d = os.path.join(HM3D, a.scene_rel)
        glb = next(p for p in (os.path.join(d, a.stem + ".basis.glb"),
                               os.path.join(d, a.stem + ".glb")) if os.path.exists(p))
    sim = make_sim(glb, a.gpu)
    pf = sim.pathfinder

    result = {}
    for L, y in enumerate(levels_of(pf)):
        seg = segment(pf, y)
        if seg is None:
            continue
        nav, x0, z0, lab = seg
        rooms = []
        for rid in sorted(set(lab[lab > 0].ravel())):
            cells = np.argwhere(lab == rid)
            cx = x0 + cells[:, 0].mean() * CELL
            cz = z0 + cells[:, 1].mean() * CELL
            rooms.append({'cells': cells.tolist(), 'centroid': [float(cx), y, float(cz)],
                          'area_m2': round(float(len(cells)) * CELL * CELL, 2)})
        result[L] = {'y': y, 'x0': float(x0), 'z0': float(z0),
                     'nx': int(nav.shape[0]), 'nz': int(nav.shape[1]), 'rooms': rooms}

    if a.validate:
        DG = os.path.dirname(os.path.abspath(__file__))
        fn = 'room_graphs_hm3d_v2.json' if a.dataset == 'hm3d' else 'room_graphs_mp3d.json'
        G = json.load(open(os.path.join(DG, fn)))[a.stem]
        scores = []
        for L, seg in result.items():
            gm = G.get('grids', {}).get(str(L))
            if gm is None:
                continue
            # ground-truth label image on OUR grid, via world coords
            gt = np.zeros((seg['nx'], seg['nz']), np.int32)
            for gi, (rid, r) in enumerate(G['rooms'].items(), 1):
                if r['level'] != L:
                    continue
                for (i, j) in r.get('cells', []):
                    wx = gm['x0'] + i * CELL; wz = gm['z0'] + j * CELL
                    a_ = int(round((wx - seg['x0']) / CELL))
                    b_ = int(round((wz - seg['z0']) / CELL))
                    if 0 <= a_ < seg['nx'] and 0 <= b_ < seg['nz']:
                        gt[a_, b_] = gi
            pred = np.zeros_like(gt)
            for pi, room in enumerate(seg['rooms'], 1):
                for (i, j) in room['cells']:
                    pred[i, j] = pi
            m = (gt > 0) & (pred > 0)
            if m.sum() < 100:
                continue
            x, yv = gt[m], pred[m]
            # adjusted Rand index, plain numpy
            n = len(x)
            ct = collections.Counter(zip(x.tolist(), yv.tolist()))
            ai = collections.Counter(x.tolist()); bj = collections.Counter(yv.tolist())
            s_ij = sum(v * (v - 1) / 2 for v in ct.values())
            s_a = sum(v * (v - 1) / 2 for v in ai.values())
            s_b = sum(v * (v - 1) / 2 for v in bj.values())
            tot = n * (n - 1) / 2
            exp = s_a * s_b / tot
            ari = (s_ij - exp) / (0.5 * (s_a + s_b) - exp) if 0.5 * (s_a + s_b) != exp else 1.0
            n_gt = len(set(x.tolist())); n_pr = len(set(yv.tolist()))
            scores.append((L, ari, n_gt, n_pr))
            print(f'{a.stem} L{L}: ARI {ari:.3f}   rooms true {n_gt} / pred {n_pr}')
        if scores:
            print(f'{a.stem} MEAN_ARI {sum(s[1] for s in scores)/len(scores):.3f}')
        return

    json.dump(result, open(a.out, 'w'))
    print(f"ROOMS {a.stem}: " + ', '.join(
        f"L{L}:{len(v['rooms'])}" for L, v in result.items()) + f" -> {a.out}")


if __name__ == '__main__':
    main()
