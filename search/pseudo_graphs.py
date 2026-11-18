"""Room graphs for the 592-scene unlabelled expansion: watershed pseudo-rooms
(annot_rooms output) + audited auto-annotated objects (annot_detect/fuse output,
human corrections applied) -> the exact room_graphs_hm3d_v2 schema, so
c4_collect_expert runs on these scenes unchanged (point it via GRAPHS_HM3D).

Room types come from the same prior vote the v2 builder uses -- with only the 6
detected categories the vote is sparse, and object-free rooms land on 'unknown',
which the transition bank already contains. Renderer-less sim throughout (no
EGL), so this parallelizes on CPU under the running benchmarks.

  python pseudo_graphs.py --scenes $RTC_DATA/roll592_scenes.txt \
      --out room_graphs_hm3d_p592.json --procs 16 [--stems A,B]
"""
from paths import DATA_ROOT
import argparse
import collections
import json
import math
import multiprocessing as mp
import os

import numpy as np
import habitat_sim

from build_room_graphs import CELL, nav_grid, adjacency, room_viewpoints, sweep_cost, geo
from annot_rooms import make_sim
from c4_collect_expert import HM3D

DG = os.path.dirname(os.path.abspath(__file__))
ROOMS_DIR = f'{DATA_ROOT}/rooms592'
OBJ_DIR = f'{DATA_ROOT}/annot/obj'
CORRECTIONS = f'{DATA_ROOT}/annot/corrections.json'
OBJ_LEVEL_DY = (-1.0, 3.0)     # object center height above floor to belong to a level
OBJ_SNAP_M = 3.0               # farthest an object may sit from its room's nav cells


def load_objects(stem):
    """fused instances with the human audit's corrections (1-based indices)."""
    p = os.path.join(OBJ_DIR, stem + '.json')
    if not os.path.exists(p):
        return None
    inst = json.load(open(p))
    corr = json.load(open(CORRECTIONS)).get(stem, {})
    drop = {i - 1 for i in corr.get('drop', [])}
    for pair in corr.get('merge', []):
        keep, gone = pair[0] - 1, pair[1] - 1
        a, b = inst[keep], inst[gone]
        wa, wb = a['n_views'], b['n_views']
        a['center'] = [(x * wa + y * wb) / (wa + wb) for x, y in zip(a['center'], b['center'])]
        a['r'] = max(a['r'], b['r']); a['n_views'] = wa + wb
        drop.add(gone)
    return [o for i, o in enumerate(inst) if i not in drop]


def build_scene(stem, rel, prior):
    rooms_file = os.path.join(ROOMS_DIR, stem + '.json')
    if not os.path.exists(rooms_file):
        return None, 'no rooms file'
    seg = json.load(open(rooms_file))
    if not seg:
        return None, 'empty segmentation'
    objects = load_objects(stem)
    if objects is None:
        return None, 'no obj file'

    glb_dir = os.path.join(HM3D, rel)
    glb = next((p for p in (os.path.join(glb_dir, stem + '.glb'),
                            os.path.join(glb_dir, stem + '.basis.glb'))
                if os.path.exists(p)), None)
    if glb is None:
        return None, 'no glb'
    sim = make_sim(glb, 0)
    pf = sim.pathfinder

    levels = sorted(seg, key=int)
    lvl_y = {L: seg[L]['y'] for L in levels}

    # object -> level by height above floor, then -> room by nav-cell membership
    obj_of = collections.defaultdict(list)
    for o in objects:
        ox, oy, oz = o['center']
        cand = [(oy - y, L) for L, y in lvl_y.items()
                if OBJ_LEVEL_DY[0] <= oy - y <= OBJ_LEVEL_DY[1]]
        if not cand:
            continue
        _, L = min(cand, key=lambda t: abs(t[0] - 0.75))
        obj_of[L].append(o)

    rooms, edges_all, grids, level_used = {}, [], {}, set()
    for L in levels:
        sv = seg[L]
        y, x0, z0, nx, nz = sv['y'], sv['x0'], sv['z0'], sv['nx'], sv['nz']
        nav, x0r, z0r = nav_grid(pf, y)
        if abs(x0r - x0) > 0.5 or abs(z0r - z0) > 0.5 or nav.shape != (nx, nz):
            # same navmesh settings as annot_rooms, so drift means the recompute
            # diverged (asset changed since segmentation) -- flag, don't guess
            sim.close()
            return None, f'grid drift L{L}: ({x0:.2f},{z0:.2f},{nx},{nz}) vs ' \
                         f'({x0r:.2f},{z0r:.2f},{nav.shape[0]},{nav.shape[1]})'
        li = int(L)
        grids[str(li)] = {'x0': round(float(x0), 4), 'z0': round(float(z0), 4),
                          'y': round(float(y), 4), 'nx': nx, 'nz': nz}
        lab = np.full((nx, nz), -1, np.int32)
        for k, room in enumerate(sv['rooms']):
            for (i, j) in room['cells']:
                lab[i, j] = k

        # objects of this level -> containing room, else nearest room cells
        labelled = np.argwhere(lab >= 0)
        counts = collections.defaultdict(collections.Counter)
        objs_of = collections.defaultdict(list)
        for o in obj_of.get(L, []):
            ox, oy, oz = o['center']
            i = int(round((ox - x0) / CELL - .5)); j = int(round((oz - z0) / CELL - .5))
            if 0 <= i < nx and 0 <= j < nz and lab[i, j] >= 0:
                k = int(lab[i, j])
            else:
                d2 = ((labelled[:, 0] - i) ** 2 + (labelled[:, 1] - j) ** 2)
                b = int(d2.argmin())
                if math.sqrt(float(d2[b])) * CELL > OBJ_SNAP_M:
                    continue
                k = int(lab[labelled[b, 0], labelled[b, 1]])
            counts[k][o['cat']] += 1
            objs_of[k].append([o['cat'], round(ox, 2), round(oy, 2), round(oz, 2),
                               round(o['r'], 2)])

        for k, room in enumerate(sv['rooms']):
            cells_r = np.asarray(room['cells'], np.int32)
            if len(cells_r) < 4:
                continue
            votes = collections.Counter()
            for cat, n in counts[k].items():
                if cat in prior:
                    for rt, p in prior[cat].items():
                        votes[rt] += p * n
            rtype = votes.most_common(1)[0][0] if votes else 'unknown'
            vps = room_viewpoints(cells_r)
            ci, cj = cells_r.mean(0)
            snap = pf.snap_point([x0 + (ci + .5) * CELL, y, z0 + (cj + .5) * CELL])
            if snap is None or np.any(np.isnan(snap)):
                continue
            rooms[f'{li}_{k}'] = {
                'type': rtype, 'level': li, 'floor_y': round(y, 3),
                'centroid': [round(float(v), 3) for v in snap],
                'n_cells': int(len(cells_r)), 'area_m2': round(len(cells_r) * CELL * CELL, 2),
                'sweep_cost': sweep_cost(vps), 'n_viewpoints': len(vps),
                'counts': dict(counts[k]), 'objects': [[o[0], o[1], o[3]] for o in objs_of[k]],
                'objects_full': objs_of[k], 'cells': cells_r.tolist()}

        for (u, v), pts in adjacency(nav, lab).items():
            ku, kv = f'{li}_{u}', f'{li}_{v}'
            if ku not in rooms or kv not in rooms or len(pts) < 2:
                continue
            mi, mj = np.mean(pts, 0)
            snap = pf.snap_point([x0 + (mi + .5) * CELL, y, z0 + (mj + .5) * CELL])
            if snap is None or np.any(np.isnan(snap)):
                continue
            g = geo(pf, rooms[ku]['centroid'], rooms[kv]['centroid'])
            cu, cv = rooms[ku]['centroid'], rooms[kv]['centroid']
            eu = math.hypot(cu[0] - cv[0], cu[2] - cv[2])
            if not math.isfinite(g) or g > 4.0 * eu + 3.0:    # through-wall contact
                continue
            edges_all.append({'u': ku, 'v': kv, 'width_cells': len(pts),
                              'door': [round(float(x), 3) for x in snap],
                              'dnav_m': round(g, 2)})

        # bridge navmesh-reachable components that share no cell contact (v2/E2)
        lvl_rooms = [r for r in rooms if rooms[r]['level'] == li]
        parent = {r: r for r in lvl_rooms}
        def find(r):
            while parent[r] != r:
                parent[r] = parent[parent[r]]; r = parent[r]
            return r
        for e in edges_all:
            if e['u'] in parent and e['v'] in parent:
                parent[find(e['u'])] = find(e['v'])
        for _ in range(len(lvl_rooms)):
            comps = collections.defaultdict(list)
            for r in lvl_rooms:
                comps[find(r)].append(r)
            if len(comps) <= 1:
                break
            main = max(comps.values(), key=lambda rs: sum(rooms[r]['n_cells'] for r in rs))
            best = None
            for rs in comps.values():
                if rs is main:
                    continue
                for a in rs:
                    for b in main:
                        gg = geo(pf, rooms[a]['centroid'], rooms[b]['centroid'])
                        if math.isfinite(gg) and (best is None or gg < best[0]):
                            best = (gg, a, b)
            if best is None:
                break
            gg, a, b = best
            sp = habitat_sim.ShortestPath()
            sp.requested_start = np.asarray(rooms[a]['centroid'], np.float32)
            sp.requested_end = np.asarray(rooms[b]['centroid'], np.float32)
            pf.find_path(sp)
            mid = sp.points[len(sp.points) // 2] if len(sp.points) else rooms[a]['centroid']
            edges_all.append({'u': a, 'v': b, 'width_cells': 0, 'bridge': True,
                              'door': [round(float(x), 3) for x in mid],
                              'dnav_m': round(gg, 2)})
            parent[find(a)] = find(b)
        level_used.add(li)
    sim.close()
    if not rooms:
        return None, 'no rooms'
    return {'rooms': rooms, 'edges': edges_all, 'levels': sorted(level_used),
            'grids': grids, 'scene_rel': rel}, None


def worker(args):
    stem, rel, prior = args
    try:
        data, err = build_scene(stem, rel, prior)
    except Exception as e:
        data, err = None, f'{type(e).__name__}: {e}'[:200]
    return stem, data, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default=f'{DATA_ROOT}/roll592_scenes.txt')
    ap.add_argument('--out', default=os.path.join(DG, 'room_graphs_hm3d_p592.json'))
    ap.add_argument('--procs', type=int, default=16)
    ap.add_argument('--stems', default='', help='comma list: build only these (pilot)')
    a = ap.parse_args()

    prior = json.load(open(os.path.join(DG, 'prior_table.json')))['prior']
    scenes = []
    for line in open(a.scenes):
        t = line.split()
        stem, rel = t[-2], t[-1]
        if not a.stems or stem in a.stems.split(','):
            scenes.append((stem, rel, prior))

    out, errs = {}, {}
    with mp.Pool(a.procs, maxtasksperchild=8) as pool:
        for n, (stem, data, err) in enumerate(pool.imap_unordered(worker, scenes), 1):
            if data is not None:
                out[stem] = data
            else:
                errs[stem] = err
            if n % 25 == 0 or n == len(scenes):
                print(f'{n}/{len(scenes)} built={len(out)} err={len(errs)}', flush=True)

    json.dump(out, open(a.out, 'w'))
    nr = sum(len(g['rooms']) for g in out.values())
    ne = sum(len(g['edges']) for g in out.values())
    no = sum(len(r['objects_full']) for g in out.values() for r in g['rooms'].values())
    print(f'WROTE {a.out}: {len(out)} scenes, {nr} rooms, {ne} edges, {no} placed objects')
    for stem, e in sorted(errs.items()):
        print(f'  SKIP {stem}: {e}')


if __name__ == '__main__':
    main()
