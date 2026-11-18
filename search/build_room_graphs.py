"""Stage C1 - build room-connectivity graphs for all MP3D scenes (headless, no GL).

MP3D .house files ship regions/types/objects but ZERO portals (verified across all 90
scenes: header portal count = 0; 'P' lines are panoramas). Connectivity is therefore
DERIVED from the navmesh: nav-grid cells are assigned to regions by bbox containment,
and two rooms are connected if their cells touch directly or through a short unlabeled
doorway gap. Everything runs off PathFinder + .navmesh -- no Simulator, no EGL.

Per room: type, level, bbox, centroid, nav-cell count, per-mpcat40 object counts n_{v,y},
sweep cost c_v in PRIMITIVES (coverage-viewpoint tour: forward=0.25m steps + 12 turns/vp,
matching how the covtour executor actually spends budget). Per edge: door position (habitat
coords, navmesh-snapped) and boundary width in cells.

Coordinate note: .house is z-up; habitat is y-up. (x,y,z)_house -> (x, z, -y)_habitat
(verified against navmesh bounds).

  python build_room_graphs.py --out room_graphs_mp3d.json --gate-dir $RTC_DATA/roomgraph_gate
"""
from paths import DATA_ROOT
import argparse, collections, glob, json, math, os
import numpy as np
import habitat_sim

MP3D = os.environ.get("MP3D_ROOT", f"{DATA_ROOT}/mp3d/scene_datasets/mp3d")
CELL = 0.25          # nav-grid resolution, matches frontier/covtour experts
FORWARD_M = 0.25
TURNS_360 = 12       # look-around cost per viewpoint (30 deg turns)
VIS_R = 2.0          # viewpoint sight radius (same as covtour_expert)
GAP_CELLS = 8        # max unlabeled-cell bridge between rooms (= 2m doorway gap)
FLOOR_TOL = 1.2      # vertical tolerance when matching cells to region bboxes (m)
STRUCT = {"wall", "floor", "ceiling", "door", "window", "misc", "objects", "void",
          "unknown", "", "stairs", "railing", "column", "beam"}  # not collectible targets

ROOM = {  # region label codes -> names (same table as build_prior.py)
    'a': 'bathroom', 'b': 'bedroom', 'c': 'closet', 'd': 'dining_room',
    'e': 'entryway', 'f': 'family_room', 'g': 'garage', 'h': 'hallway',
    'i': 'library', 'j': 'laundry_room', 'k': 'kitchen', 'l': 'living_room',
    'm': 'meeting_room', 'n': 'lounge', 'o': 'office', 'p': 'porch',
    'r': 'rec_room', 's': 'stairs', 't': 'toilet_room', 'u': 'utility_room',
    'v': 'tv_room', 'w': 'gym', 'x': 'outdoor', 'y': 'balcony', 'z': 'other_room',
    'B': 'bar', 'C': 'classroom', 'D': 'dining_booth', 'S': 'spa', 'Z': 'junk',
}


def h2hab(p):
    """house (z-up) -> habitat (y-up)."""
    x, y, z = p
    return np.array([x, z, -y], np.float32)


def parse_house(path):
    """Return (regions, objects).
    regions: idx -> {level, type, lo, hi (habitat), center}
    objects: list of (region_idx, mpcat40_name, center_habitat, r_xz) where r_xz is the
    object's horizontal obb half-extent (max of the two in-plane radii, capped at 2m)."""
    regions, cat_mp, objects = {}, {}, []
    for ln in open(path, errors='ignore'):
        t = ln.split()
        if not t:
            continue
        if t[0] == 'R':
            lo = h2hab([float(t[9]), float(t[10]), float(t[11])])
            hi = h2hab([float(t[12]), float(t[13]), float(t[14])])
            regions[int(t[1])] = {
                'level': int(t[2]), 'type': ROOM.get(t[5], 'other_room'),
                'lo': np.minimum(lo, hi), 'hi': np.maximum(lo, hi),
                'center': h2hab([float(t[6]), float(t[7]), float(t[8])])}
        elif t[0] == 'C':
            if len(t) >= 6 and t[4].lstrip('-').isdigit():
                cat_mp[t[1]] = t[5]
        elif t[0] == 'O':
            ridx, mp = int(t[2]), cat_mp.get(t[3])
            if ridx >= 0 and mp:
                # O ... px py pz  a0(3) a1(3)  r0 r1 r2: r0/r1 are the in-plane half-extents
                r_xz = min(2.0, max(float(t[13]), float(t[14]))) if len(t) >= 16 else 0.5
                objects.append((ridx, mp, h2hab([float(t[4]), float(t[5]), float(t[6])]), r_xz))
    return regions, objects


def floor_heights(regions):
    """level -> floor y (habitat) = median of region bbox floor per level."""
    per = collections.defaultdict(list)
    for r in regions.values():
        per[r['level']].append(float(r['lo'][1]))
    return {lv: float(np.median(v)) for lv, v in per.items()}


def nav_grid(pf, y):
    """(nav bool grid [nx,nz], x0, z0). Cell (i,j) center = (x0+(i+.5)*CELL, z0+(j+.5)*CELL)."""
    td = pf.get_topdown_view(CELL, y)     # [rows=z, cols=x]
    nav = np.ascontiguousarray(td.T)      # [x, z]
    lo, _ = pf.get_bounds()
    return nav, float(lo[0]), float(lo[2])


def assign_cells(nav, x0, z0, y, regions, level):
    """Assign each nav cell to a region idx on this level (-1 = unlabeled).
    Containment in xz bbox (small margin); ties -> smallest xz area."""
    cand = [(idx, r) for idx, r in regions.items()
            if r['level'] == level and r['lo'][1] - FLOOR_TOL <= y <= r['hi'][1] + FLOOR_TOL]
    lab = np.full(nav.shape, -1, np.int32)
    order = sorted(cand, key=lambda kv: -float((kv[1]['hi'][0]-kv[1]['lo'][0])*(kv[1]['hi'][2]-kv[1]['lo'][2])))
    m = 0.05
    for idx, r in order:                  # big rooms first; small rooms overwrite on overlap
        i0 = max(0, int((r['lo'][0]-m - x0)/CELL)); i1 = min(nav.shape[0], int((r['hi'][0]+m - x0)/CELL)+1)
        j0 = max(0, int((r['lo'][2]-m - z0)/CELL)); j1 = min(nav.shape[1], int((r['hi'][2]+m - z0)/CELL)+1)
        sl = lab[i0:i1, j0:j1]
        sl[nav[i0:i1, j0:j1]] = idx
    return lab


def adjacency(nav, lab):
    """Edges between rooms: direct cell adjacency, or a straight bridge of <=GAP_CELLS
    unlabeled nav cells (doorframes often fall outside region bboxes). Returns
    {(u,v): [boundary midpoints (i,j) float]}."""
    edges = collections.defaultdict(list)
    nx, nz = nav.shape
    for i in range(nx):
        for j in range(nz):
            if not nav[i, j]:
                continue
            a = lab[i, j]
            for di, dj in ((1, 0), (0, 1)):
                ii, jj = i + di, j + dj
                if ii >= nx or jj >= nz or not nav[ii, jj]:
                    continue
                b = lab[ii, jj]
                if a >= 0 and b >= 0 and a != b:
                    edges[(min(a, b), max(a, b))].append(((i+ii)/2, (j+jj)/2))
                elif (a >= 0) != (b >= 0):      # room touching unlabeled: walk the gap
                    si, sj, room = (ii, jj, a) if a >= 0 else (i, j, b)
                    gi, gj = si, sj
                    for step in range(1, GAP_CELLS + 1):
                        gi, gj = gi + di, gj + dj
                        if not (0 <= gi < nx and 0 <= gj < nz) or not nav[gi, gj]:
                            break
                        c = lab[gi, gj]
                        if c >= 0:
                            if c != room:
                                edges[(min(room, c), max(room, c))].append(((si+gi)/2, (sj+gj)/2))
                            break
    return edges


def geo(pf, a, b):
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, np.float32); sp.requested_end = np.asarray(b, np.float32)
    return sp.geodesic_distance if pf.find_path(sp) else math.inf


def room_viewpoints(cells):
    """Greedy set-cover of a room's cells by VIS_R discs (cell-space; room interiors
    rarely self-occlude at this scale, so no LoS test -- keeps C1 headless-cheap)."""
    cells = list(map(tuple, cells))
    if not cells:
        return []
    cset = set(cells)
    r = int(VIS_R / CELL); r2 = r * r
    disc = [(a, b) for a in range(-r, r+1) for b in range(-r, r+1) if a*a + b*b <= r2]
    cover = {c: {(c[0]+a, c[1]+b) for a, b in disc if (c[0]+a, c[1]+b) in cset} for c in cells}
    uncovered, vps = set(cells), []
    while uncovered:
        best = max(cells, key=lambda c: len(cover[c] & uncovered))
        gain = cover[best] & uncovered
        if not gain:
            break
        vps.append(best); uncovered -= gain
    return vps


def sweep_cost(vps):
    """Primitives to sweep a room: NN tour through its viewpoints (euclidean, cell space)
    + TURNS_360 look-around per viewpoint. Matches covtour execution accounting."""
    if not vps:
        return 0
    cost = len(vps) * TURNS_360
    rem = vps[1:]; cur = vps[0]
    while rem:
        nxt = min(rem, key=lambda v: (v[0]-cur[0])**2 + (v[1]-cur[1])**2)
        cost += int(round(math.hypot(nxt[0]-cur[0], nxt[1]-cur[1]) * CELL / FORWARD_M))
        rem.remove(nxt); cur = nxt
    return cost


def build_scene(stem):
    house = f"{MP3D}/{stem}/{stem}.house"
    navmesh = f"{MP3D}/{stem}/{stem}.navmesh"
    regions, objects = parse_house(house)
    pf = habitat_sim.PathFinder(); pf.load_nav_mesh(navmesh)
    if not pf.is_loaded:
        return None
    counts = collections.defaultdict(collections.Counter)
    objs_of = collections.defaultdict(list)   # per-room [mp, x, z] for door-visibility voting
    for ridx, mp, c, _r in objects:
        if mp not in STRUCT:
            counts[ridx][mp] += 1
            objs_of[ridx].append([mp, round(float(c[0]), 2), round(float(c[2]), 2)])

    rooms, edges_all, floors = {}, [], floor_heights(regions)
    gate = {}
    for level, y in sorted(floors.items()):
        nav, x0, z0 = nav_grid(pf, y)
        if nav.sum() < 20:
            continue
        lab = assign_cells(nav, x0, z0, y, regions, level)
        edges = adjacency(nav, lab)
        cw = lambda i, j: (x0 + (i + .5) * CELL, z0 + (j + .5) * CELL)
        for idx in sorted({int(v) for v in np.unique(lab) if v >= 0}):
            cells = np.argwhere(lab == idx)
            if len(cells) < 4:
                continue
            vps = room_viewpoints(cells)
            ci, cj = cells.mean(0)
            cxz = cw(ci, cj)
            snap = pf.snap_point([cxz[0], y, cxz[1]])
            rooms[idx] = {
                'type': regions[idx]['type'], 'level': level, 'floor_y': round(y, 3),
                'centroid': [round(float(snap[0]), 3), round(float(snap[1]), 3), round(float(snap[2]), 3)],
                'n_cells': int(len(cells)), 'area_m2': round(len(cells) * CELL * CELL, 2),
                'sweep_cost': sweep_cost(vps), 'n_viewpoints': len(vps),
                'counts': dict(counts.get(idx, {})), 'objects': objs_of.get(idx, [])}
        for (u, v), pts in edges.items():
            if u not in rooms or v not in rooms or len(pts) < 2:
                continue        # require a >=2-cell-wide contact to call it a door
            mi, mj = np.mean(pts, 0)
            dx, dz = cw(mi, mj)
            snap = pf.snap_point([dx, y, dz])
            edges_all.append({'u': int(u), 'v': int(v), 'width_cells': len(pts),
                'door': [round(float(snap[0]), 3), round(float(snap[1]), 3), round(float(snap[2]), 3)]})
        gate[level] = (nav, lab, x0, z0, y)
    return {'rooms': rooms, 'edges': edges_all, 'levels': sorted(floors)}, gate, pf


def render_gate(stem, data, gate, out_dir):
    """Overlay: region colors on nav grid + edges + doors. Pure PIL, no GL."""
    from PIL import Image, ImageDraw, ImageFont
    fp = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    F = ImageFont.truetype(fp, 14) if os.path.exists(fp) else ImageFont.load_default()
    MS = 6
    rng = np.random.default_rng(0)
    palette = {idx: tuple(int(c) for c in rng.integers(70, 230, 3)) for idx in data['rooms']}
    for level, (nav, lab, x0, z0, y) in gate.items():
        nx, nz = nav.shape
        im = np.full((nz * MS, nx * MS, 3), 15, np.uint8)
        for (i, j) in np.argwhere(nav):
            col = palette.get(int(lab[i, j]), (55, 58, 66))
            im[(nz-1-j)*MS:(nz-j)*MS, i*MS:(i+1)*MS] = col
        pim = Image.fromarray(im); d = ImageDraw.Draw(pim)
        px = lambda wx, wz: (int((wx - x0) / CELL * MS), int((nz - (wz - z0) / CELL) * MS))
        for e in data['edges']:
            ru, rv = data['rooms'][e['u']], data['rooms'][e['v']]
            if ru['level'] != level or rv['level'] != level:
                continue
            d.line([px(ru['centroid'][0], ru['centroid'][2]), px(rv['centroid'][0], rv['centroid'][2])],
                   fill=(255, 255, 255), width=2)
            dx, dy = px(e['door'][0], e['door'][2])
            d.ellipse([dx-4, dy-4, dx+4, dy+4], fill=(255, 60, 60))
        for idx, r in data['rooms'].items():
            if r['level'] != level:
                continue
            cx, cy = px(r['centroid'][0], r['centroid'][2])
            d.ellipse([cx-5, cy-5, cx+5, cy+5], fill=(255, 255, 255), outline=(0, 0, 0))
            d.text((cx+6, cy-8), f"{r['type']}", font=F, fill=(255, 255, 0))
        pim.save(os.path.join(out_dir, f"{stem}_L{level}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='room_graphs_mp3d.json')
    ap.add_argument('--gate-dir', default='')
    ap.add_argument('--gate-scenes', type=int, default=5)
    ap.add_argument('--scenes', nargs='*', default=None)
    a = ap.parse_args()

    stems = a.scenes or sorted(os.path.basename(p) for p in glob.glob(f"{MP3D}/*") if os.path.isdir(p))
    out, n_rooms, n_edges = {}, 0, 0
    if a.gate_dir:
        os.makedirs(a.gate_dir, exist_ok=True)
    for k, stem in enumerate(stems):
        r = build_scene(stem)
        if r is None:
            print(f"[C1] {stem}: navmesh load FAILED"); continue
        data, gate, pf = r
        # dnav between connected rooms (centroid->centroid geodesic, meters)
        for e in data['edges']:
            g = geo(pf, data['rooms'][e['u']]['centroid'], data['rooms'][e['v']]['centroid'])
            e['dnav_m'] = round(g, 2) if math.isfinite(g) else -1
        out[stem] = {'rooms': {str(i): v for i, v in data['rooms'].items()},
                     'edges': data['edges'], 'levels': data['levels']}
        n_rooms += len(data['rooms']); n_edges += len(data['edges'])
        if a.gate_dir and k < a.gate_scenes:
            render_gate(stem, data, gate, a.gate_dir)
        print(f"[C1] {stem}: rooms={len(data['rooms'])} edges={len(data['edges'])} levels={len(data['levels'])}", flush=True)
    json.dump(out, open(a.out, 'w'), indent=1, sort_keys=True)
    print(f"[C1] TOTAL scenes={len(out)} rooms={n_rooms} edges={n_edges} -> {a.out}")


if __name__ == '__main__':
    main()
