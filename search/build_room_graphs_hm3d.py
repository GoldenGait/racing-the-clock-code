"""Stage C1-HM3D - room graphs for HM3D-Sem scenes (offline, mostly headless).

HM3D has no room-type labels and no portals, but .semantic.txt DOES ship
instance -> (category, REGION) groupings, and .semantic.glb encodes instance IDs as
texture colors. So:

  rooms       = the shipped semantic regions (no watershed pseudo-rooms needed)
  room type   = observed-object VOTING via the MP3D-counted P(room|object) prior
                (validated on MP3D ground truth: bathroom 96% / bedroom 99% /
                 kitchen 82% top-1; object-generic types are weak -- known)
  objects     = texture-baked mesh parse: per-instance centroid + xz half-extent
  region area = footprint of the region's own floor/rug instances (KD-tree cell
                assignment on the nav grid)
  edges/doors/sweep costs = same navmesh machinery as MP3D (imported)

Coordinates: semantic.glb is z-up; habitat is y-up: (x,y,z) -> (x, z, -y).
GATE per scene: >=90% of parsed object centroids must land inside the navmesh
xz-bounds (+/-1m) after transform, else the scene is rejected loudly.

Navmeshes: HM3D ships none; computed renderer-less (no sensors -> no EGL) and
cached under --navmesh-cache.

  python build_room_graphs_hm3d.py --out room_graphs_hm3d.json --workers 10
"""
from paths import DATA_ROOT
import argparse, collections, glob, json, math, os
import numpy as np
import habitat_sim

from build_room_graphs import (CELL, FORWARD_M, nav_grid, adjacency,
                               room_viewpoints, sweep_cost, geo)

HM3D = os.environ.get("HM3D_ROOT",
    f"{DATA_ROOT}/scene_datasets/versioned_data/hm3d-0.2/hm3d")
NAVMESH_CACHE = f"{DATA_ROOT}/hm3d_navmeshes"
FLOOR_CATS = {"floor", "rug", "carpet", "floor mat", "mat"}
STRUCT = {"wall", "floor", "ceiling", "door", "window", "misc", "objects", "void",
          "unknown", "", "stairs", "railing", "column", "beam", "rug", "carpet",
          "floor mat", "mat", "door frame", "window frame", "doorframe", "unlabeled"}
ASSIGN_R = 1.0       # nav cell -> region if nearest region geometry point within this (m)
                     # [v2/E2] footprints use ALL region geometry; radius widened 0.75->1.0

# raw HM3D label -> prior (mpcat40) vocabulary. Exact matches pass through; these
# rules catch the common variants; unmapped labels are excluded from voting/counts.
SYN = {"couch": "sofa", "l-shaped sofa": "sofa", "sofa chair": "sofa",
       "tv": "tv_monitor", "television": "tv_monitor", "monitor": "tv_monitor",
       "potted plant": "plant", "houseplant": "plant", "plant pot": "plant",
       "flower pot": "plant", "flowerpot": "plant",
       "armchair": "chair", "office chair": "chair", "dining chair": "chair",
       "rocking chair": "chair", "chairs": "chair", "highchair": "chair",
       "toilet bowl": "toilet", "wc": "toilet",
       "bathtub": "bathtub", "bath tub": "bathtub", "tub": "bathtub",
       "kitchen counter": "counter", "countertop": "counter",
       "nightstand": "chest_of_drawers", "night stand": "chest_of_drawers",
       "dresser": "chest_of_drawers", "drawers": "chest_of_drawers",
       "wardrobe": "cabinet", "closet": "cabinet", "cupboard": "cabinet",
       "kitchen cabinet": "cabinet", "shelf": "shelving", "bookshelf": "shelving",
       "shelves": "shelving", "lamp": "lighting", "ceiling lamp": "lighting",
       "chandelier": "lighting", "washbasin": "sink", "wash basin": "sink",
       "basin": "sink", "sink cabinet": "sink", "mirror frame": "mirror",
       "curtains": "curtain", "shower curtain": "curtain", "drapes": "curtain",
       "pillow": "cushion", "cushions": "cushion", "sofa cushion": "cushion",
       "bedside table": "table", "coffee table": "table", "dining table": "table",
       "desk": "table", "side table": "table", "washing machine": "appliances",
       "refrigerator": "appliances", "fridge": "appliances", "oven": "appliances",
       "stove": "appliances", "dishwasher": "appliances", "microwave": "appliances",
       "picture frame": "picture", "painting": "picture", "photo": "picture",
       "poster": "picture", "wall art": "picture", "artwork": "picture"}


def norm_cat(raw, prior):
    n = raw.lower().strip().replace("_", " ")
    if n in SYN:
        return SYN[n]
    u = n.replace(" ", "_")
    return u if u in prior else None


def h2hab(pts):
    """z-up -> y-up, vectorized."""
    p = np.asarray(pts, np.float32)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def parse_semantic(stem_dir, stem, prior):
    """-> (objects, floor_pts, region_pts).
    objects: [(region, cat, center_hab(3), r_xz)] over mapped non-structural cats;
    floor_pts: region -> Nx3 hab points of floor instances (level estimation);
    region_pts: region -> Nx3 hab points of ALL the region's geometry, subsampled to
    the nav grid (v2/E2: walls outline rooms far more densely than sparse floor
    annotations; used for nav-cell footprint assignment)."""
    import trimesh
    txt = os.path.join(stem_dir, stem + ".semantic.txt")
    glb = os.path.join(stem_dir, stem + ".semantic.glb")
    cmap = {}
    for ln in open(txt).readlines()[1:]:
        p = ln.strip().split(",")
        if len(p) >= 4 and len(p[1]) == 6:
            try:
                key = (int(p[1][0:2], 16), int(p[1][2:4], 16), int(p[1][4:6], 16))
            except ValueError:
                continue                     # malformed color line (2 scenes have these)
            cmap[key] = (p[2].strip('"').lower(), p[3])
    sc = trimesh.load(glb, process=False)
    pts_of = collections.defaultdict(list)
    for g in sc.geometry.values():
        try:
            cols = np.asarray(g.visual.to_color().vertex_colors)[:, :3]
        except Exception:
            continue
        uniq, inv = np.unique(cols, axis=0, return_inverse=True)
        V = np.asarray(g.vertices, np.float32)
        for k, c in enumerate(map(tuple, uniq)):
            if c in cmap:
                pts_of[c].append(V[inv == k])
    objects, floor_pts = [], collections.defaultdict(list)
    region_pts = collections.defaultdict(list)
    for c, chunks in pts_of.items():
        raw, reg = cmap[c]
        P = h2hab(np.concatenate(chunks))
        # v2/E2: every annotated point outlines its region -- subsample to the nav
        # grid and keep for footprint assignment (ceilings project onto the room's xz)
        sub = np.unique(np.round(P / CELL).astype(np.int32), axis=0).astype(np.float32) * CELL
        region_pts[reg].append(sub)
        if raw in FLOOR_CATS:
            floor_pts[reg].append(P)
            continue
        cat = norm_cat(raw, prior)
        if cat is None or cat in STRUCT:
            continue
        ctr = P.mean(0)
        r = float(min(2.0, np.abs(P[:, [0, 2]] - ctr[[0, 2]]).max()))
        objects.append((reg, cat, ctr, r))
    floor_pts = {k: np.concatenate(v) for k, v in floor_pts.items()}
    region_pts = {k: np.concatenate(v) for k, v in region_pts.items()}
    return objects, floor_pts, region_pts


def get_navmesh(stem_dir, stem, cache):
    os.makedirs(cache, exist_ok=True)
    nm = os.path.join(cache, stem + ".navmesh")
    if not os.path.exists(nm):
        b = habitat_sim.SimulatorConfiguration()
        b.scene_id = os.path.join(stem_dir, stem + ".glb")
        b.create_renderer = False; b.enable_physics = False
        ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = []
        sim = habitat_sim.Simulator(habitat_sim.Configuration(b, [ag]))
        ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
        ns.agent_radius = 0.18; ns.agent_height = 0.88
        sim.recompute_navmesh(sim.pathfinder, ns)
        sim.pathfinder.save_nav_mesh(nm)
        sim.close()
    pf = habitat_sim.PathFinder(); pf.load_nav_mesh(nm)
    return pf if pf.is_loaded else None


def build_scene(stem_dir, stem, prior):
    objects, floor_pts, region_pts = parse_semantic(stem_dir, stem, prior)
    if not objects:
        return None, "no mapped objects"
    pf = get_navmesh(stem_dir, stem, NAVMESH_CACHE)
    if pf is None:
        return None, "navmesh failed"
    lo, hi = pf.get_bounds()
    ctrs = np.stack([o[2] for o in objects])
    inb = ((ctrs[:, 0] > lo[0] - 1) & (ctrs[:, 0] < hi[0] + 1) &
           (ctrs[:, 2] > lo[2] - 1) & (ctrs[:, 2] < hi[2] + 1)).mean()
    if inb < 0.9:                                    # coordinate-transform gate
        return None, f"transform gate FAIL ({inb:.0%} in bounds)"

    # levels from region floor heights; v2/E2 fallback: regions without floor
    # annotations use the 5th percentile of all their geometry (~floor height)
    reg_y = {r: float(np.median(P[:, 1])) for r, P in floor_pts.items() if len(P)}
    for r, P in region_pts.items():
        if r not in reg_y and len(P):
            reg_y[r] = float(np.percentile(P[:, 1], 5))
    if not reg_y:
        return None, "no floor annotations"
    ys = sorted(set(round(y, 1) for y in reg_y.values()))
    levels = []
    for y in ys:
        if not levels or y - levels[-1][-1] > 0.8:
            levels.append([y])
        else:
            levels[-1].append(y)
    lvl_y = [float(np.mean(g)) for g in levels]
    lvl_of = {r: int(np.argmin([abs(y - ly) for ly in lvl_y])) for r, y in reg_y.items()}

    from scipy.spatial import cKDTree
    rooms, edges_all, level_used, grids = {}, [], set(), {}
    for li, y in enumerate(lvl_y):
        regs = [r for r, l in lvl_of.items() if l == li]
        if not regs:
            continue
        nav, x0, z0 = nav_grid(pf, y)
        if nav.sum() < 20:
            continue
        grids[str(li)] = {"x0": round(float(x0), 4), "z0": round(float(z0), 4),
                          "y": round(float(y), 4), "nx": int(nav.shape[0]), "nz": int(nav.shape[1])}
        pts, labels = [], []
        for r in regs:
            P = region_pts.get(r)                     # v2/E2: ALL geometry, not floor only
            if P is None or not len(P):
                continue
            pts.append(P[:, [0, 2]]); labels += [r] * len(P)
        if not pts:
            continue
        tree = cKDTree(np.concatenate(pts)); labels = np.array(labels)
        lab = np.full(nav.shape, -1, np.int32)
        cells = np.argwhere(nav)
        cw = np.stack([x0 + (cells[:, 0] + .5) * CELL, z0 + (cells[:, 1] + .5) * CELL], 1)
        d, idx = tree.query(cw)
        reg_ids = sorted({r for r in regs})
        rid_num = {r: k for k, r in enumerate(reg_ids)}
        # [v2/E2] UNCAPPED nearest-region assignment: HM3D only annotates objects, so
        # unannotated wall/floor chunks (doorways!) belong to no region -- a radius cap
        # leaves them unlabeled and rooms never touch. Every navigable cell joins its
        # nearest region; bogus through-wall contacts are killed by the edge sanity
        # filter below.
        for (ci, cj), dd, ii in zip(cells, d, idx):
            lab[ci, cj] = rid_num[labels[ii]]
        counts = collections.defaultdict(collections.Counter)
        objs_of = collections.defaultdict(list)
        for reg, cat, ctr, r in objects:
            if reg in rid_num:
                counts[reg][cat] += 1
                objs_of[reg].append([cat, round(float(ctr[0]), 2), round(float(ctr[1]), 2),
                                     round(float(ctr[2]), 2), round(r, 2)])
        for reg in reg_ids:
            k = rid_num[reg]
            cells_r = np.argwhere(lab == k)
            if len(cells_r) < 4:
                continue
            votes = collections.Counter()
            for cat, n in counts[reg].items():
                if cat in prior:
                    for room, p in prior[cat].items():
                        votes[room] += p * n
            rtype = votes.most_common(1)[0][0] if votes else "unknown"
            vps = room_viewpoints(cells_r)
            ci, cj = cells_r.mean(0)
            snap = pf.snap_point([x0 + (ci + .5) * CELL, y, z0 + (cj + .5) * CELL])
            if snap is None or np.any(np.isnan(snap)):
                continue
            rooms[f"{li}_{reg}"] = {
                "type": rtype, "level": li, "floor_y": round(y, 3),
                "centroid": [round(float(v), 3) for v in snap],
                "n_cells": int(len(cells_r)), "area_m2": round(len(cells_r) * CELL * CELL, 2),
                "sweep_cost": sweep_cost(vps), "n_viewpoints": len(vps),
                "counts": dict(counts[reg]), "objects": [o[:1] + o[1:2] + o[3:4] for o in objs_of[reg]],
                "objects_full": objs_of[reg],
                "cells": cells_r.tolist()}   # nav-grid footprint for the C4 sweep
        num_rid = {v: k for k, v in rid_num.items()}
        for (u, v), ptsb in adjacency(nav, lab).items():
            ku, kv = f"{li}_{num_rid[u]}", f"{li}_{num_rid[v]}"
            if ku not in rooms or kv not in rooms or len(ptsb) < 2:
                continue
            mi, mj = np.mean(ptsb, 0)
            snap = pf.snap_point([x0 + (mi + .5) * CELL, y, z0 + (mj + .5) * CELL])
            if snap is None or np.any(np.isnan(snap)):
                continue
            g = geo(pf, rooms[ku]["centroid"], rooms[kv]["centroid"])
            # [v2/E2] edge sanity: a real door means the rooms are navigably close;
            # a through-wall cell contact shows a geodesic detour far beyond euclidean.
            cu, cv = rooms[ku]["centroid"], rooms[kv]["centroid"]
            eu = math.hypot(cu[0] - cv[0], cu[2] - cv[2])
            if not math.isfinite(g) or g > 4.0 * eu + 3.0:
                continue
            edges_all.append({"u": ku, "v": kv, "width_cells": len(ptsb),
                              "door": [round(float(x), 3) for x in snap],
                              "dnav_m": round(g, 2)})
        # [v2/E2] bridge edges: connect navmesh-reachable room components that share no
        # cell contact (scan-blocked doorways force detours). Door = geodesic midpoint.
        lvl_rooms = [rid for rid in rooms if rooms[rid]["level"] == li]
        parent = {r: r for r in lvl_rooms}
        def find(r):
            while parent[r] != r:
                parent[r] = parent[parent[r]]; r = parent[r]
            return r
        for e in edges_all:
            if e["u"] in parent and e["v"] in parent:
                parent[find(e["u"])] = find(e["v"])
        for _ in range(len(lvl_rooms)):
            comps = collections.defaultdict(list)
            for r in lvl_rooms:
                comps[find(r)].append(r)
            if len(comps) <= 1:
                break
            main = max(comps.values(), key=lambda rs: sum(rooms[r]["n_cells"] for r in rs))
            best = None
            for rs in comps.values():
                if rs is main:
                    continue
                for a in rs:
                    for b in main:
                        gg = geo(pf, rooms[a]["centroid"], rooms[b]["centroid"])
                        if math.isfinite(gg) and (best is None or gg < best[0]):
                            best = (gg, a, b)
            if best is None:
                break                                  # remaining components are true islands
            gg, a, b = best
            sp_ = habitat_sim.ShortestPath()
            sp_.requested_start = np.asarray(rooms[a]["centroid"], np.float32)
            sp_.requested_end = np.asarray(rooms[b]["centroid"], np.float32)
            pf.find_path(sp_)
            mid = sp_.points[len(sp_.points) // 2] if len(sp_.points) else rooms[a]["centroid"]
            edges_all.append({"u": a, "v": b, "width_cells": 0, "bridge": True,
                              "door": [round(float(x), 3) for x in mid],
                              "dnav_m": round(gg, 2)})
            parent[find(a)] = find(b)
        level_used.add(li)
    if not rooms:
        return None, "no rooms with cells"
    return {"rooms": rooms, "edges": edges_all, "levels": sorted(level_used),
            "grids": grids, "scene_rel": os.path.relpath(stem_dir, HM3D)}, None


def worker(args):
    stem_dir, stem = args
    prior = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "prior_table.json")))["prior"]
    try:
        data, err = build_scene(stem_dir, stem, prior)
    except Exception as e:
        data, err = None, f"EXC {type(e).__name__}: {e}"
    return stem, data, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="room_graphs_hm3d.json")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--scenes", nargs="*", default=None)
    a = ap.parse_args()
    jobs = []
    # [v2/E1] exclude .basis texture variants -- same building under another name;
    # every semantically-annotated building also ships the plain variant.
    for t in sorted(t for t in glob.glob(f"{HM3D}/*/*/*.semantic.txt")
                    if not t.endswith(".basis.semantic.txt")):
        d = os.path.dirname(t); s = os.path.basename(t)[:-len(".semantic.txt")]
        if a.scenes and s not in a.scenes:
            continue
        if (d, s) not in jobs:
            jobs.append((d, s))
    jobs = sorted(set(jobs))
    print(f"[C1-HM3D] scenes to build: {len(jobs)}")
    out, fails = {}, []
    if a.workers <= 1:
        results = map(worker, jobs)
    else:
        import multiprocessing as mp
        results = mp.Pool(a.workers).imap_unordered(worker, jobs)
    n_rooms = n_edges = 0
    for stem, data, err in results:
        if data is None:
            fails.append((stem, err)); print(f"[C1-HM3D] {stem}: FAIL {err}", flush=True)
            continue
        out[stem] = data
        n_rooms += len(data["rooms"]); n_edges += len(data["edges"])
        print(f"[C1-HM3D] {stem}: rooms={len(data['rooms'])} edges={len(data['edges'])} "
              f"levels={len(data['levels'])}", flush=True)
    json.dump(out, open(a.out, "w"), sort_keys=True)
    print(f"[C1-HM3D] TOTAL scenes={len(out)}/{len(jobs)} rooms={n_rooms} edges={n_edges} "
          f"fails={len(fails)} -> {a.out}")
    tcount = collections.Counter()
    for d in out.values():
        for r in d["rooms"].values():
            tcount[r["type"]] += 1
    print("[C1-HM3D] voted room types:", dict(tcount.most_common(10)))


if __name__ == "__main__":
    main()
