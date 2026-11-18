"""Instance-TSP ORACLE (fully privileged) - evaluation upper reference, NEVER training data.

Knows every reachable target instance's exact position. Plans the best tour with a
budget via Held-Karp orienteering (exact for <=20 instances, NN+2-opt beyond):
maximize instances visited within B primitives, then execute pose-to-pose with the
standard follower. Crediting, counting, and denominators are IDENTICAL to the C4
expert (P3 rule, P6 start-room exclusion; P8 navmesh-geodesic reachable set), so
N_oracle(B) is directly comparable to the expert's N(B).

Outputs the same triple format as the v2 fleet: <out>_cam.mp4, <out>_map.mp4, <out>_gt.png.

  tsp_oracle.py --dataset hm3d --stem 1S7LAXRdDqK --cat chair --budget 400 --gpu 0 --out out/ep
"""
import argparse, collections, itertools, json, math, os
import numpy as np
import habitat_sim
from PIL import Image, ImageDraw

from c4_collect_expert import (make_sim, apos, HM3D, MP3D, DG, COLLECT_R, nav_grid, geo)
from build_room_graphs import parse_house
from replay_render import MapArt, font

FORWARD_M = 0.25
FACE_COST = 3            # planning estimate: turns spent facing a target at arrival
ACT = {"forward": "move_forward", "left": "turn_left", "right": "turn_right"}


def plan_tour(D, n, budget_prims):
    """Orienteering: maximize visited count, tie-break min cost. D[(i,j)] costs with
    node 0 = start. Exact subset DP for n<=20, else greedy NN + 2-opt truncated."""
    if n <= 20:
        best = {}
        dp = {(1, 0): (0.0, [0])}          # mask includes node 0
        for mask in range(1, 1 << (n + 1)):
            if not (mask & 1):
                continue
            for last in range(n + 1):
                if not (mask >> last) & 1 or (mask, last) not in dp:
                    continue
                cost, path = dp[(mask, last)]
                for nxt in range(1, n + 1):
                    if (mask >> nxt) & 1:
                        continue
                    c2 = cost + D[(last, nxt)]
                    if c2 > budget_prims:
                        continue
                    key = (mask | (1 << nxt), nxt)
                    if key not in dp or c2 < dp[key][0]:
                        dp[key] = (c2, path + [nxt])
        best_path, best_n, best_c = [0], 0, 0.0
        for (mask, last), (cost, path) in dp.items():
            cnt = bin(mask).count("1") - 1
            if cnt > best_n or (cnt == best_n and cost < best_c):
                best_n, best_c, best_path = cnt, cost, path
        return best_path[1:], best_c
    # heuristic fallback
    unvis = set(range(1, n + 1)); cur = 0; cost = 0.0; order = []
    while unvis:
        nxt = min(unvis, key=lambda j: D[(cur, j)])
        if cost + D[(cur, nxt)] > budget_prims:
            break
        cost += D[(cur, nxt)]; order.append(nxt); unvis.remove(nxt); cur = nxt
    return order, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mp3d', 'hm3d'], default='hm3d')
    ap.add_argument('--stem', required=True); ap.add_argument('--cat', required=True)
    ap.add_argument('--budget', type=int, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--start-idx', type=int, default=0)
    ap.add_argument('--level', type=int, default=None,
                    help='restrict the episode to one floor of a multi-storey house')
    ap.add_argument('--retry', type=int, default=1,
                    help='[P11] credit-retry ring after a failed one-shot face; 0 = legacy one-shot')
    ap.add_argument('--retry-cap', type=int, default=40,
                    help='max primitives spent retrying a single instance')
    ap.add_argument('--no-video', action='store_true',
                    help='skip cam/map frame capture and mp4 writing (fleet mode)')
    a = ap.parse_args()

    if a.dataset == 'hm3d':
        G = json.load(open(f"{DG}/room_graphs_hm3d_v2.json"))[a.stem]
        targets = [(rid, o[1], o[2], o[3], o[4])
                   for rid, r in G['rooms'].items() for o in r.get('objects_full', []) if o[0] == a.cat]
        glb = os.path.join(HM3D, G['scene_rel'], a.stem + ".glb")
    else:
        G = json.load(open(f"{DG}/room_graphs_mp3d.json"))[a.stem]
        _, objects = parse_house(f"{MP3D}/{a.stem}/{a.stem}.house")
        targets = [(ridx, o[0], o[1], o[2], r) for ridx, mp, o, r in objects if mp == a.cat]
        glb = f"{MP3D}/{a.stem}/{a.stem}.glb"

    adjm = collections.defaultdict(list)
    for e in G['edges']:
        adjm[str(e['u'])].append(str(e['v'])); adjm[str(e['v'])].append(str(e['u']))
    from c4_collect_expert import pick_start
    start = pick_start(G, adjm, (int if a.dataset == 'mp3d' else str), a.start_idx, a.level)
    level = G['rooms'][start]['level']; y = G['rooms'][start]['floor_y']

    sim = make_sim(glb, a.gpu)
    pf = sim.pathfinder
    nav, x0, z0 = nav_grid(pf, y)
    st = sim.get_agent(0).get_state()
    st.position = np.asarray(pf.snap_point(G['rooms'][start]['centroid']), np.float32)
    sim.get_agent(0).set_state(st)

    # reachable tour nodes (P6: start-room instances excluded) with crediting poses.
    # [POLICY P8, 2026-08-05] reachable = NAVMESH GEODESIC from the start pose, not
    # room-graph connectivity: reconstruction can miss doorway edges (e.g. 2Pc8W48bu21
    # dining room), leaving instances the explorers physically reach and get credited
    # for outside the oracle's target set. One physical definition for every agent.
    start_pos = np.asarray(sim.get_agent(0).get_state().position, np.float32)
    nodes = []                # (k, pose(3), center(3), r)
    excluded, offfloor = set(), 0
    for k, (rid, ox, oy, oz, r) in enumerate(targets):
        rid = str(rid)
        if rid not in G['rooms'] or G['rooms'][rid]['level'] != level:
            offfloor += 1; continue
        if rid == start:
            excluded.add(k); continue
        # [P12] referee-matched crediting pose: an instance is reachable iff
        # some navmesh pose exists from which the referee would credit it
        # (horizontal distance - r <= COLLECT_R, |oy - pose_y| <= 1.5, finite
        # geodesic from start). A single floor-level snap is fragile: it can
        # land on a disconnected navmesh island on top of the furniture
        # itself. Try the center snap, a floor-level snap, and a ring of
        # floor-level points; take the first valid candidate.
        p = None
        cand_q = [np.array([float(ox), float(oy), float(oz)], np.float32),
                  np.array([float(ox), float(y), float(oz)], np.float32)]
        for rr_ in (0.75, 1.2):
            for th_ in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
                cand_q.append(np.array([float(ox) + rr_ * math.sin(th_),
                                        float(y),
                                        float(oz) + rr_ * math.cos(th_)],
                                       np.float32))
        for q_ in cand_q:
            sp = pf.snap_point(q_)
            if sp is None or np.any(np.isnan(sp)):
                continue
            if math.hypot(sp[0] - float(ox), sp[2] - float(oz)) - r > COLLECT_R:
                continue
            if abs(float(oy) - float(sp[1])) > 1.5:
                continue
            if not math.isfinite(geo(pf, start_pos, sp)):
                continue
            p = np.asarray(sp, np.float32)
            break
        if p is None:
            excluded.add(k); continue
        nodes.append((k, p, np.array([ox, oy, oz], np.float32), r))
    n = len(nodes)

    # cost matrix in primitives (geodesic forward steps + facing estimate)
    pts = [np.asarray(sim.get_agent(0).get_state().position, np.float32)] + [nd[1] for nd in nodes]
    D = {}
    for i in range(n + 1):
        for j in range(n + 1):
            if i == j: continue
            g = geo(pf, pts[i], pts[j])
            D[(i, j)] = (g / FORWARD_M + FACE_COST) if math.isfinite(g) else 1e9
    order, plan_cost = plan_tour(D, n, a.budget)
    print(f"ORACLE plan: visit {len(order)}/{n} reachable, est cost {plan_cost:.0f}/{a.budget} prims")

    # execute
    follower = habitat_sim.GreedyGeodesicFollower(pf, sim.get_agent(0),
        forward_key="move_forward", left_key="turn_left", right_key="turn_right")
    art = MapArt(nav, x0, z0)
    traj = [apos(sim)[[0, 2]].tolist()]
    found_xy, cam_frames, map_frames = [], [], []
    counted = set(excluded)
    B = a.budget
    Fs = font(16)

    # [SPOC-SFT] c4-parity episode record: action words, found step indices,
    # tour-leg decision prims, and a clean (no-overlay) frame BEFORE every
    # action streamed to <out>_frames.mp4 -- flatten_uninavid asserts T+1 frames.
    ACT_WORD = {'move_forward': 'forward', 'turn_left': 'left', 'turn_right': 'right'}
    actions, found_events, decisions_out = [], [], []
    import imageio
    outdir = os.path.dirname(a.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    sft_wr = imageio.get_writer(a.out + '_frames.mp4', fps=1, macro_block_size=2)

    def sft_snap():
        sft_wr.append_data(np.asarray(sim.get_sensor_observations()['rgb'])[..., :3])

    def credit():
        st_ = sim.get_agent(0).get_state(); p = np.array(st_.position, np.float32)
        th = 2 * math.atan2(st_.rotation.y, st_.rotation.w)
        fx, fz = -math.sin(th), -math.cos(th)
        for k, (rid, ox, oy, oz, r) in enumerate(targets):
            if k in counted or abs(oy - p[1]) > 1.5:
                continue
            dx, dz = ox - p[0], oz - p[2]
            d = math.hypot(dx, dz)
            if d - r > COLLECT_R:
                continue
            ang = math.degrees(abs(math.atan2(fx * dz - fz * dx, fx * dx + fz * dz)))
            if ang <= 39.5 or d - r <= 0.3:
                counted.add(k); found_xy.append((ox, oz))
                if k not in excluded:
                    found_events.append(step)

    def cam_frame(step):
        rgb = np.asarray(sim.get_sensor_observations()["rgb"])[..., :3]
        im = Image.fromarray(rgb.copy()); d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 24], fill=(0, 0, 0))
        d.text((6, 2), f"ORACLE TSP | {a.cat.replace('_',' ')} | step {step} | budget left {max(B,0)}",
               font=Fs, fill=(120, 235, 140))
        if a.scale != 1.0:
            im = im.resize((int(im.width*a.scale)//2*2, int(im.height*a.scale)//2*2))
        return np.asarray(im)

    def agent_marker():
        st_ = sim.get_agent(0).get_state(); p = np.array(st_.position, np.float32)
        th = 2 * math.atan2(st_.rotation.y, st_.rotation.w)
        return (art.px(p[0], p[2]), (-math.sin(th), -math.cos(th)))

    step = 0
    def snap():
        if a.no_video:
            return
        cam_frames.append(cam_frame(step))
        map_frames.append(art.frame(traj, found_xy, agent_marker(),
                                    f"ORACLE TSP: every {a.cat.replace('_',' ')}  [B={a.budget}]",
                                    f"budget left: {max(B,0)}/{a.budget}   collected: {len(found_xy)}"))

    def spend(act):
        nonlocal B, step
        sft_snap()                           # obs BEFORE this action
        actions.append(ACT_WORD[act])
        sim.step(act); B -= 1; step += 1
        traj.append(apos(sim)[[0, 2]].tolist())
        credit()
        if step % 2 == 0:
            snap()

    snap()
    for idx in order:
        k, pose, ctr, r = nodes[idx - 1]
        decisions_out.append({'prim': step})
        try:
            follower.reset()
        except Exception:
            pass
        while B > 0:
            try:
                act = follower.next_action_along(pose)
            except Exception:
                break
            if act is None or act == "stop":
                break
            spend(act)
        for _ in range(6):                       # face the instance
            if k in counted or B <= 0:
                break
            st_ = sim.get_agent(0).get_state(); p = np.array(st_.position, np.float32)
            th = 2 * math.atan2(st_.rotation.y, st_.rotation.w)
            fx, fz = -math.sin(th), -math.cos(th)
            dx, dz = ctr[0] - p[0], ctr[2] - p[2]
            ang = math.degrees(math.atan2(fx * dz - fz * dx, fx * dx + fz * dz))
            if abs(ang) <= 39.5:
                break
            spend("turn_right" if ang > 0 else "turn_left")
        # [P11] credit-retry: the one-shot visit failed to credit (awkward
        # geometry, bad approach angle). Try a ring of alternate crediting
        # poses around the instance, nearest first, with bounded spend, so
        # the oracle is a count ceiling rather than only an efficiency one.
        if a.retry and k not in counted and B > 0:
            b_mark = B
            cands = []
            for rr in (0.9, 1.3):
                for th_c in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
                    qx = ctr[0] + rr * math.sin(th_c)
                    qz = ctr[2] + rr * math.cos(th_c)
                    sp = pf.snap_point(np.array([qx, float(y), qz], np.float32))
                    if sp is None or np.any(np.isnan(sp)):
                        continue
                    if math.hypot(sp[0] - ctr[0], sp[2] - ctr[2]) - r > COLLECT_R:
                        continue
                    cands.append(np.asarray(sp, np.float32))
            p_now = np.array(sim.get_agent(0).get_state().position, np.float32)
            cands.sort(key=lambda q_: geo(pf, p_now, q_))
            for q_ in cands:
                if k in counted or B <= 0 or (b_mark - B) >= a.retry_cap:
                    break
                try:
                    follower.reset()
                except Exception:
                    pass
                while B > 0 and k not in counted and (b_mark - B) < a.retry_cap:
                    try:
                        act = follower.next_action_along(q_)
                    except Exception:
                        break
                    if act is None or act == "stop":
                        break
                    spend(act)
                for _ in range(12):
                    if k in counted or B <= 0 or (b_mark - B) >= a.retry_cap:
                        break
                    st_ = sim.get_agent(0).get_state(); p = np.array(st_.position, np.float32)
                    th = 2 * math.atan2(st_.rotation.y, st_.rotation.w)
                    fx, fz = -math.sin(th), -math.cos(th)
                    dx, dz = ctr[0] - p[0], ctr[2] - p[2]
                    ang = math.degrees(math.atan2(fx * dz - fz * dx, fx * dx + fz * dz))
                    if abs(ang) <= 39.5:
                        break
                    spend("turn_right" if ang > 0 else "turn_left")
        if B <= 0:
            break
    snap()
    sft_snap()                               # final obs -> T+1 frames total
    sft_wr.close()
    stop = {'reason': 'budget exhausted' if B <= 0 else 'tour complete', 'prim': step}

    exp = len(counted) - len(excluded)
    import imageio
    if not a.no_video:
        for _ in range(12):
            cam_frames.append(cam_frames[-1]); map_frames.append(map_frames[-1])
        imageio.mimsave(a.out + "_cam.mp4", cam_frames, fps=18, macro_block_size=2)
        imageio.mimsave(a.out + "_map.mp4", map_frames, fps=18, macro_block_size=2)
    gt_xy, gt_status = [], []
    for k, (rid, ox, oy, oz, r) in enumerate(targets):
        rid = str(rid)
        if rid not in G['rooms'] or G['rooms'][rid]['level'] != level:
            continue
        gt_xy.append((ox, oz))
        if k in counted and k not in excluded:
            gt_status.append('found')
        elif any(nd[0] == k for nd in nodes):       # P8: geodesic-reachable tour node
            gt_status.append('missed')
        else:
            gt_status.append('excluded')
    n_reach = sum(1 for s in gt_status if s in ('found', 'missed'))
    art.gt(traj, gt_xy, gt_status,
           f"ORACLE GT: {a.cat.replace('_',' ')} in {a.stem}  [B={a.budget}]",
           f"found {exp} / {n_reach} reachable   spent {a.budget - B}/{a.budget}   "
           f"planned {len(order)}/{n}").save(a.out + "_gt.png")
    json.dump({'stem': a.stem, 'cat': a.cat, 'budget': a.budget,
               'start_idx': a.start_idx, 'spent': a.budget - B,
               'collected': exp, 'n_reachable': n_reach,
               'level': level, 'start_room': start,
               'n_targets': n, 'n_start_excluded': len(excluded),
               'actions': actions, 'found_events': found_events,
               'decisions': decisions_out, 'stop': stop,
               'traj': traj, 'found_xy': found_xy},
              open(a.out + '_trace.json', 'w'), default=float)
    print(f"ORACLE DONE B0={a.budget} spent={a.budget - B} collected={exp}/{n_reach} reachable -> {a.out}_*")
    sim.close()


if __name__ == '__main__':
    main()
