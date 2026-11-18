"""Stage C4 (examples) - planner-driven COLLECTION expert, executed in habitat.

Room ordering = the C3 empirical Bellman planner (replaces covtour's tour_order,
extension Point 1). Loop (base doc S11): observe frontier -> Qhat for each candidate
(voted type + live dnav from the agent's actual pose) -> enter argmax -> sweep its
coverage viewpoints -> count NEW target instances within reach (Point 7) -> decrement
the budget by primitives actually spent -> replan. STOP when no candidate is feasible
under the conservative bound d_bar = dnav + cmax(voted type) (Points 2 & 4).

Leak firewall at execution: candidates are scored ONLY from (voted type via door
glimpse, observed graph, dnav); true contents/labels of unentered rooms are never
read for scoring. The true graph is used only to reveal a room once ENTERED.

Films [RGB | top-down] with budget countdown + collected count.

  sg render -c "... c4_collect_expert.py --stem 1pXnuDYAj8r --cat chair --budget 600 \
      --out $RTC_DATA/c4_vids/chair_B600.mp4"
"""
from paths import DATA_ROOT
import argparse, collections, json, math, os
import numpy as np
import habitat_sim
import magnum as mn
from PIL import Image, ImageDraw, ImageFont

import budget_planner as bp
from build_transition_bank import voted_type, edges_of, context_of
from build_room_graphs import MP3D, CELL, parse_house, floor_heights, nav_grid, assign_cells, room_viewpoints

CAM_H, HFOV, W, H = 0.88, 79.0, 640, 480   # overridden by --cam-h/--hfov/--res (robot: 0.35, 69, 640x360)
SWEEP_ONCE = False       # [P15] one look-around per room at its best-covering viewpoint
SWEEP_N = [0]            # look-arounds so far, to alternate scan direction across rooms
FORWARD_M, TURN_DEG = 0.25, 30.0
COLLECT_R = 1.5          # instance counted when agent passes within this (m, xz)
TURNS_360 = 12
R_VIS = 5.0              # [P9 pilot, 2026-08-07] sight range for approach actions (m)
LOS_RATIO = 1.35         # geodesic/euclid <= this ~ straight walkable line (occlusion proxy)
PANEL_H = 480
DG = os.path.dirname(os.path.abspath(__file__))


def font(sz):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()
FB, FS = font(24), font(17)


def make_sim(glb, gpu):
    b = habitat_sim.SimulatorConfiguration(); b.scene_id = glb; b.gpu_device_id = gpu; b.enable_physics = False
    s = habitat_sim.CameraSensorSpec(); s.uuid = "rgb"; s.sensor_type = habitat_sim.SensorType.COLOR
    s.resolution = [H, W]; s.position = mn.Vector3(0, CAM_H, 0); s.hfov = HFOV
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [s]
    ag.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_M)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=TURN_DEG))}
    sim = habitat_sim.Simulator(habitat_sim.Configuration(b, [ag]))
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults(); ns.agent_radius = 0.18; ns.agent_height = CAM_H
    sim.recompute_navmesh(sim.pathfinder, ns)
    return sim


def geo(pf, a, b):
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, np.float32); sp.requested_end = np.asarray(b, np.float32)
    return sp.geodesic_distance if pf.find_path(sp) else math.inf


def apos(sim):
    return np.array(sim.get_agent(0).get_state().position, np.float32)


class Recorder:
    def __init__(self, sim, nav, x0, z0, y, cat, budget, label):
        self.sim, self.nav, self.x0, self.z0, self.y = sim, nav, x0, z0, y
        self.cat, self.B0, self.label = cat, budget, label
        self.traj, self.frames, self.found, self.actions = [apos(sim)[[0, 2]].tolist()], [], [], []
        self.nx, self.nz = nav.shape
        self.MS = max(6, min(18, 900 // max(self.nx, self.nz)))

    def px(self, wx, wz):
        return (int((wx - self.x0) / CELL * self.MS), int((self.nz - (wz - self.z0) / CELL) * self.MS))

    def snap(self, B, note=""):
        rgb = np.asarray(self.sim.get_sensor_observations()["rgb"])[..., :3]
        im = np.full((self.nz * self.MS, self.nx * self.MS, 3), 15, np.uint8)
        for (i, j) in np.argwhere(self.nav):
            im[(self.nz-1-j)*self.MS:(self.nz-j)*self.MS, i*self.MS:(i+1)*self.MS] = (66, 72, 86)
        pim = Image.fromarray(im); d = ImageDraw.Draw(pim)
        if len(self.traj) >= 2:
            d.line([self.px(x, z) for x, z in self.traj], fill=(90, 175, 255), width=5)
        for (ox, oz) in self.found:
            tx, ty = self.px(ox, oz)
            d.polygon([(tx,ty-13),(tx+4,ty-4),(tx+13,ty-4),(tx+5,ty+3),(tx+8,ty+13),(tx,ty+6),
                       (tx-8,ty+13),(tx-5,ty+3),(tx-13,ty-4),(tx-4,ty-4)], fill=(255, 208, 0), outline=(0, 0, 0))
        p = apos(self.sim); q = self.sim.get_agent(0).get_state().rotation
        th = 2 * math.atan2(q.y, q.w); fx, fz = -math.sin(th), -math.cos(th)
        ax, ay = self.px(p[0], p[2])
        d.line([(ax, ay), (ax + int(fx*22), ay - int(fz*22))], fill=(255, 55, 55), width=6)
        d.ellipse([ax-6, ay-6, ax+6, ay+6], fill=(255, 55, 55))
        mp = pim.resize((max(1, int(pim.width * PANEL_H / pim.height)), PANEL_H))
        full = Image.fromarray(np.concatenate([rgb[:PANEL_H], np.asarray(mp)[:PANEL_H]], axis=1))
        D = ImageDraw.Draw(full); D.rectangle([0, 0, full.width, 60], fill=(0, 0, 0))
        D.text((8, 4), f"collect every {self.cat.replace('_',' ')}   [{self.label}]", font=FB, fill=(255, 208, 0))
        D.text((8, 36), f"budget left: {max(B,0)}/{self.B0}   collected: {len(self.found)}   {note}",
               font=FS, fill=(160, 255, 160) if self.found else (200, 205, 215))
        self.frames.append(np.asarray(full))


HM3D = os.environ.get("HM3D_ROOT",
    f"{DATA_ROOT}/scene_datasets/versioned_data/hm3d-0.2/hm3d")


def pick_start(G, adj, rkey, start_idx=0, level=None):
    """start_idx 0 reproduces the original rule exactly; k>=1 walks a deterministic
    ranking (hallway hubs in id order, then all rooms by descending degree, id) with
    earlier picks removed -- for multi-start evaluation protocols.

    level: restrict the start (and hence the whole episode, since the planner works
    within the start room's level) to one floor. Needed for multi-floor houses where
    every floor is its own episode."""
    rooms = {i: r for i, r in G['rooms'].items() if level is None or r['level'] == level}
    if not rooms:
        raise KeyError(f'no rooms on level {level}')
    first = next((i for i, r in sorted(rooms.items(), key=lambda kv: rkey(kv[0]))
                  if r['type'] == 'hallway' and len(adj[i]) >= 4),
                 max(rooms, key=lambda i: len(adj[i])))
    if start_idx == 0:
        return first
    ranked = [i for i, r in sorted(rooms.items(), key=lambda kv: rkey(kv[0]))
              if r['type'] == 'hallway' and len(adj[i]) >= 4]
    ranked += [i for i in sorted(rooms, key=lambda i: (-len(adj[i]), str(i))) if i not in ranked]
    ranked = [i for i in ranked if i != first]
    return ranked[min(start_idx - 1, len(ranked) - 1)]


def run(stem, cat, budget, gpu, out, seed=0, trace_dir='', dataset='mp3d', start_idx=0, bank_file=None, policy='bellman',
        approach=False, earlystop=0.0, level=None, anytime=False, resweep=False):
    req_level = level
    if dataset == 'hm3d':
        graphs = json.load(open(os.environ.get('GRAPHS_HM3D', f"{DG}/room_graphs_hm3d_v2.json")))  # [v2] deduped, bridged; env override for the p592 expansion
        bank = json.load(open(bank_file or f"{DG}/transition_bank_hm3d_v2.json"))
    else:
        graphs = json.load(open(os.environ.get('GRAPHS_MP3D', f"{DG}/room_graphs_mp3d.json")))   # [P15] env override, like GRAPHS_HM3D
        bank = json.load(open(bank_file or os.environ.get('BANK_MP3D', f"{DG}/transition_bank.json")))
    prior = json.load(open(f"{DG}/prior_table.json"))['prior']
    G = graphs[stem]; adj = edges_of(G)
    pl = None
    if policy == 'bellman':
        fold = bank['meta']['fold_of'].get(stem)  # None => scene absent from bank (eval-free): use ALL folds
        pl = bp.Planner(bank['transitions'], plan_fold=fold, y=cat)
    import random as _random
    brng = _random.Random(f"{stem}|{cat}|{start_idx}|{budget}")   # deterministic baseline choices
    SWEEP_N[0] = brng.randint(0, 1)      # [P15] first scan direction random per episode, not always left
    TIE_DEG = 150.0                      # [P15] rotations beyond this are direction-ambiguous: pick at random

    # conservative sweep bound per voted type: 95th pct sweep cost across ALL scenes (Point 4)
    sw = collections.defaultdict(list)
    for g in graphs.values():
        for r in g['rooms'].values():
            sw[r['type']].append(r['sweep_cost'])
    # 75th pct: the Point-4 audit showed realized/bound > 2x at the 95th (needlessly timid)
    allsw = sorted(c for v in sw.values() for c in v)
    cmax = {t: float(np.percentile(v, 75)) for t, v in sw.items()}
    cmax['unknown'] = float(np.percentile(allsw, 75))

    # start = a hallway hub if the scene has one, else the best-connected room
    def rkey(k):
        return int(k) if dataset == 'mp3d' else k
    start = pick_start(G, adj, rkey, start_idx, level)
    level = G['rooms'][start]['level']; y = G['rooms'][start]['floor_y']
    # targets: (room_id, x, y, z, horizontal half-extent) per instance
    if dataset == 'hm3d':
        targets = [(rid, o[1], o[2], o[3], o[4])
                   for rid, r in G['rooms'].items() for o in r.get('objects_full', []) if o[0] == cat]
        glb = os.path.join(HM3D, G['scene_rel'], stem + ".glb")
    else:
        regions, objects = parse_house(f"{MP3D}/{stem}/{stem}.house")
        targets = [(ridx, o[0], o[1], o[2], r) for ridx, mp, o, r in objects if mp == cat]
        glb = f"{MP3D}/{stem}/{stem}.glb"
    # [v2/E2] reachable = same level AND graph-connected to the start (BFS over edges);
    # rooms on navmesh islands no longer inflate the denominator. P6: start room excluded.
    comp, fringe = {start}, [start]
    while fringe:
        u = fringe.pop()
        for v, _ in adj[u]:
            if v not in comp and G['rooms'][v]['level'] == level:
                comp.add(v); fringe.append(v)
    n_reachable = sum(1 for t in targets
                      if str(t[0]) in comp and str(t[0]) != str(start))

    sim = make_sim(glb, gpu)
    pf = sim.pathfinder
    nav, x0, z0 = nav_grid(pf, y)
    if dataset == 'hm3d':
        gm = G['grids'][str(level)]
        assert abs(gm['x0'] - x0) < 0.5 and abs(gm['z0'] - z0) < 0.5, \
            f"grid origin drift: stored ({gm['x0']},{gm['z0']}) vs runtime ({x0:.3f},{z0:.3f})"
        lab = None
        def cells_of(room_id):
            return np.asarray(G['rooms'][room_id]['cells'], np.int32)
    else:
        lab = assign_cells(nav, x0, z0, y, regions, level)
        def cells_of(room_id):
            return np.argwhere(lab == int(room_id))
    st = sim.get_agent(0).get_state()
    st.position = np.asarray(pf.snap_point(G['rooms'][start]['centroid']), np.float32)
    sim.get_agent(0).set_state(st)

    rec = Recorder(sim, nav, x0, z0, y, cat, budget, f"B={budget}")
    follower = habitat_sim.GreedyGeodesicFollower(pf, sim.get_agent(0),
        forward_key="move_forward", left_key="turn_left", right_key="turn_right")
    B = budget
    counted = set()
    obs = {start}
    plan_log = []
    raw_frames = []          # RGB captured BEFORE every primitive (locked cadence)
    found_events = []        # primitive index of each new-instance collection
    WORD = {"move_forward": "forward", "turn_left": "left", "turn_right": "right"}

    def spend(action):
        nonlocal B
        raw_frames.append(np.asarray(sim.get_sensor_observations()["rgb"])[..., :3].copy())
        sim.step(action); B -= 1
        rec.traj.append(apos(sim)[[0, 2]].tolist()); rec.actions.append(WORD[action])
        # Point 7 crediting: within COLLECT_R of the object's obb (not its center), on the
        # SAME floor, and inside the forward camera cone -- what a detector could confirm.
        st_ = sim.get_agent(0).get_state(); p = np.array(st_.position, np.float32)
        th = 2 * math.atan2(st_.rotation.y, st_.rotation.w)
        fx, fz = -math.sin(th), -math.cos(th)
        for k, (ridx, ox, oy, oz, r) in enumerate(targets):
            if k in counted or abs(oy - p[1]) > 1.5:
                continue
            dx, dz = ox - p[0], oz - p[2]
            d = math.hypot(dx, dz)
            if d - r > COLLECT_R:
                continue
            ang = math.degrees(abs(math.atan2(fx * dz - fz * dx, fx * dx + fz * dz)))
            if ang <= 39.5 or d - r <= 0.3:      # in view cone, or practically on top of it
                counted.add(k); rec.found.append((ox, oz)); found_events.append(len(rec.actions))
        if len(rec.actions) % 2 == 0:
            rec.snap(B)

    def bearing_to(goal):
        """Signed angle (deg, +left) from the agent's heading to the first path waypoint."""
        try:
            path = habitat_sim.ShortestPath(); path.requested_start = apos(sim); path.requested_end = np.asarray(goal, np.float32)
            if not sim.pathfinder.find_path(path) or len(path.points) < 2:
                return None
            tgt = np.asarray(path.points[1], np.float32)
        except Exception:
            return None
        st_ = sim.get_agent(0).get_state(); p = np.array(st_.position, np.float32)
        th = 2 * math.atan2(st_.rotation.y, st_.rotation.w)
        fx, fz = -math.sin(th), -math.cos(th)
        dx, dz = tgt[0] - p[0], tgt[2] - p[2]
        if math.hypot(dx, dz) < 1e-3:
            return None
        cross = fx * dz - fz * dx; dot = fx * dx + fz * dz
        return math.degrees(math.atan2(cross, dot))

    def goto(goal, cap):
        try:
            follower.reset()
        except Exception:
            pass
        tie_dir = None     # [P15] the follower always breaks a ~180 deg tie the same way; we don't
        for _ in range(cap):
            if B <= 0:
                return False
            try:
                a = follower.next_action_along(np.asarray(goal, np.float32))
            except Exception:
                return False
            if a is None or a == "stop":
                return True
            if a in ("turn_left", "turn_right"):
                ang = bearing_to(goal)
                if ang is not None and abs(ang) > TIE_DEG:
                    if tie_dir is None:
                        tie_dir = brng.choice(("turn_left", "turn_right"))
                    a = tie_dir
                else:
                    tie_dir = None
            spend(a)
        return False

    # [P9 pilot, 2026-08-07] V2.1/V2.2 approach-visible mechanism (ladder row 4).
    # "Visible" = uncounted, same floor, within R_VIS of the pose, inside the P3 camera
    # cone, crediting pose snappable, and line-of-sight by the geodesic/euclid proxy
    # (walls force geodesic >> euclid; a near-straight walkable path ~ visible).
    # Sight points: after each scan turn at a sweep viewpoint, on viewpoint arrival, and
    # at the top of the planning loop (exploit what you can see before buying a room).
    # Not yet: sighting during travel legs, early-stop sweeps, cost-aware Q (later rows).
    unapproachable = set()

    def sighted():
        st_ = sim.get_agent(0).get_state(); p = np.array(st_.position, np.float32)
        th = 2 * math.atan2(st_.rotation.y, st_.rotation.w)
        fx, fz = -math.sin(th), -math.cos(th)
        hits = []
        for k, (ridx, ox, oy, oz, r) in enumerate(targets):
            if k in counted or k in unapproachable or abs(oy - p[1]) > 1.5:
                continue
            dx, dz = ox - p[0], oz - p[2]
            d = math.hypot(dx, dz)
            if d - r > R_VIS:
                continue
            ang = math.degrees(abs(math.atan2(fx * dz - fz * dx, fx * dx + fz * dz)))
            if ang > 39.5:
                continue
            g = pf.snap_point(np.array([float(ox), float(y), float(oz)], np.float32))
            if g is None or np.any(np.isnan(g)):
                continue
            gd = geo(pf, p, np.asarray(g, np.float32))
            if not math.isfinite(gd) or gd > LOS_RATIO * max(d, 0.5):
                continue
            hits.append((d, k, np.asarray(g, np.float32)))
        return sorted(hits, key=lambda h: h[0])

    def approach_one(d, k, g):
        # walk to the crediting pose, then turn toward the instance until credited
        ridx, ox, oy, oz, r = targets[k]
        goto(g, cap=int(max(24, 2 * d / FORWARD_M + 24)))
        face_dir = None      # [P15] random side for a target behind us, fixed for this approach
        for _ in range(TURNS_360):
            if k in counted or B <= 0:
                break
            st_ = sim.get_agent(0).get_state(); p = np.array(st_.position, np.float32)
            th = 2 * math.atan2(st_.rotation.y, st_.rotation.w)
            fx, fz = -math.sin(th), -math.cos(th)
            dx, dz = ox - p[0], oz - p[2]
            cross, dot = fx * dz - fz * dx, fx * dx + fz * dz
            if abs(math.degrees(math.atan2(cross, dot))) > TIE_DEG:     # [P15] target behind: random side
                if face_dir is None:
                    face_dir = brng.choice(("turn_left", "turn_right"))
                spend(face_dir)
            else:
                face_dir = None
                spend("turn_left" if cross > 0 else "turn_right")
        if k not in counted:
            unapproachable.add(k)      # e.g. creditable pose blocked; never chase it again
        return k in counted

    def exploit_sighted(where):
        # approach every currently visible target, nearest first, re-scanning after each
        while B > 0:
            hits = sighted()
            if not hits:
                return
            d, k, g = hits[0]
            plan_log.append(('APPROACH', where, f"{d:.1f}m", None, B, len(rec.actions)))
            rec.snap(B, f"-> approach sighted {cat.replace('_',' ')} ({d:.1f} m)")
            approach_one(d, k, g)

    def sweep(room_id, er=None, alt_density=0.0, full=False):
        # full=True (P14 re-sweep): complete every viewpoint scan regardless of what
        # is credited -- a second pass finds nothing for THIS expert, but the student's
        # first-pass detection is imperfect, so the demonstration to give is "leftover
        # budget buys systematic re-coverage", not the cleared-exit shortcut.
        cells = cells_of(room_id)
        # [POLICY P5, 2026-07-30] visit viewpoints in nearest-neighbor order from the
        # current pose. Set-cover discovery order (most-covering first) zigzags through
        # the room center repeatedly; NN order walks it like a tour and matches the C1
        # sweep-cost model used for d and the feasibility bound.
        p0 = apos(sim)
        cur = ((p0[0] - x0) / CELL, (p0[2] - z0) / CELL)
        rem = [(int(a), int(b)) for a, b in room_viewpoints(cells)]
        # [P15, 2026-09-12] a 360 at every coverage viewpoint over-represents turning
        # in the demonstrations (12 turns per viewpoint, several viewpoints per room).
        # Scan once, from the viewpoint the set-cover ranked first (it sees the most of
        # the room), then approach whatever that scan revealed and move on.
        if SWEEP_ONCE:
            rem = rem[:1]
        ordered = []
        while rem:
            nxt = min(rem, key=lambda v: (v[0] - cur[0]) ** 2 + (v[1] - cur[1]) ** 2)
            rem.remove(nxt); ordered.append(nxt); cur = nxt
        # [P10 pilot, 2026-08-07] V2.1 early-stop: before each further viewpoint, price
        # the room's remaining unswept value (retrieved whole-room yield minus credits
        # since entry, discounted by uncovered fraction) per remaining sweep step, and
        # leave when it drops below `earlystop` x the best alternative's value density.
        # Caveat (accepted for the pilot): alt_density's Qhat includes continuation
        # value while in-room density is immediate-only; margin m >= 1 leans against
        # over-staying rather than over-leaving.
        n0_found = len(counted)
        vp_total = max(1, len(ordered))
        # [P13] targets that live in this room and are still uncredited
        def room_left():
            return [k for k, t in enumerate(targets)
                    if str(t[0]) == str(room_id) and k not in counted]
        for vi, (ci, cj) in enumerate(ordered):
            if B <= 0:
                return
            if anytime and not full and vi > 0 and not room_left():
                plan_log.append(('CLEARED', room_id, f"vp{vi}/{vp_total}", None, B,
                                 len(rec.actions)))
                rec.snap(B, "-> room cleared, moving on")
                return
            if earlystop and er is not None and vi > 0:
                found_here = len(counted) - n0_found
                vp_left = vp_total - vi
                exp_rem = max(0.0, er - found_here) * (vp_left / vp_total)
                in_density = exp_rem / max(1, 20 * vp_left)   # ~8 walk + 12 turn prims per vp
                if in_density < earlystop * alt_density:
                    plan_log.append(('LEAVE', room_id, f"vp{vi}/{vp_total}", round(exp_rem, 2), B,
                                     len(rec.actions)))
                    rec.snap(B, f"-> leave early ({vp_left} viewpoints unswept)")
                    return
            wx, wz = x0 + (ci + .5) * CELL, z0 + (cj + .5) * CELL
            g = pf.snap_point([wx, y, wz])
            if g is None or np.any(np.isnan(g)):
                continue
            if not goto(g, cap=200):
                continue
            if approach:
                exploit_sighted('sweep-arrive')      # [P9] chairs seen on arrival
            # [P13] alternate scan direction by viewpoint parity (kills the 2:1
            # left bias the fixed CCW scan stamped on every demonstration)
            turn = "turn_left" if (vi + SWEEP_N[0]) % 2 == 0 else "turn_right"
            SWEEP_N[0] += 1
            for _ in range(TURNS_360):
                if B <= 0:
                    return
                spend(turn)
                if approach:
                    exploit_sighted('sweep-scan')    # [P9] sight interrupt mid-scan
                if anytime and not full and not room_left():
                    break                            # [P13] nothing left here to see

    # [POLICY P6, 2026-08-01] start-room instances are EXCLUDED from collection: the
    # base spec's objective sums rewards over SELECTED frontier rooms only -- the start
    # room is context (c0), never a chosen action, so its contents are not "collected".
    # (Replaces P1's free crediting, which pre-lit stars at t=0 in object-rich starts.)
    # They are pre-marked counted so they can never be credited, and n_reachable already
    # excludes them below.
    for k, (ridx, ox, oy, oz, r) in enumerate(targets):
        if str(ridx) == str(start):
            counted.add(k)
    n_start_excluded = len(counted)
    rec.snap(B, f"start: {G['rooms'][start]['type']} (its {n_start_excluded} instances excluded)")
    cur = start
    while B > 0:
        if approach:
            exploit_sighted('replan')                # [P9] exploit visibles before buying a room
            if B <= 0:
                break
        # frontier candidates from the OBSERVED graph, scored leak-free.
        # [v2/E4] raw frontier and affordable set tracked separately for honest stops.
        raw, cands = [], []
        p = apos(sim)
        for u in obs:
            for v, e in adj[u]:
                if v in obs or any(c['room'] == v for c in raw):
                    continue
                if G['rooms'][v]['level'] != level:
                    continue
                gdoor = geo(pf, p, np.asarray(e['door'], np.float32))
                if not math.isfinite(gdoor):
                    continue
                dn = max(1, int(round(gdoor / FORWARD_M)))
                vt = voted_type(v, G, prior, e['door'])
                cd = {'room': v, 'edge': e, 'voted_type': vt, 'dnav_prims': dn}
                raw.append(cd)
                if dn + cmax.get(vt, cmax['unknown']) <= B:    # conservative feasibility
                    cands.append(cd)
        if anytime:
            cands = raw                              # [P13] no affordability gate
        if not raw:
            if anytime and resweep and B > 0:
                break                                # [P14] leftover budget -> re-sweep phase below
            plan_log.append(('STOP', 'frontier exhausted', B)); break
        if not cands:
            plan_log.append(('STOP', 'nothing affordable', B)); break
        if policy == 'random':                    # classical baseline: uniform affordable frontier
            best = brng.choice(cands); q = 0.0
            plan_log.append((best['room'], G['rooms'][best['room']]['type'], best['voted_type'], None, B,
                             len(rec.actions)))
            rec.snap(B, f"-> random")
            goto(np.asarray(pf.snap_point(G['rooms'][best['room']]['centroid']), np.float32), cap=400)
            obs.add(best['room']); cur = best['room']
            sweep(cur)
            continue
        if policy == 'greedy':                    # classical baseline: nearest affordable frontier
            best = min(cands, key=lambda cd: cd['dnav_prims']); q = 0.0
            plan_log.append((best['room'], G['rooms'][best['room']]['type'], best['voted_type'], None, B,
                             len(rec.actions)))
            rec.snap(B, f"-> nearest")
            goto(np.asarray(pf.snap_point(G['rooms'][best['room']]['centroid']), np.float32), cap=400)
            obs.add(best['room']); cur = best['room']
            sweep(cur)
            continue
        c = context_of(cur, obs, G)
        qs = pl.rank(c, cands, B)
        scored = [(q, cd) for q, cd in zip(qs, cands) if q is not None]
        if not scored or max(q for q, _ in scored) <= 0:
            # [POLICY P7, 2026-08-02] zero branch is NOT terminal: with surplus budget,
            # leftover time is worthless under the collection objective (rewards are
            # non-negative), so continuing weakly dominates stopping while any room is
            # affordable. Curiosity fallback: cheapest unexplored room.
            best = min(cands, key=lambda cd: cd['dnav_prims'])
            q = 0.0
        else:
            q, best = max(scored, key=lambda t: t[0])
        plan_log.append((best['room'], G['rooms'][best['room']]['type'], best['voted_type'], round(q, 2), B,
                         len(rec.actions)))     # prim index of this decision
        rec.snap(B, f"-> {best['voted_type']}? (Q={q:.2f})")
        er = alt_density = None
        if earlystop:
            er = pl.yield_est(c, best['voted_type'], best['dnav_prims'])
            others = [(qq, cd) for qq, cd in scored if cd is not best]
            alt_density = max(0.0, max((qq / max(1, cd['dnav_prims'] + cmax.get(cd['voted_type'], cmax['unknown']))
                                        for qq, cd in others), default=0.0))
        goto(np.asarray(pf.snap_point(G['rooms'][best['room']]['centroid']), np.float32), cap=400)
        obs.add(best['room']); cur = best['room']
        sweep(cur, er=er, alt_density=alt_density or 0.0)

    # [P14, 2026-08-29] re-sweep: frontier exhausted with budget left. Cycle the visited
    # rooms that hold this category, richest first, with FULL viewpoint scans. Finds
    # nothing for this expert (everything reachable is already credited) but supervises
    # exactly the regime the dedicated student lacked: productive behavior deep into a
    # long budget. Stop reason stays the default 'budget exhausted'.
    resweep_passes = 0
    if anytime and resweep and B > 0 and not any(r[0] == 'STOP' for r in plan_log):
        rs = sorted((r for r in obs if any(str(t[0]) == str(r) for t in targets)),
                    key=lambda r: -sum(str(t[0]) == str(r) for t in targets))
        # after 2 passes of the category rooms, widen to the whole visited floor in
        # original tour order -- otherwise a single-small-room category degenerates
        # into pacing one room for the entire tail (observed: 181 passes of one
        # bedroom), which is the scan junk P13 removed sneaking back in
        seen = set()
        tour = [r for r in [start] + [p[0] for p in plan_log if p[0] in G['rooms']]
                if not (r in seen or seen.add(r))]
        # progress guard: a pass in which every room is skipped (centroid off-navmesh,
        # goto fails without spending) would otherwise spin forever at constant B
        while B > 0 and rs and resweep_passes < 300:
            B_pass0 = B
            resweep_passes += 1
            for r in (rs if resweep_passes <= 2 else tour):
                if B <= 0:
                    break
                plan_log.append(('RESWEEP', r, f"pass{resweep_passes}", None, B, len(rec.actions)))
                rec.snap(B, f"-> re-sweep {G['rooms'][r]['type']} (pass {resweep_passes})")
                g = pf.snap_point(G['rooms'][r]['centroid'])
                if g is None or np.any(np.isnan(g)):
                    continue
                if not goto(np.asarray(g, np.float32), cap=400):
                    continue
                sweep(r, full=True)
            if B == B_pass0:
                plan_log.append(('STOP', 'resweep made no progress', B))
                break

    rec.snap(B, "STOP")
    import imageio
    if out:
        for _ in range(20):
            rec.frames.append(rec.frames[-1])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        imageio.mimsave(out, rec.frames, fps=18, macro_block_size=2)
    if trace_dir:
        # final observation AFTER the last primitive (frame for the terminal record)
        raw_frames.append(np.asarray(sim.get_sensor_observations()["rgb"])[..., :3].copy())
        os.makedirs(trace_dir, exist_ok=True)
        ep = f"{stem}_{cat}_B{budget}" if req_level is None else f"{stem}_L{req_level}_{cat}_B{budget}"
        imageio.mimsave(os.path.join(trace_dir, ep + "_frames.mp4"), raw_frames, fps=1, macro_block_size=2)
        json.dump({
            'stem': stem, 'cat': cat, 'budget': budget, 'level': level, 'start_room': start,
            'actions': rec.actions, 'spent': budget - B,
            'collected': len(counted) - n_start_excluded,   # P6: exploration-only
            'n_start_excluded': n_start_excluded,
            'n_targets': len(targets), 'n_reachable': n_reachable,
            'decisions': [{'prim': r[5], 'room': r[0], 'true_type': r[1], 'voted_type': r[2], 'q': r[3], 'B': r[4]}
                          for r in plan_log if r[0] != 'STOP'],
            'stop': {'reason': next((r[1] for r in plan_log if r[0] == 'STOP'), 'budget exhausted'),
                     'prim': len(rec.actions)},
            'resweep_passes': resweep_passes,        # [P14] 0 = mode off or budget ran out first

            'found_events': found_events,
        }, open(os.path.join(trace_dir, ep + "_trace.json"), 'w'), indent=1)
        print(f"TRACE {ep}: {len(raw_frames)} frames, {len(rec.actions)} prims -> {trace_dir}")
    print(f"DONE B0={budget} spent={budget-B} collected={len(counted)-n_start_excluded}/{n_reachable} reachable "
          f"(total {len(targets)} incl. other floors) "
          f"rooms={[r[0] for r in plan_log if r[0] != 'STOP']} -> {out or trace_dir}")
    for row in plan_log:
        print("  plan:", row)
    sim.close()
    return len(counted)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', default='1pXnuDYAj8r'); ap.add_argument('--cat', default='chair')
    ap.add_argument('--budget', type=int, default=600); ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', default=''); ap.add_argument('--trace-dir', default='')
    ap.add_argument('--dataset', choices=['mp3d', 'hm3d'], default='mp3d')
    ap.add_argument('--start-idx', type=int, default=0)
    ap.add_argument('--level', type=int, default=None,
                    help='restrict the episode to one floor of a multi-storey house')
    ap.add_argument('--anytime', type=int, default=0,
                    help='P13: budget-blind anytime expert -- no affordability gate, '
                         'no early-stop, alternating scans, leave-when-cleared; '
                         'every budget is a truncation of the same trajectory')
    ap.add_argument('--resweep', type=int, default=0,
                    help='P14: after frontier exhaustion, spend leftover budget on full '
                         're-sweeps of the category-bearing rooms (richest first) instead '
                         'of stopping -- supervises deep-budget behavior; anytime only')
    ap.add_argument('--bank-file', default=None)
    ap.add_argument('--sweep-once', type=int, default=0,
                    help='P15: one 360 per room at its best-covering viewpoint instead of a '
                         'tour of every coverage viewpoint (less turning in the data)')
    ap.add_argument('--cam-h', type=float, default=0.88, help='camera height (m); RealSense on the Go2: measure it')
    ap.add_argument('--hfov', type=float, default=79.0, help='horizontal FOV (deg); RealSense D435i colour: 69')
    ap.add_argument('--res', default='640x480', help='WxH; RealSense recipe: 640x360')
    ap.add_argument('--policy', choices=['bellman', 'random', 'greedy'], default='bellman')
    ap.add_argument('--approach', type=int, default=0, help='P9 pilot: approach-visible-target mechanism')
    ap.add_argument('--earlystop', type=float, default=0.0,
                    help='P10 pilot: leave-room margin m (0 = off); bellman policy only')
    a = ap.parse_args()
    SWEEP_ONCE = bool(a.sweep_once)
    CAM_H, HFOV = a.cam_h, a.hfov
    W, H = map(int, a.res.lower().split('x'))
    PANEL_H = H
    run(a.stem, a.cat, a.budget, a.gpu, a.out, trace_dir=a.trace_dir, dataset=a.dataset, start_idx=a.start_idx, bank_file=a.bank_file, policy=a.policy,
        approach=bool(a.approach), earlystop=a.earlystop, level=a.level, anytime=bool(a.anytime),
        resweep=bool(a.resweep))
