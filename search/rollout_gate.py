"""Round-2 gate (ii): model-in-the-loop rollout diff, B=150 vs B=2000.

Rolls the fine-tuned model in habitat on frozen eval tuples under two budgets and
diffs the trajectories. Token probes ask "does one answer change?"; this asks the
question that matters: does BEHAVIOR change with the clock? In-episode the countdown
decrements every step, so even weak conditioning can compound into divergent routes.

Protocol per episode (faithful to training cadence):
  frame -> model sees history (uniform-subsampled to 448, keep last) -> predict 4
  actions -> execute them (crediting = P3/P6, same as the expert fleet) -> B -= steps
  -> requery with the DECREMENTED countdown. Episode ends on budget exhausted, 'stop'
  as the first predicted token, or follower failure.

  python rollout_gate.py --ckpt $RTC_DATA/sft_ckpts/uninavid-budget-full-v3 --gpu 1
"""
from paths import DATA_ROOT
import argparse, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{DATA_ROOT}/Uni-NaVid')

import numpy as np

REPEAT_CONTROL = ('5cdEh9F2hJL', 'chair')   # run B=150 twice: greedy determinism check
TUPLES = [  # (stem, cat) from the frozen eval scenes, start_idx 0
    ('5cdEh9F2hJL', 'chair'), ('VBzV5z6i1WS', 'chair'), ('mv2HUxq3B53', 'chair'),
    ('q5QZSEeHe5g', 'plant'), ('h1zeeAwLh9Z', 'chair'), ('GLAQ4DNUx5U', 'chair'),
]
BUDGETS = (150, 2000)
EXEC_N = int(os.environ.get('EXEC_N', '4'))   # primitives executed per observation (4 = original protocol, 2 = P15)
MAX_HIST = 448
VOCAB = ('forward', 'left', 'right', 'stop')
ACT = {'forward': 'move_forward', 'left': 'turn_left', 'right': 'turn_right'}


def rollout(model, stem, cat, budget, gpu, record=None, start_idx=0):
    import collections
    import instr
    from c4_collect_expert import (make_sim, pick_start, geo, nav_grid, HM3D, COLLECT_R)
    import c4_collect_expert as _c4
    # [P15] robot-camera benchmarks: CAM_H / HFOV / RES env override the simulator camera
    if os.environ.get('CAM_H'): _c4.CAM_H = float(os.environ['CAM_H'])
    if os.environ.get('HFOV'): _c4.HFOV = float(os.environ['HFOV'])
    if os.environ.get('RES'): _c4.W, _c4.H = map(int, os.environ['RES'].lower().split('x'))
    from build_transition_bank import edges_of
    DG = os.path.dirname(os.path.abspath(__file__))
    # GRAPHS_HM3D lets the 592-scene expansion graphs be used, as c4_collect_expert.py already does
    G = json.load(open(os.environ.get('GRAPHS_HM3D', f'{DG}/room_graphs_hm3d_v2.json')))[stem]
    adj = edges_of(G)
    # START_LEVEL env restricts the start to one floor (multi-floor houses: the dense floor)
    _lv = os.environ.get('START_LEVEL')
    start = pick_start(G, adj, str, start_idx, level=(int(_lv) if _lv not in (None, '') else None))
    level = G['rooms'][start]['level']; y = G['rooms'][start]['floor_y']
    targets = [(rid, o[1], o[2], o[3], o[4])
               for rid, r in G['rooms'].items() for o in r.get('objects_full', []) if o[0] == cat]
    sim = make_sim(os.path.join(HM3D, G['scene_rel'], stem + '.glb'), 0)
    pf = sim.pathfinder
    st = sim.get_agent(0).get_state()
    st.position = np.asarray(pf.snap_point(G['rooms'][start]['centroid']), np.float32)
    sim.get_agent(0).set_state(st)

    counted = {k for k, t in enumerate(targets) if str(t[0]) == str(start)}   # P6
    n_start_ex = len(counted)
    frames, actions, traj = [], [], []
    B = budget
    art = vid_frames = found_xy = None
    if record:
        from replay_render import MapArt, font
        from PIL import Image, ImageDraw
        nav, x0, z0 = nav_grid(pf, y)
        art = MapArt(nav, x0, z0)
        vid_frames, found_xy = [], []

    def snap_frame(rgb):
        st_ = sim.get_agent(0).get_state(); p = np.array(st_.position, np.float32)
        th = 2 * math.atan2(st_.rotation.y, st_.rotation.w)
        agent = (art.px(p[0], p[2]), (-math.sin(th), -math.cos(th)))
        m = art.frame([(t[0], t[1]) for t in traj] or [[p[0], p[2]]], found_xy, agent,
                      f"MODEL (budget-full-v3): every {cat.replace('_',' ')}  [B={budget}]",
                      f"budget left: {max(B,0)}/{budget}   collected: {len(counted)-n_start_ex}")
        from PIL import Image
        rgb_im = Image.fromarray(rgb)
        m_im = Image.fromarray(m)
        h = 400
        r2 = rgb_im.resize((int(rgb_im.width*h/rgb_im.height)//2*2, h))
        m2 = m_im.resize((int(m_im.width*h/m_im.height)//2*2, h))
        fr = Image.new('RGB', (r2.width + m2.width + 4, h), (10, 10, 12))
        fr.paste(r2, (0, 0)); fr.paste(m2, (r2.width + 4, 0))
        vid_frames.append(np.asarray(fr))

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
                counted.add(k)

    stop_reason = 'budget exhausted'
    while B > 0:
        frames.append(np.asarray(sim.get_sensor_observations()['rgb'])[..., :3].copy())
        hist = frames
        if len(hist) > MAX_HIST:            # training frame cap: uniform, keep last
            idx = np.unique(np.linspace(0, len(hist) - 1, MAX_HIST).astype(int))
            hist = [hist[i] for i in idx]
        model.reset()
        model.rgb_list = list(hist)
        txt = model.predict(instr.collect_task(cat, None if os.environ.get('BLIND') == '1' else B))
        toks = [w for w in txt.split() if w in VOCAB][:4]
        if not toks:
            stop_reason = f'unparseable: {txt[:40]}'
            break
        if toks[0] == 'stop':
            stop_reason = 'model stop'
            break
        for w in toks[:EXEC_N]:            # [P15] chunk-2 models: 2 per frame (EXEC_N=2)
            if w == 'stop' or B <= 0:
                break
            sim.step(ACT[w]); B -= 1
            actions.append(w)
            traj.append(np.asarray(sim.get_agent(0).get_state().position)[[0, 2]].round(2).tolist())
            n0 = len(counted)
            credit()
            if record:
                for k in counted:
                    pass
                if len(counted) > n0:
                    for k, (rid, ox, oy, oz, r) in enumerate(targets):
                        if k in counted and (ox, oz) not in found_xy and str(rid) != str(start):
                            found_xy.append((ox, oz))
                snap_frame(np.asarray(sim.get_sensor_observations()['rgb'])[..., :3])
    collected = len(counted) - n_start_ex
    if record and vid_frames:
        import imageio
        os.makedirs(record, exist_ok=True)
        outp = f'{record}/{stem}_{cat}_B{budget}_model.mp4'
        for _ in range(12):
            vid_frames.append(vid_frames[-1])
        imageio.mimsave(outp, vid_frames, fps=18, macro_block_size=2)
        print(f'RECORDED {outp} ({len(vid_frames)} frames)', flush=True)
    sim.close()
    return {'stem': stem, 'cat': cat, 'budget': budget, 'spent': budget - B,
            'collected': collected, 'stop': stop_reason, 'n_actions': len(actions),
            'actions': ''.join(w[0] for w in actions), 'traj_end': traj[-1] if traj else None,
            'traj': traj}   # full (x, z) after every primitive, for figure renders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--record', default=None, help='dir: save [rgb|map] mp4 per episode')
    ap.add_argument('--only', default=None, help='comma list stem:cat — restrict tuples')
    ap.add_argument('--budgets', default=None, help='comma list, overrides (150,2000)')
    a = ap.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(a.gpu))
    import objectnav_uninavid as ov
    ov.CKPT = a.ckpt
    model = ov.Model(0)

    global BUDGETS
    if a.budgets:
        BUDGETS = tuple(int(b) for b in a.budgets.split(','))
    tuples = TUPLES
    if a.only:
        want = {tuple(x.split(':')) for x in a.only.split(',')}
        tuples = [t for t in TUPLES if t in want]
    if not a.record:
        r1 = rollout(model, *REPEAT_CONTROL, 150, a.gpu)
        r2 = rollout(model, *REPEAT_CONTROL, 150, a.gpu)
        print(f"CONTROL same-budget repeat identical: {r1['actions'] == r2['actions']}", flush=True)
    results = []
    for stem, cat in tuples:
        eps = {}
        for B in BUDGETS:
            r = rollout(model, stem, cat, B, a.gpu, record=a.record)
            eps[B] = r
            print(f"ROLLOUT {stem}/{cat} B={B}: collected {r['collected']}, spent {r['spent']}, "
                  f"stop={r['stop']}", flush=True)
        lo, hi = eps[BUDGETS[0]], eps[BUDGETS[1]]
        pre = os.path.commonprefix([lo['actions'], hi['actions']])
        div = len(pre) if (lo['actions'] != hi['actions'][:len(lo['actions'])] or
                           lo['spent'] != hi['spent']) else None
        print(f"  PAIR: routes diverge at step {len(pre)} "
              f"({'IDENTICAL prefix — no behavioral divergence' if div is None else 'DIVERGED'})",
              flush=True)
        results.append({'lo': lo, 'hi': hi, 'diverge_step': len(pre)})

    name = os.path.basename(a.ckpt.rstrip('/'))
    out = f'{DATA_ROOT}/sft_ckpts/rollout_gate_{name}.json'
    json.dump(results, open(out, 'w'), indent=1)
    n_div = sum(1 for r in results if r['diverge_step'] < r['lo']['spent'])
    print(f"\n=== ROLLOUT GATE ({name}) ===")
    print(f"pairs with behavioral divergence before the low budget ran out: {n_div}/{len(results)}")
    print(f"detail -> {out}")


if __name__ == '__main__':
    main()
