"""Replay a Figure-1 student trace ( *_student.json from fig1_student_traces.py ) and
render the classic [RGB | top-down map] episode video, 16:9.

The student json stores the exact action string and the (x,z) after every primitive,
so the episode is re-executed in habitat with no model in the loop. The start pose is
rebuilt the way rollout_gate.rollout builds it (pick_start on the same room graph,
snapped room centroid, default heading); the replayed trajectory is checked against
the stored one step by step, and crediting (P3/P6, same as the executor) is re-run
and checked against the stored collected count. A mismatch aborts rather than
producing a video that does not match the numbers in the paper.

    replay_student.py --trace fig1_fork/s1/q5QZSEeHe5g_chair_B400_student.json \
        --start-idx 1 --graphs room_graphs_hm3d_v2.json --gpu 0 --out /path/ep.mp4

Layout (default 1920x1080): left pane, camera scaled to 1120 px wide with the
"find every ... | step | budget left" caption; right pane, the MapArt top-down map
with trajectory, gold stars on finds and the budget banner.
"""
import argparse, json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from c4_collect_expert import make_sim, pick_start, HM3D, COLLECT_R, apos   # noqa: E402
from build_room_graphs import nav_grid                                        # noqa: E402
from build_transition_bank import edges_of                                    # noqa: E402
from replay_render import MapArt, font                                        # noqa: E402

ACT = {'f': 'move_forward', 'l': 'turn_left', 'r': 'turn_right'}
DARK = (10, 10, 12)
GOLD = (255, 208, 0)


def map_frame(art, traj, found, agent, banner, sub, pane_w, pane_h):
    """MapArt.frame, but upscaled with NEAREST so the occupancy blocks stay crisp."""
    im = Image.fromarray(art.base.copy()); d = ImageDraw.Draw(im)
    if len(traj) >= 2:
        d.line([art.px(x, z) for x, z in traj], fill=(90, 175, 255), width=4)
    for (ox, oz) in found:
        x, y = art.px(ox, oz); art.star(d, x, y, 11, GOLD)
    if agent is not None:
        (ax, ay), (fx, fz) = agent
        d.line([(ax, ay), (ax + int(fx * 18), ay - int(fz * 18))], fill=(255, 55, 55), width=5)
        d.ellipse([ax - 5, ay - 5, ax + 5, ay + 5], fill=(255, 55, 55))
    # fit the map under a banner strip inside the pane
    ban_h = 64
    avail_h = pane_h - ban_h
    s = min(pane_w / im.width, avail_h / im.height)
    mw, mh = max(2, int(im.width * s) // 2 * 2), max(2, int(im.height * s) // 2 * 2)
    im = im.resize((mw, mh), Image.NEAREST if s >= 1 else Image.BILINEAR)
    out = Image.new('RGB', (pane_w, pane_h), (0, 0, 0))
    out.paste(im, ((pane_w - mw) // 2, ban_h + (avail_h - mh) // 2))
    D = ImageDraw.Draw(out)
    D.text((10, 4), banner, font=font(26), fill=GOLD)
    D.text((10, 36), sub, font=font(19), fill=(200, 205, 215))
    return out


def cam_pane(rgb, caption, pane_w, pane_h):
    im = Image.fromarray(rgb)
    s = pane_w / im.width
    cw, ch = pane_w, int(im.height * s) // 2 * 2
    im = im.resize((cw, ch), Image.BILINEAR)
    out = Image.new('RGB', (pane_w, pane_h), DARK)
    y0 = (pane_h - ch) // 2
    out.paste(im, (0, y0))
    D = ImageDraw.Draw(out)
    D.rectangle([0, y0, pane_w, y0 + 40], fill=(0, 0, 0))
    D.text((10, y0 + 6), caption, font=font(24), fill=GOLD)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', required=True)
    ap.add_argument('--start-idx', type=int, default=0)
    ap.add_argument('--level', default='', help='START_LEVEL used when the trace was made ("" = none)')
    ap.add_argument('--graphs', default=f'{HERE}/room_graphs_hm3d_v2.json')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=18)
    ap.add_argument('--snap', type=int, default=2, help='render a frame every N primitives')
    ap.add_argument('--size', default='1920x1080')
    ap.add_argument('--cam-w', type=int, default=1120)
    ap.add_argument('--hold', type=float, default=1.5, help='seconds to hold the last frame')
    ap.add_argument('--tol', type=float, default=0.03, help='max |replayed - stored| per step (m)')
    a = ap.parse_args()

    tr = json.load(open(a.trace))
    stem, cat, budget = tr['stem'], tr['cat'], tr['budget']
    W, H = map(int, a.size.lower().split('x'))
    cam_w = a.cam_w; map_w = W - cam_w - 4

    G = json.load(open(a.graphs))[stem]
    adj = edges_of(G)
    level = int(a.level) if a.level not in ('', None) else None
    start = pick_start(G, adj, str, a.start_idx, level=level)
    y = G['rooms'][start]['floor_y']
    targets = [(rid, o[1], o[2], o[3], o[4])
               for rid, r in G['rooms'].items() for o in r.get('objects_full', []) if o[0] == cat]

    sim = make_sim(os.path.join(HM3D, G['scene_rel'], stem + '.glb'), a.gpu)
    pf = sim.pathfinder
    st = sim.get_agent(0).get_state()
    st.position = np.asarray(pf.snap_point(G['rooms'][start]['centroid']), np.float32)
    sim.get_agent(0).set_state(st)
    nav, x0, z0 = nav_grid(pf, y)
    art = MapArt(nav, x0, z0)

    counted = {k for k, t in enumerate(targets) if str(t[0]) == str(start)}     # P6
    n_start_ex = len(counted)
    found_xy = []
    traj = [apos(sim)[[0, 2]].tolist()]
    stored = tr['traj']
    acts = tr['actions']
    B = budget
    worst = 0.0

    import imageio
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    wr = imageio.get_writer(a.out, fps=a.fps, codec='libx264', quality=8, macro_block_size=2)
    n_frames = 0

    def agent_marker():
        s_ = sim.get_agent(0).get_state(); p = np.array(s_.position, np.float32)
        th = 2 * math.atan2(s_.rotation.y, s_.rotation.w)
        return (art.px(p[0], p[2]), (-math.sin(th), -math.cos(th)))

    def snap(step):
        nonlocal n_frames
        rgb = np.asarray(sim.get_sensor_observations()['rgb'])[..., :3].copy()
        left = cam_pane(rgb, f"find every {cat.replace('_', ' ')} | step {step} | budget left {max(B, 0)}",
                        cam_w, H)
        right = map_frame(art, traj, found_xy, agent_marker(),
                          f"collect every {cat.replace('_', ' ')}  [B={budget}]",
                          f"budget left: {max(B, 0)}/{budget}   collected: {len(counted) - n_start_ex}",
                          map_w, H)
        fr = Image.new('RGB', (W, H), DARK)
        fr.paste(left, (0, 0)); fr.paste(right, (cam_w + 4, 0))
        wr.append_data(np.asarray(fr)); n_frames += 1

    def credit():
        s_ = sim.get_agent(0).get_state(); p = np.array(s_.position, np.float32)
        th = 2 * math.atan2(s_.rotation.y, s_.rotation.w)
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
                if str(rid) != str(start):
                    found_xy.append((ox, oz))

    snap(0)
    for i, ch in enumerate(acts, 1):
        sim.step(ACT[ch]); B -= 1
        p = apos(sim)[[0, 2]]
        traj.append(p.tolist())
        if i - 1 < len(stored):
            dev = float(np.max(np.abs(p - np.asarray(stored[i - 1]))))
            worst = max(worst, dev)
            if dev > a.tol:
                wr.close(); sim.close()
                raise SystemExit(f'TRAJECTORY MISMATCH at step {i}: replayed {p.round(2).tolist()} vs '
                                 f'stored {stored[i - 1]} (dev {dev:.3f} m). Start room {start}, level '
                                 f'{level}, graphs {os.path.basename(a.graphs)} are probably not what made this trace.')
        credit()
        if i % a.snap == 0 or i == len(acts):
            snap(i)
    for _ in range(int(a.hold * a.fps)):
        snap(len(acts))
    wr.close(); sim.close()

    got = len(counted) - n_start_ex
    ok = got == tr['collected']
    print(json.dumps({'trace': os.path.basename(a.trace), 'out': a.out, 'frames': n_frames,
                      'start_room': start, 'level': G['rooms'][start]['level'],
                      'collected_replayed': got, 'collected_stored': tr['collected'],
                      'max_traj_dev_m': round(worst, 3), 'verified': ok}))
    if not ok:
        raise SystemExit('COLLECTED MISMATCH: replayed crediting disagrees with the stored trace')


if __name__ == '__main__':
    main()
