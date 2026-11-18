"""Only the countdown changed: B=150 vs B=2000 side by side, same scene, same start,
same model, deterministic decoding. Replays the two recorded action strings from the
rollout gate, verifies each against its stored end position and count, and renders:
slow start -> freeze at the fork -> fast-forward -> end state -> six-pair grid.

    countdown_clip.py --pair GLAQ4DNUx5U:chair --gpu 0 --out clip.mp4 [--keyframes dir]
"""
from paths import DATA_ROOT
import argparse, json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c4_collect_expert import make_sim, pick_start, HM3D, COLLECT_R, apos   # noqa: E402
from build_room_graphs import nav_grid                                        # noqa: E402
from build_transition_bank import edges_of                                    # noqa: E402
from replay_render import MapArt, font                                        # noqa: E402
ACT = {'f': 'move_forward', 'l': 'turn_left', 'r': 'turn_right'}
BG = (10, 10, 12); GOLD = (255, 208, 0); INK = (235, 235, 235); SOFT = (185, 190, 200); RED = (255, 55, 55)
C150, C2000 = (90, 175, 255), (255, 120, 30)


def prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n


def replay(sim, G, start, cat, actions, keep_frames=None):
    pf = sim.pathfinder
    st = sim.get_agent(0).get_state(); st.position = np.asarray(pf.snap_point(G['rooms'][start]['centroid']), np.float32)
    st.rotation = np.array([0, 0, 0, 1], np.float32); sim.get_agent(0).set_state(st)
    targets = [(rid, o[1], o[2], o[3], o[4]) for rid, r in G['rooms'].items() for o in r.get('objects_full', []) if o[0] == cat]
    counted = {k for k, t in enumerate(targets) if str(t[0]) == str(start)}; n_ex = len(counted)
    def pose():
        s_ = sim.get_agent(0).get_state(); p = np.array(s_.position, np.float32); th = 2 * math.atan2(s_.rotation.y, s_.rotation.w)
        return p, (-math.sin(th), -math.cos(th))
    def credit():
        p, (fx, fz) = pose()
        for k, (rid, ox, oy, oz, r) in enumerate(targets):
            if k in counted or abs(oy - p[1]) > 1.5: continue
            dx, dz = ox - p[0], oz - p[2]; d = math.hypot(dx, dz)
            if d - r > COLLECT_R: continue
            ang = math.degrees(abs(math.atan2(fx * dz - fz * dx, fx * dx + fz * dz)))
            if ang <= 39.5 or d - r <= 0.3: counted.add(k); finds.append((len(P) - 1, k))
    P, H, finds, frames = [], [], [], {}
    p, h = pose(); P.append(p[[0, 2]].tolist()); H.append(h)
    if keep_frames is not None and 0 in keep_frames: frames[0] = np.asarray(sim.get_sensor_observations()['rgb'])[..., :3].copy()
    for i, ch in enumerate(actions, 1):
        sim.step(ACT[ch]); p, h = pose(); P.append(p[[0, 2]].tolist()); H.append(h); credit()
        if keep_frames is not None and i in keep_frames: frames[i] = np.asarray(sim.get_sensor_observations()['rgb'])[..., :3].copy()
    return np.array(P), H, finds, targets, len(counted) - n_ex, frames


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pair', default='GLAQ4DNUx5U:chair'); ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--gate', required=True, help='the json written by rollout_gate.py for the checkpoint to show')
    ap.add_argument('--out', required=True); ap.add_argument('--fps', type=int, default=24); ap.add_argument('--keyframes', default='')
    a = ap.parse_args()
    gate = json.load(open(a.gate)); Gall = json.load(open(f'{HERE}/room_graphs_hm3d_v2.json'))
    stem, cat = a.pair.split(':')
    # replay every pair once (verification + grid), keep camera frames only for the featured pair
    runs = {}
    for pr in gate:
        s_, c_ = pr['lo']['stem'], pr['lo']['cat']; G = Gall[s_]; start = pick_start(G, edges_of(G), str, 0)
        y = G['rooms'][start]['floor_y']
        sim = make_sim(os.path.join(HM3D, G['scene_rel'], s_ + '.glb'), a.gpu); nav, x0, z0 = nav_grid(sim.pathfinder, y)
        feat = (s_ == stem and c_ == cat)
        res = {}
        for side in ('lo', 'hi'):
            rec = pr[side]; keep = set(range(0, len(rec['actions']) + 1)) if feat else None
            P, H, finds, targets, got, frames = replay(sim, G, start, c_, rec['actions'], keep)
            end_ok = float(np.abs(P[-1] - np.asarray(rec['traj_end'])).max()) < 0.03
            print(f"{s_} {c_} B{rec['budget']}: replayed {got} (stored {rec['collected']}) end {'OK' if end_ok else 'MISMATCH'}", flush=True)
            res[side] = dict(P=P, H=H, finds=finds, targets=targets, got=got, frames=frames, rec=rec)
        sim.close()
        runs[(s_, c_)] = dict(res=res, art=MapArt(nav, x0, z0), fork=prefix(pr['lo']['actions'], pr['hi']['actions']), start=start)
    R = runs[(stem, cat)]; art = R['art']; fork = R['fork']; lo, hi = R['res']['lo'], R['res']['hi']
    W, H_ = 1920, 1080; PW = 952

    def map_img(r, t, col, art_=None, size=None):
        art_ = art_ or art
        im = Image.fromarray(art_.base.copy()); d = ImageDraw.Draw(im); P = r['P']; tt = min(t, len(P) - 1)
        for s, k in r['finds']:
            if s <= tt: rid, ox, oy, oz, _ = r['targets'][k]; x, yy = art_.px(ox, oz); art_.star(d, x, yy, 11, GOLD)
        if tt >= 1: d.line([art_.px(*q) for q in P[:tt + 1]], fill=col, width=5)
        ax, ay = art_.px(*P[tt]); fx, fz = r['H'][tt]
        d.line([(ax, ay), (ax + int(fx * 18), ay - int(fz * 18))], fill=RED, width=5); d.ellipse([ax - 6, ay - 6, ax + 6, ay + 6], fill=RED)
        if size:
            s = min(size[0] / im.width, size[1] / im.height); im = im.resize((max(2, int(im.width * s)), max(2, int(im.height * s))), Image.NEAREST if s >= 1 else Image.BILINEAR)
        return im

    def pane(r, t, col, budget, x_off):
        p = Image.new('RGB', (PW, H_), BG); d = ImageDraw.Draw(p)
        tt = min(t, len(r['P']) - 1)
        d.text((16, 14), f'"Find as many {cat}s as possible.', font=font(26), fill=SOFT)
        d.text((16, 48), f'You have {budget} steps remaining."', font=font(30), fill=col)
        fr = r['frames'].get(tt)
        if fr is not None:
            im = Image.fromarray(fr); s = PW / im.width; im = im.resize((PW, int(im.height * s) // 2 * 2), Image.BILINEAR); p.paste(im, (0, 100))
        m = map_img(r, tt, col, size=(400, 300)); p.paste(m, (PW - m.width - 16, 100 + 714 - m.height - 16))
        n = sum(1 for s, _ in r['finds'] if s <= tt)
        stopped = tt >= len(r['P']) - 1
        d.text((16, H_ - 80), f"step {tt}    budget left {max(budget - tt, 0)}/{budget}    collected {n}", font=font(24), fill=INK)
        if stopped: d.text((16, H_ - 44), f"ended: {r['rec']['stop']}", font=font(22), fill=SOFT)
        return p

    def frame(t_lo, t_hi, banner=None, sub=None, grid=False):
        fr = Image.new('RGB', (W, H_), BG)
        fr.paste(pane(lo, t_lo, C150, 150, 0), (0, 0)); fr.paste(pane(hi, t_hi, C2000, 2000, PW + 16), (PW + 16, 0))
        ImageDraw.Draw(fr).line([(PW + 8, 0), (PW + 8, H_)], fill=(60, 60, 66), width=2)
        if banner:
            d = ImageDraw.Draw(fr); f = font(50); tw = d.textlength(banner, font=f)
            d.rectangle([(W - tw) / 2 - 30, 826, (W + tw) / 2 + 30, 916], fill=(0, 0, 0), outline=GOLD, width=3)
            d.text(((W - tw) / 2, 838), banner, font=f, fill=GOLD)
            if sub:
                f2 = font(24); tw2 = d.textlength(sub, font=f2); d.text(((W - tw2) / 2, 930), sub, font=f2, fill=INK)
        return fr

    def grid_card():
        fr = Image.new('RGB', (W, H_), BG); d = ImageDraw.Draw(fr)
        d.text((60, 30), 'All six held-out scene pairs diverge', font=font(40), fill=GOLD)
        d.text((60, 86), 'blue: told 150 steps    orange: told 2000 steps    gold star: chair credited', font=font(22), fill=SOFT)
        cw, ch = 600, 430
        for i, ((s_, c_), r_) in enumerate(runs.items()):
            gx, gy = 40 + (i % 3) * 620, 140 + (i // 3) * 470
            base = Image.fromarray(r_['art'].base.copy()); d2 = ImageDraw.Draw(base)
            for side, col in (('hi', C2000), ('lo', C150)):
                rr = r_['res'][side]; d2.line([r_['art'].px(*q) for q in rr['P']], fill=col, width=4)
                for s, k in rr['finds']:
                    rid, ox, oy, oz, _ = rr['targets'][k]; x, yy = r_['art'].px(ox, oz); r_['art'].star(d2, x, yy, 9, GOLD)
            sx0, sy0 = r_['art'].px(*r_['res']['lo']['P'][0]); d2.ellipse([sx0 - 8, sy0 - 8, sx0 + 8, sy0 + 8], fill=INK)
            s = min(cw / base.width, (ch - 70) / base.height); base = base.resize((int(base.width * s), int(base.height * s)), Image.BILINEAR)
            fr.paste(base, (gx + (cw - base.width) // 2, gy + 56))
            d.text((gx + 10, gy), f"{s_} \u00b7 {c_}", font=font(19), fill=SOFT)
            d.text((gx + 10, gy + 24), f"fork after {r_['fork']} actions     {r_['res']['lo']['got']} \u2192 {r_['res']['hi']['got']} found", font=font(19), fill=INK)
        return fr

    import imageio
    wr = imageio.get_writer(a.out, fps=a.fps, codec='libx264', quality=8, macro_block_size=2); n = 0
    def emit(fr, k):
        nonlocal n
        for _ in range(k): wr.append_data(np.asarray(fr)); n += 1
    keys = {}
    emit(frame(0, 0), a.fps)                                                       # identical start, 1 s
    for t in range(1, fork + 1): emit(frame(t, t), a.fps // 3)                     # slow: 3 steps per second up to the fork
    keys['fork'] = frame(fork, fork, 'Only the countdown changed', f'same scene, same start, same weights, deterministic decoding: trajectories diverge after {fork} actions')
    emit(keys['fork'], 2 * a.fps)
    for t in range(fork + 1, fork + 6): emit(frame(t, t), a.fps // 4)             # the first divergent steps, slow
    keys['diverged'] = frame(fork + 5, fork + 5)
    n_lo, n_hi = len(lo['P']) - 1, len(hi['P']) - 1
    fast = list(range(fork + 6, 151, 1))                                          # fast-forward to the B=150 horizon
    for t in fast: emit(frame(min(t, n_lo), min(t, n_hi)), 1 if t % 2 else 0)
    keys['b150_end'] = frame(n_lo, 150, 'B=150 run ends here', f"told 150: {lo['got']} found      told 2000: keeps searching")
    emit(keys['b150_end'], int(1.5 * a.fps))
    for t in range(151, n_hi + 1, 14): emit(frame(n_lo, t), 1)                     # the long run, 4 steps per frame
    keys['end'] = frame(n_lo, n_hi, f"{lo['got']} → {hi['got']} chairs", 'the same policy, told a different number')
    emit(keys['end'], 2 * a.fps)
    keys['grid'] = grid_card(); emit(keys['grid'], 3 * a.fps)
    wr.close(); print(json.dumps({'out': a.out, 'frames': n, 'seconds': round(n / a.fps, 1), 'fork': fork, 'lo': lo['got'], 'hi': hi['got']}))
    if a.keyframes:
        os.makedirs(a.keyframes, exist_ok=True)
        for k, im in keys.items(): im.save(f'{a.keyframes}/{k}.png')


if __name__ == '__main__':
    main()
