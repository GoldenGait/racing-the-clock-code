"""Search -> discover -> collect -> replan: the room-level expert's closed loop, on the
Fig. 2(c) episode (U3oQjwTuMX8 / chair / B=400). Replays the stored actions in habitat,
projects the chairs into the camera so TARGET VISIBLE can point at a real chair, and
derives every phase label from the trace's decisions and found_events.

  --storyboard  writes a 2x3 still sheet (overview map + key moments) for review
  --video       writes the 16:9 clip with slow-motion and freezes (default off)
"""
from paths import DATA_ROOT
import argparse, json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c4_collect_expert import make_sim, HM3D, COLLECT_R, apos, CAM_H, HFOV, W as CW, H as CH   # noqa: E402
from build_room_graphs import nav_grid, CELL                                                     # noqa: E402
from replay_render import MapArt, font                                                           # noqa: E402
ACT = {'forward': 'move_forward', 'left': 'turn_left', 'right': 'turn_right'}
BG = (10, 10, 12); GOLD = (255, 208, 0); BLUE = (90, 175, 255); ORANGE = (255, 120, 30); LIT = (66, 72, 86); DIM = (34, 37, 44)
INK = (235, 235, 235); SOFT = (185, 190, 200); GREEN = (80, 220, 120); RED = (255, 55, 55)


def phases(tr, room):
    """(start, end, label) segments for the chosen room, from the trace."""
    dec = tr['decisions']; fe = sorted(tr['found_events'])
    i0 = next(i for i, d in enumerate(dec) if d.get('room') == room)
    t_enter = dec[i0]['prim']
    t_leave = next(d['prim'] for d in dec[i0 + 1:] if d.get('room') in ('CLEARED', 'LEAVE'))
    nxt = next((d for d in dec[i0 + 1:] if d['prim'] >= t_leave and d.get('room') not in ('CLEARED', 'LEAVE', 'APPROACH')), None)
    appr = [d['prim'] for d in dec[i0 + 1:] if d.get('room') == 'APPROACH' and d['prim'] < t_leave]
    segs, t = [], t_enter
    for ta in appr:
        segs.append((t, ta, 'SEARCHING'))
        tc = next((s for s in fe if s >= ta), t_leave)      # first credit after the interruption
        segs.append((ta, tc, 'APPROACHING')); t = tc
    segs.append((t, t_leave, 'SEARCHING'))
    return t_enter, t_leave, appr, segs, nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', default=os.path.join(HERE, 'examples', 'U3oQjwTuMX8_L0_chair_B400_trace.json'),
                    help='a room-level expert trace; the default is the episode shown in the paper')
    ap.add_argument('--room', default='0_7'); ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--storyboard', default=''); ap.add_argument('--video', default='')
    ap.add_argument('--fps', type=int, default=18)
    ap.add_argument('--full', default='', help='write the whole-episode clip to this path')
    ap.add_argument('--per-step', type=int, default=2, help='frames per primitive in the full clip')
    a = ap.parse_args()
    tr = json.load(open(a.trace)); stem, cat, budget = tr['stem'], tr['cat'], tr['budget']
    G = json.load(open(f'{HERE}/room_graphs_hm3d_v2.json'))[stem]
    start = str(tr['start_room']); level = G['rooms'][start]['level']; y = G['rooms'][start]['floor_y']
    rooms = {rid: r for rid, r in G['rooms'].items() if r['level'] == level}
    grid = G['grids'][str(level)]; gx0, gz0 = grid['x0'], grid['z0']
    t_enter, t_leave, appr, segs, nxt = phases(tr, a.room)
    targets = [(rid, o[1], o[2], o[3], o[4]) for rid, r in G['rooms'].items() for o in r.get('objects_full', []) if o[0] == cat]

    sim = make_sim(os.path.join(HM3D, G['scene_rel'], stem + '.glb'), a.gpu); pf = sim.pathfinder
    nav, x0, z0 = nav_grid(pf, y); art = MapArt(nav, x0, z0)
    st = sim.get_agent(0).get_state(); st.position = np.asarray(pf.snap_point(G['rooms'][start]['centroid']), np.float32)
    st.rotation = np.array([0, 0, 0, 1], np.float32); sim.get_agent(0).set_state(st)
    counted = {k for k, t in enumerate(targets) if str(t[0]) == start}; n_ex = len(counted)
    fx = (CW / 2) / math.tan(math.radians(HFOV) / 2)

    def pose():
        s_ = sim.get_agent(0).get_state(); p = np.array(s_.position, np.float32); th = 2 * math.atan2(s_.rotation.y, s_.rotation.w)
        return p, th

    def project(p, th, ox, oy, oz):
        cam = p + np.array([0, CAM_H, 0], np.float32); d = np.array([ox, oy, oz], np.float32) - cam
        f = np.array([-math.sin(th), 0, -math.cos(th)]); r = np.array([math.cos(th), 0, -math.sin(th)]); u = np.array([0, 1, 0])
        cz = float(d @ f)
        if cz < 0.2:
            return None
        return (CW / 2 + fx * float(d @ r) / cz, CH / 2 - fx * float(d @ u) / cz, cz)

    def credit(step, finds):
        p, th = pose(); fwd = (-math.sin(th), -math.cos(th))
        for k, (rid, ox, oy, oz, r) in enumerate(targets):
            if k in counted or abs(oy - p[1]) > 1.5: continue
            dx, dz = ox - p[0], oz - p[2]; dd = math.hypot(dx, dz)
            if dd - r > COLLECT_R: continue
            ang = math.degrees(abs(math.atan2(fwd[0] * dz - fwd[1] * dx, fwd[0] * dx + fwd[1] * dz)))
            if ang <= 39.5 or dd - r <= 0.3:
                counted.add(k); finds.append((step, k))

    P, TH, finds, frames = [], [], [], {}
    p, th = pose(); P.append(p[[0, 2]].tolist()); TH.append(th)
    want = set(range(t_enter, t_leave + 2)) if not a.full else set(range(0, len(tr['actions']) + 1))
    frames[0] = np.asarray(sim.get_sensor_observations()['rgb'])[..., :3].copy()
    for i, act in enumerate(tr['actions'], 1):
        sim.step(ACT[act]); p, th = pose(); P.append(p[[0, 2]].tolist()); TH.append(th); credit(i, finds)
        if i in want or i == 0:
            frames[i] = np.asarray(sim.get_sensor_observations()['rgb'])[..., :3].copy()
    sim.close()
    got = len(counted) - n_ex
    print(f'replayed: {got} credited (stored {tr["collected"]}) {"OK" if got == tr["collected"] else "MISMATCH"}; room {a.room} window {t_enter}-{t_leave}, approaches {appr}, next {nxt and nxt.get("room")}')
    room_finds = [(s, k) for s, k in finds if t_enter <= s <= t_leave]
    print('credits in room:', [s for s, _ in room_finds])

    def label_at(t):
        for s0, s1, lab in segs:
            if s0 <= t < s1: return lab
        return 'ROOM CLEARED' if t >= t_leave else 'SEARCHING'

    def cam_pane(t, pw, ph, big=None):
        im = Image.fromarray(frames[t]); d = ImageDraw.Draw(im)
        # rings on chairs in view, gold once credited
        p3 = np.array([P[t][0], y, P[t][1]]); th = TH[t]
        # ring rule = the executor's trigger: an uncredited chair in view within 5 m (orange);
        # credited chairs stay ringed in gold
        for k, (rid, ox, oy, oz, r) in enumerate(targets):
            if not a.full and str(rid) != a.room: continue
            done = any(kk == k and s <= t for s, kk in finds)
            pr = project(np.array([P[t][0], 0.0, P[t][1]], np.float32) + np.array([0, y, 0], np.float32), th, ox, oy, oz)
            if pr is None: continue
            u, v, cz = pr
            if not (0 <= u < CW and 0 <= v < CH): continue
            if not done and cz > 5.0: continue
            rad = min(90, max(12, int(80 / cz))); col = GOLD if done else ORANGE
            d.ellipse([u - rad, v - rad, u + rad, v + rad], outline=col, width=5 if not done else 3)
        s = pw / im.width; im = im.resize((pw, int(im.height * s) // 2 * 2), Image.BILINEAR)
        out = Image.new('RGB', (pw, ph), BG); yo = (ph - im.height) // 2; out.paste(im, (0, yo)); D = ImageDraw.Draw(out)
        D.rectangle([0, yo, pw, yo + 44], fill=(0, 0, 0))
        D.text((12, yo + 8), f"find every {cat} | step {t} | budget left {budget - t}", font=font(24), fill=GOLD)
        if big:
            f = font(64); tw = D.textlength(big[0], font=f)
            D.rectangle([(pw - tw) / 2 - 24, yo + 70, (pw + tw) / 2 + 24, yo + 160], fill=(0, 0, 0))
            D.text(((pw - tw) / 2, yo + 78), big[0], font=f, fill=big[1])
        return out

    def map_pane(t, pw, ph, note=None):
        base = art.base.copy(); base[(base == LIT).all(-1)] = DIM
        im = Image.fromarray(base); d = ImageDraw.Draw(im); ms = art.MS
        entered = []
        cell_room = {(i, j): rid for rid, r in rooms.items() for i, j in r['cells']}
        for k in range(t + 1):
            rid = cell_room.get((int((P[k][0] - gx0) // CELL), int((P[k][1] - gz0) // CELL)))
            if rid and rid not in entered: entered.append(rid)
        for rid in entered:
            for i, j in rooms[rid]['cells']:
                px, py = art.px(gx0 + (i + .5) * CELL, gz0 + (j + .5) * CELL)
                d.rectangle([px - ms // 2, py - ms // 2, px + ms // 2, py + ms // 2], fill=LIT)
        pts = [art.px(*q) for q in P[:t + 1]]
        if len(pts) > 1: d.line(pts, fill=BLUE, width=5)
        for s0, s1, lab in segs:                              # detours in orange, like Fig. 2(c)
            if lab == 'APPROACHING' and t > s0:
                d.line([art.px(*q) for q in P[s0:min(t, s1) + 1]], fill=ORANGE, width=6)
        for s, k in finds:
            if s <= t:
                rid, ox, oy, oz, r = targets[k]; x, yy = art.px(ox, oz); art.star(d, x, yy, 12, GOLD)
        ax, ay = art.px(*P[t]); fxh, fzh = -math.sin(TH[t]), -math.cos(TH[t])
        d.line([(ax, ay), (ax + int(fxh * 20), ay - int(fzh * 20))], fill=RED, width=6); d.ellipse([ax - 7, ay - 7, ax + 7, ay + 7], fill=RED)
        if a.full and t in BOX:
            cx, cz = art.px(rooms[BOX[t]]['centroid'][0], rooms[BOX[t]]['centroid'][2]); d.rectangle([cx - 12, cz - 12, cx + 12, cz + 12], outline=GOLD, width=4)
        if not a.full and nxt and t >= t_leave:               # the replan target, priced
            cx, cz = art.px(rooms[nxt['room']]['centroid'][0], rooms[nxt['room']]['centroid'][2])
            d.rectangle([cx - 12, cz - 12, cx + 12, cz + 12], outline=GOLD, width=4)
        s = min((pw - 16) / im.width, (ph - 106) / im.height); im = im.resize((int(im.width * s) // 2 * 2, int(im.height * s) // 2 * 2), Image.NEAREST if s >= 1 else Image.BILINEAR)
        out = Image.new('RGB', (pw, ph), BG); out.paste(im, ((pw - im.width) // 2, 96 + (ph - 106 - im.height) // 2)); D = ImageDraw.Draw(out)
        cur_room = CUR.get(t, a.room) if a.full else a.room
        n_room = sum(1 for s, k in finds if s <= t and str(targets[k][0]) == cur_room); n_all = sum(1 for s, _ in finds if s <= t)
        n_in_room = sum(1 for x in targets if str(x[0]) == cur_room); vp = VP.get(t, f"1/{rooms[a.room]['n_viewpoints']}") if a.full else f"1/{rooms[a.room]['n_viewpoints']}"
        D.text((14, 8), f"collect every {cat}  [B={budget}]", font=font(26), fill=GOLD)
        D.text((14, 42), f"budget left {budget - t}/{budget}     collected {n_all}/{tr['n_reachable']}", font=font(18), fill=SOFT)
        D.text((14, 64), f"this room {n_room}/{n_in_room}     sweep viewpoints visited {vp}", font=font(18), fill=SOFT)
        if note:
            D.rectangle([14, ph - 64, pw - 14, ph - 14], fill=(0, 0, 0), outline=GOLD, width=2); D.text((24, ph - 54), note, font=font(18), fill=INK)
        return out

    def frame(t, big=None, note=None):
        fr = Image.new('RGB', (1920, 1080), BG); fr.paste(cam_pane(t, 1120, 1080, big), (0, 0)); fr.paste(map_pane(t, 784, 1080, note), (1136, 0)); return fr

    CUR, VP, BOX = {}, {}, {}
    if a.full:
        import imageio
        cell_room = {(i, j): rid for rid, r in rooms.items() for i, j in r['cells']}
        room_at = lambda q: cell_room.get((int((q[0] - gx0) // CELL), int((q[1] - gz0) // CELL)))
        dec = tr['decisions']; N = len(tr['actions']); fe = sorted(tr['found_events'])
        choices = [d for d in dec if d.get('room') in rooms]
        clears = [d for d in dec if d.get('room') in ('CLEARED', 'LEAVE')]
        apprs = [d['prim'] for d in dec if d.get('room') == 'APPROACH']
        big, note, hold = {}, {}, {}
        for ci, c in enumerate(choices):
            t0 = c['prim']; t1 = choices[ci + 1]['prim'] if ci + 1 < len(choices) else N
            room = c['room']; entered = next((t for t in range(t0, t1 + 1) if room_at(P[t]) == room), t1)
            q = c.get('q'); qs = f"{q:.1f}" if isinstance(q, (int, float)) else '-'
            txt = f"choose room: doorway vote {str(c.get('voted_type')).replace('_', ' ')}, Q {qs}, budget {c.get('B')}"
            big[t0] = ('CHOOSE ROOM', GREEN); hold[t0] = a.fps; BOX[t0] = room
            for t in range(t0, t1 + 1):
                CUR[t] = room; note[t] = txt if t < entered else None
                big.setdefault(t, ('MOVING TO ROOM', SOFT) if t < entered else ('SEARCHING', BLUE))
                if t <= entered: BOX[t] = room
            cl = next((d for d in clears if t0 < d['prim'] <= t1), None)
            nvp = rooms[room]['n_viewpoints']
            for t in range(t0, t1 + 1):        # 0/N while sweeping, k/N once the CLEARED record says how many it visited
                VP[t] = (str(cl.get('voted_type')).replace('vp', '') if (cl and t >= cl['prim']) else f"0/{nvp}")
            if cl and cl['prim'] < N:
                big[cl['prim']] = ('ROOM CLEARED', GOLD); hold[cl['prim']] = a.fps // 2
        for ta in apprs:
            tc = next((f for f in fe if f >= ta), ta)
            big[ta] = ('TARGET VISIBLE', ORANGE); hold[ta] = a.fps
            for t in range(ta + 1, tc): big[t] = ('APPROACHING', ORANGE)
        for f in fe: big[f] = ('COLLECTED +1', GOLD); hold[f] = 5
        big[N] = ('BUDGET EXHAUSTED' if tr['stop'].get('reason') == 'budget exhausted' else 'STOP', RED); hold[N] = a.fps
        seg_for = lambda t: any(s0 <= t < s1 for s0, s1, lab in segs if lab == 'APPROACHING')
        segs[:] = [(ta, next((f for f in fe if f >= ta), ta), 'APPROACHING') for ta in apprs]   # orange detours everywhere
        wr = imageio.get_writer(a.full, fps=a.fps, codec='libx264', quality=8, macro_block_size=2)
        for t in range(0, N + 1):
            fr = frame(t, big.get(t), note.get(t))
            for _ in range(max(a.per_step, hold.get(t, 0))): wr.append_data(np.asarray(fr))
        card = Image.new('RGB', (1920, 1080), BG); D = ImageDraw.Draw(card); f = font(72); txt = 'Search  \u2192  Discover  \u2192  Collect  \u2192  Replan'
        D.text(((1920 - D.textlength(txt, font=f)) / 2, 490), txt, font=f, fill=GOLD)
        for _ in range(2 * a.fps): wr.append_data(np.asarray(card))
        wr.close(); print('full clip', a.full, 'choices', len(choices), 'interruptions', len(apprs), 'credits', len(fe))
    if a.storyboard:
        first_credit = min(s for s, _ in room_finds)
        dd = next(d for d in tr['decisions'] if d.get('room') == a.room)
        keys = [(t_enter + 8, ('SEARCHING', BLUE), f"entered room: doorway vote {dd.get('voted_type')}, Q {dd.get('q')}, budget {dd.get('B')}"),
                (appr[0], ('TARGET VISIBLE', ORANGE), 'chair in view mid-sweep: sweep interrupted'),
                (first_credit, ('COLLECTED', GOLD), 'approach-and-collect: credited'),
                (appr[-1], ('TARGET VISIBLE', ORANGE), 'second interruption'),
                (t_leave - 1, ('COLLECTED', GOLD), f'{len(room_finds)} chairs credited'),
                (t_leave + 1, ('REPLAN', GREEN), f"room cleared after 1/{rooms[a.room]['n_viewpoints']} viewpoints; next: {nxt.get('room')} vote {nxt.get('voted_type')} Q {float(nxt.get('q') or 0):.1f}  B {nxt.get('B')}")]
        tiles = [frame(t, big, note).resize((960, 540), Image.BILINEAR) for t, big, note in keys]
        sheet = Image.new('RGB', (2 * 960 + 30, 3 * 540 + 40), (30, 30, 34))
        for i, tl in enumerate(tiles): sheet.paste(tl, (10 + (i % 2) * 970, 10 + (i // 2) * 550))
        sheet.save(a.storyboard); print('storyboard', a.storyboard, 'steps', [k[0] for k in keys])
    if a.video:
        import imageio
        wr = imageio.get_writer(a.video, fps=a.fps, codec='libx264', quality=8, macro_block_size=2)
        big_for = {}
        for s0, s1, lab in segs:
            for t in range(s0, s1): big_for[t] = (lab, BLUE if lab == 'SEARCHING' else ORANGE)
        for s, _ in room_finds: big_for[s] = ('COLLECTED +1', GOLD)
        dd = next(d for d in tr['decisions'] if d.get('room') == a.room)
        entry_note = f"entered room: doorway vote {dd.get('voted_type')}, Q {dd.get('q')}, budget {dd.get('B')}"
        for t in range(t_enter, t_leave + 1):
            fr = frame(t, big_for.get(t), entry_note if t < appr[0] else None)
            reps = 3
            if t in appr: fr = frame(t, ('TARGET VISIBLE', ORANGE)); reps = a.fps      # 1 s freeze
            if t in {s for s, _ in room_finds}: reps = 6
            for _ in range(reps): wr.append_data(np.asarray(fr))
        fr = frame(t_leave + 1, ('REPLAN', GREEN), f"cleared after 1/{rooms[a.room]['n_viewpoints']} viewpoints → next: {str(nxt.get('voted_type')).replace('_', ' ')}, Q {float(nxt.get('q') or 0):.1f}, B {nxt.get('B')}")
        for _ in range(a.fps): wr.append_data(np.asarray(fr))
        card = Image.new('RGB', (1920, 1080), BG); D = ImageDraw.Draw(card); f = font(72); txt = 'Search  →  Discover  →  Collect  →  Replan'
        D.text(((1920 - D.textlength(txt, font=f)) / 2, 490), txt, font=f, fill=GOLD)
        for _ in range(2 * a.fps): wr.append_data(np.asarray(card))
        wr.close(); print('video', a.video)


if __name__ == '__main__':
    main()
