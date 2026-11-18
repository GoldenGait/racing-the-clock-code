"""Object-level (TSP) expert vs room-level Bellman expert, side by side, 16:9.

Left pane: the object-level expert with the full map. Every chair is drawn from the
first frame, the route grows between them, a green ring marks each collection.
Right pane: the room-level expert. Only rooms it has entered are lit and only their
chairs are drawn; doorways it has observed are marked; at every room choice a callout
shows the doorway-evidence vote, the cross-scene prior Q and the budget left.

Both experts are replayed from their stored action lists in habitat from the same
start pose, crediting is re-run and checked against the stored counts.

    expert_duel.py --stem y9hTuugGdiq --cat chair --budget 400 --gpu 0 --out duel.mp4
"""
from paths import DATA_ROOT
import argparse, json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c4_collect_expert import make_sim, HM3D, COLLECT_R, apos       # noqa: E402
from build_room_graphs import nav_grid, CELL                          # noqa: E402
from replay_render import MapArt, font                                # noqa: E402

ACT = {'forward': 'move_forward', 'left': 'turn_left', 'right': 'turn_right'}
BG = (10, 10, 12); GOLD = (255, 208, 0); PINK = (232, 49, 138); BLUE = (90, 175, 255)
GREEN = (80, 220, 120); CYAN = (90, 220, 230); DIM = (34, 37, 44); LIT = (66, 72, 86)
INK = (235, 235, 235); SOFT = (185, 190, 200)


def replay(sim, G, start, cat, actions):
    """positions/headings per step plus the credited finds, using the executor's rule."""
    pf = sim.pathfinder
    st = sim.get_agent(0).get_state()
    st.position = np.asarray(pf.snap_point(G['rooms'][start]['centroid']), np.float32)
    st.rotation = np.array([0, 0, 0, 1], np.float32)
    sim.get_agent(0).set_state(st)
    targets = [(rid, o[1], o[2], o[3], o[4]) for rid, r in G['rooms'].items()
               for o in r.get('objects_full', []) if o[0] == cat]
    counted = {k for k, t in enumerate(targets) if str(t[0]) == str(start)}
    n_ex = len(counted)
    P, H, finds = [], [], []            # finds: (step, target idx)

    def pose():
        s_ = sim.get_agent(0).get_state(); p = np.array(s_.position, np.float32)
        th = 2 * math.atan2(s_.rotation.y, s_.rotation.w)
        return p, (-math.sin(th), -math.cos(th))

    def credit(step):
        p, (fx, fz) = pose()
        for k, (rid, ox, oy, oz, r) in enumerate(targets):
            if k in counted or abs(oy - p[1]) > 1.5:
                continue
            dx, dz = ox - p[0], oz - p[2]; d = math.hypot(dx, dz)
            if d - r > COLLECT_R:
                continue
            ang = math.degrees(abs(math.atan2(fx * dz - fz * dx, fx * dx + fz * dz)))
            if ang <= 39.5 or d - r <= 0.3:
                counted.add(k); finds.append((step, k))
    p, h = pose(); P.append(p[[0, 2]].tolist()); H.append(h)
    for i, a in enumerate(actions, 1):
        sim.step(ACT[a]); p, h = pose(); P.append(p[[0, 2]].tolist()); H.append(h); credit(i)
    return np.array(P), H, finds, targets, len(counted) - n_ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', required=True); ap.add_argument('--cat', default='chair')
    ap.add_argument('--budget', type=int, required=True); ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', required=True); ap.add_argument('--fps', type=int, default=18)
    ap.add_argument('--snap', type=int, default=2); ap.add_argument('--hold', type=float, default=2.5)
    ap.add_argument('--clean', action='store_true',
                    help='no pane titles and no room-choice callout; status line lifted clear of a web player\'s controls')
    ap.add_argument('--room-dir', default='hm3d_v3_traces',
                    help=f'room-level trace dir under {DATA_ROOT} (fig6_traces = P14 expert with re-sweeps)')
    a = ap.parse_args()
    B = f'{DATA_ROOT}'
    G = json.load(open(f'{HERE}/room_graphs_hm3d_v2.json'))[a.stem]
    T = json.load(open(f'{B}/tsp_traces/{a.stem}_{a.cat}_B{a.budget}_trace.json'))
    R = json.load(open(f'{B}/{a.room_dir}/{a.stem}_{a.cat}_B{a.budget}_trace.json'))
    start = str(R['start_room']); assert str(T['start_room']) == start, 'starts differ'
    level = G['rooms'][start]['level']; y = G['rooms'][start]['floor_y']
    grid = G['grids'][str(level)]; gx0, gz0 = grid['x0'], grid['z0']

    sim = make_sim(os.path.join(HM3D, G['scene_rel'], a.stem + '.glb'), a.gpu)
    nav, x0, z0 = nav_grid(sim.pathfinder, y)
    art = MapArt(nav, x0, z0)
    PT, HT, FT, targets, cT = replay(sim, G, start, a.cat, T['actions'])
    PR, HR, FR, _, cR = replay(sim, G, start, a.cat, R['actions'])
    sim.close()
    okT, okR = cT == T['collected'], cR == R['collected']
    print(f'replay: TSP {cT} (stored {T["collected"]}) {"OK" if okT else "MISMATCH"}; '
          f'room-level {cR} (stored {R["collected"]}) {"OK" if okR else "MISMATCH"}', flush=True)
    if not (okT and okR):
        raise SystemExit('crediting mismatch, not rendering')

    rooms = {rid: r for rid, r in G['rooms'].items() if r['level'] == level}
    cell_room = {(i, j): rid for rid, r in rooms.items() for i, j in r['cells']}

    def room_at(p):
        return cell_room.get((int((p[0] - gx0) // CELL), int((p[1] - gz0) // CELL)))

    def cells_px(rid):
        out = []
        for i, j in rooms[rid]['cells']:
            wx, wz = gx0 + (i + .5) * CELL, gz0 + (j + .5) * CELL
            out.append(art.px(wx, wz))
        return out
    doors = [e for e in G['edges'] if e['u'] in rooms and e['v'] in rooms]
    room_choices = [d for d in R['decisions'] if d['room'] in rooms]
    ms = art.MS

    # right-pane base with everything dimmed; rooms get re-lit as they are entered
    base_dim = art.base.copy(); base_dim[(base_dim == LIT).all(-1)] = DIM

    W, H = 1920, 1080; PW = 952; HDR = 118; FTR = 54
    if a.clean:
        HDR, FTR = 16, 54 + 116
    map_h = H - HDR - FTR

    def fit(im, pw, ph):
        s = min(pw / im.width, ph / im.height)
        w, h = max(2, int(im.width * s) // 2 * 2), max(2, int(im.height * s) // 2 * 2)
        return im.resize((w, h), Image.NEAREST if s >= 1 else Image.BILINEAR), s

    def draw_pane(kind, t):
        n_act = len(T['actions']) if kind == 'tsp' else len(R['actions'])
        tt = min(t, n_act)
        P, Hh, F = (PT, HT, FT) if kind == 'tsp' else (PR, HR, FR)
        base = art.base.copy() if kind == 'tsp' else base_dim.copy()
        im = Image.fromarray(base); d = ImageDraw.Draw(im)
        entered = []
        if kind == 'room':
            for k in range(tt + 1):
                rid = room_at(P[k])
                if rid and rid not in entered:
                    entered.append(rid)
            for rid in entered:
                for (px, py) in cells_px(rid):
                    d.rectangle([px - ms // 2, py - ms // 2, px + ms // 2, py + ms // 2], fill=LIT)
            # observed doorways: every door touching an entered room, joined to that room
            for e in doors:
                if e['u'] in entered or e['v'] in entered:
                    dx_, dz_ = art.px(e['door'][0], e['door'][2])
                    for rid in (e['u'], e['v']):
                        if rid in entered:
                            cx, cz = art.px(rooms[rid]['centroid'][0], rooms[rid]['centroid'][2])
                            d.line([(cx, cz), (dx_, dz_)], fill=(70, 120, 140), width=2)
                    d.rectangle([dx_ - 6, dz_ - 6, dx_ + 6, dz_ + 6], fill=CYAN)
        found_k = {k for s, k in F if s <= tt}
        for k, (rid, ox, oy, oz, r) in enumerate(targets):
            if str(rid) not in rooms or str(rid) == start:
                continue
            if kind == 'room' and str(rid) not in entered:
                continue                         # hidden until its room is entered
            x, yy = art.px(ox, oz)
            art.star(d, x, yy, 12, GOLD)
            if k in found_k:
                d.ellipse([x - 17, yy - 17, x + 17, yy + 17], outline=GREEN, width=4)
        if tt >= 1:
            d.line([tuple(art.px(*p)) for p in P[:tt + 1]], fill=PINK if kind == 'tsp' else BLUE, width=5)
        ax, ay = art.px(*P[tt]); fx, fz = Hh[tt]
        d.line([(ax, ay), (ax + int(fx * 20), ay - int(fz * 20))], fill=(255, 55, 55), width=6)
        d.ellipse([ax - 7, ay - 7, ax + 7, ay + 7], fill=(255, 55, 55), outline=(0, 0, 0))
        # room-choice callout for the latest decision at or before tt
        callout = None
        if kind == 'room':
            cur = [c for c in room_choices if c['prim'] <= tt]
            if cur and not a.clean:
                c = cur[-1]; rid = c['room']
                cx, cz = art.px(rooms[rid]['centroid'][0], rooms[rid]['centroid'][2])
                d.rectangle([cx - 10, cz - 10, cx + 10, cz + 10], outline=GOLD, width=3)
                q = c.get('q'); qs = f'{q:.1f}' if isinstance(q, (int, float)) else '-'
                callout = (f"doorway evidence → {c.get('voted_type', '?')}", f"prior Q {qs}   budget left {c.get('B', '?')}")
        pane = Image.new('RGB', (PW, H), BG); D = ImageDraw.Draw(pane)
        mim, s = fit(im, PW - 16, map_h - 16)
        mx, my = (PW - mim.width) // 2, HDR + (map_h - mim.height) // 2
        pane.paste(mim, (mx, my))
        if a.clean:
            pass
        elif kind == 'tsp':
            D.text((16, 12), 'Object-level expert', font=font(34), fill=GOLD)
            D.text((16, 58), 'Full map   ·   True target locations', font=font(20), fill=SOFT)
        else:
            D.text((16, 12), 'Room-level expert — ours', font=font(34), fill=GOLD)
            D.text((16, 58), 'Observed connectivity   ·   Doorway evidence   ·   Cross-scene semantic priors',
                   font=font(20), fill=SOFT)
        if callout:
            bx, by = mx + 14, my + 12
            D.rectangle([bx - 8, by - 6, bx + 470, by + 58], fill=(0, 0, 0), outline=GOLD, width=2)
            D.text((bx, by), callout[0], font=font(21), fill=GOLD)
            D.text((bx, by + 30), callout[1], font=font(19), fill=INK)
        n_found = len(found_k)
        n_reach = T['n_reachable']
        status = f"budget left {max(a.budget - tt, 0)}/{a.budget}    collected {n_found}/{n_reach}    step {tt}"
        if tt >= n_act and t > n_act:
            stop = (T if kind == 'tsp' else R)['stop']
            reason = stop.get('reason', '') if isinstance(stop, dict) else str(stop)
            status += f"    → {'done' if n_found == n_reach else 'stopped'}: {reason}"
        D.text((16, H - FTR + 14), status, font=font(21), fill=INK)
        return pane

    import imageio
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    wr = imageio.get_writer(a.out, fps=a.fps, codec='libx264', quality=8, macro_block_size=2)
    n_max = max(len(T['actions']), len(R['actions']))
    steps = list(range(0, n_max + 1, a.snap))
    if steps[-1] != n_max:
        steps.append(n_max)
    steps += [n_max + 1] * int(a.hold * a.fps)
    n = 0
    for t in steps:
        fr = Image.new('RGB', (W, H), BG)
        fr.paste(draw_pane('tsp', t), (0, 0)); fr.paste(draw_pane('room', t), (PW + 16, 0))
        ImageDraw.Draw(fr).line([(PW + 8, 0), (PW + 8, H)], fill=(60, 60, 66), width=2)
        wr.append_data(np.asarray(fr)); n += 1
    wr.close()
    print(json.dumps({'out': a.out, 'frames': n, 'tsp': {'found': cT, 'steps': len(T['actions'])},
                      'room': {'found': cR, 'steps': len(R['actions']), 'choices': len(room_choices),
                               'stop': R['stop']}}))


if __name__ == '__main__':
    main()
