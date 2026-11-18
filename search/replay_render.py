"""Replay-render a C4 trace (no planner, deterministic).

Re-executes the trace's stored actions from the recorded start pose. Two modes:

  --mode split (default): three artifacts per episode --
      <out>_cam.mp4  egocentric camera video (small step/budget caption)
      <out>_map.mp4  top-down map video: trajectory grows, stars light on finds,
                     budget banner
      <out>_gt.png   static ground-truth map: EVERY target instance marked --
                     gold star = found this episode, red ring = reachable but
                     missed, grey ring = excluded (start room / unreachable) --
                     final trajectory underneath, found/total banner.
  --mode combined: the classic single [RGB | map] video.

Crediting overlay replicates the executor exactly (P6 accounting) and the replayed
count is verified against the trace's recorded found_events.

  replay_render.py --trace <..._trace.json> --dataset {mp3d,hm3d} --out <stem-or-file>
"""
import argparse, collections, json, math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from c4_collect_expert import (make_sim, Recorder, apos, HM3D, MP3D, DG,
                               COLLECT_R, nav_grid, geo)
from build_room_graphs import parse_house, CELL

ACT = {"forward": "move_forward", "left": "turn_left", "right": "turn_right"}


def font(sz):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()


class MapArt:
    """Standalone top-down map renderer (same look as Recorder's right panel)."""
    def __init__(self, nav, x0, z0, target_h=480):
        self.nav, self.x0, self.z0 = nav, x0, z0
        self.nx, self.nz = nav.shape
        self.MS = max(5, min(16, 900 // max(self.nx, self.nz)))
        base = np.full((self.nz * self.MS, self.nx * self.MS, 3), 15, np.uint8)
        for (i, j) in np.argwhere(nav):
            base[(self.nz-1-j)*self.MS:(self.nz-j)*self.MS, i*self.MS:(i+1)*self.MS] = (66, 72, 86)
        self.base = base
        self.th = target_h

    def px(self, wx, wz):
        return (int((wx - self.x0) / CELL * self.MS), int((self.nz - (wz - self.z0) / CELL) * self.MS))

    def star(self, d, x, y, s, fill, outline=(0, 0, 0)):
        d.polygon([(x, y-s), (x+s//3, y-s//3), (x+s, y-s//3), (x+s//2.5, y+s//4),
                   (x+s//1.6, y+s), (x, y+s//2.2), (x-s//1.6, y+s), (x-s//2.5, y+s//4),
                   (x-s, y-s//3), (x-s//3, y-s//3)], fill=fill, outline=outline)

    def frame(self, traj, found, agent, banner, sub):
        im = Image.fromarray(self.base.copy()); d = ImageDraw.Draw(im)
        if len(traj) >= 2:
            d.line([self.px(x, z) for x, z in traj], fill=(90, 175, 255), width=4)
        for (ox, oz) in found:
            x, y = self.px(ox, oz); self.star(d, x, y, 11, (255, 208, 0))
        if agent is not None:
            (ax, ay), (fx, fz) = agent
            d.line([(ax, ay), (ax + int(fx*18), ay - int(fz*18))], fill=(255, 55, 55), width=5)
            d.ellipse([ax-5, ay-5, ax+5, ay+5], fill=(255, 55, 55))
        out = Image.new("RGB", (im.width, im.height + 52), (0, 0, 0))
        out.paste(im, (0, 52)); D = ImageDraw.Draw(out)
        D.text((8, 2), banner, font=font(20), fill=(255, 208, 0))
        D.text((8, 27), sub, font=font(15), fill=(200, 205, 215))
        w = max(1, int(out.width * self.th / out.height)) // 2 * 2
        return np.asarray(out.resize((w, self.th // 2 * 2)))

    def gt(self, traj, targets_xy, status, banner, sub):
        """status: per-target 'found' | 'missed' | 'excluded'."""
        im = Image.fromarray(self.base.copy()); d = ImageDraw.Draw(im)
        if len(traj) >= 2:
            d.line([self.px(x, z) for x, z in traj], fill=(60, 110, 160), width=3)
        for (ox, oz), st in zip(targets_xy, status):
            x, y = self.px(ox, oz)
            if st == 'found':
                self.star(d, x, y, 12, (255, 208, 0))
            elif st == 'missed':
                d.ellipse([x-9, y-9, x+9, y+9], outline=(255, 60, 60), width=4)
            else:
                d.ellipse([x-7, y-7, x+7, y+7], outline=(140, 140, 145), width=3)
        out = Image.new("RGB", (im.width, im.height + 74), (0, 0, 0))
        out.paste(im, (0, 74)); D = ImageDraw.Draw(out)
        D.text((8, 2), banner, font=font(21), fill=(255, 208, 0))
        D.text((8, 29), sub, font=font(14), fill=(200, 205, 215))
        D.text((8, 50), "gold star = found   red ring = missed (reachable)   grey ring = excluded",
               font=font(14), fill=(150, 155, 165))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', required=True)
    ap.add_argument('--dataset', choices=['mp3d', 'hm3d'], required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', required=True,
                    help='split mode: output STEM (writes _cam.mp4/_map.mp4/_gt.png); combined: .mp4 path')
    ap.add_argument('--mode', choices=['split', 'combined'], default='split')
    ap.add_argument('--scale', type=float, default=0.5)
    a = ap.parse_args()

    tr = json.load(open(a.trace))
    stem, cat, budget = tr['stem'], tr['cat'], tr['budget']
    start = str(tr['start_room'])

    if a.dataset == 'hm3d':
        G = json.load(open(os.environ.get('GRAPHS_HM3D', f"{DG}/room_graphs_hm3d_v2.json")))[stem]
        targets = [(rid, o[1], o[2], o[3], o[4])
                   for rid, r in G['rooms'].items() for o in r.get('objects_full', []) if o[0] == cat]
        glb = os.path.join(HM3D, G['scene_rel'], stem + ".glb")
    else:
        G = json.load(open(f"{DG}/room_graphs_mp3d.json"))[stem]
        _, objects = parse_house(f"{MP3D}/{stem}/{stem}.house")
        targets = [(ridx, o[0], o[1], o[2], r) for ridx, mp, o, r in objects if mp == cat]
        glb = f"{MP3D}/{stem}/{stem}.glb"

    level = G['rooms'][start]['level']; y = G['rooms'][start]['floor_y']

    sim = make_sim(glb, a.gpu)
    pf = sim.pathfinder
    nav, x0, z0 = nav_grid(pf, y)
    st = sim.get_agent(0).get_state()
    start_pos = np.asarray(pf.snap_point(G['rooms'][start]['centroid']), np.float32)
    st.position = start_pos
    sim.get_agent(0).set_state(st)

    counted, excluded = set(), set()
    for k, (ridx, ox, oy, oz, r) in enumerate(targets):     # P6
        if str(ridx) == start:
            counted.add(k); excluded.add(k)
    art = MapArt(nav, x0, z0)
    traj = [apos(sim)[[0, 2]].tolist()]
    found_xy, cam_frames, map_frames = [], [], []
    B = budget
    Fs = font(16)

    def cam_frame(step):
        rgb = np.asarray(sim.get_sensor_observations()["rgb"])[..., :3]
        im = Image.fromarray(rgb.copy())
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 24], fill=(0, 0, 0))
        d.text((6, 2), f"find every {cat.replace('_',' ')} | step {step} | budget left {max(B,0)}",
               font=Fs, fill=(255, 208, 0))
        if a.scale != 1.0:
            im = im.resize((int(im.width*a.scale)//2*2, int(im.height*a.scale)//2*2))
        return np.asarray(im)

    def agent_marker():
        st_ = sim.get_agent(0).get_state(); p = np.array(st_.position, np.float32)
        th = 2 * math.atan2(st_.rotation.y, st_.rotation.w)
        return (art.px(p[0], p[2]), (-math.sin(th), -math.cos(th)))

    def snap(step):
        cam_frames.append(cam_frame(step))
        map_frames.append(art.frame(traj, found_xy, agent_marker(),
                                    f"collect every {cat.replace('_',' ')}  [B={budget}]",
                                    f"budget left: {max(B,0)}/{budget}   collected: {len(found_xy)}"))
    snap(0)
    for i, w in enumerate(tr['actions'], 1):
        sim.step(ACT[w]); B -= 1
        traj.append(apos(sim)[[0, 2]].tolist())
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
            if ang <= 39.5 or d - r <= 0.3:
                counted.add(k); found_xy.append((ox, oz))
        if i % 2 == 0 or i == len(tr['actions']):
            snap(i)

    exp = len(counted) - len(excluded)
    want = len(tr.get('found_events', []))
    ver = "OK" if exp == want else f"MISMATCH trace={want}"

    import imageio
    outdir = os.path.dirname(a.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    if a.mode == 'combined':
        pass  # legacy path removed from fleet use; split is canonical
    # freeze tails
    for _ in range(12):
        cam_frames.append(cam_frames[-1]); map_frames.append(map_frames[-1])
    imageio.mimsave(a.out + "_cam.mp4", cam_frames, fps=18, macro_block_size=2)
    imageio.mimsave(a.out + "_map.mp4", map_frames, fps=18, macro_block_size=2)

    # ground-truth map: classify every same-floor instance
    # [POLICY P8, 2026-08-05] reachable = navmesh geodesic from the start pose (was:
    # room-graph BFS). Matches tsp_oracle's target set, so every agent's GT tile shows
    # the same denominator for the same (scene, cat, start).
    gt_xy, gt_status = [], []
    n_offfloor = 0
    for k, (ridx, ox, oy, oz, r) in enumerate(targets):
        rid = str(ridx)
        if rid not in G['rooms'] or G['rooms'][rid]['level'] != level:
            n_offfloor += 1
            continue
        gt_xy.append((ox, oz))
        p = pf.snap_point(np.array([float(ox), float(y), float(oz)], np.float32))
        reach = (p is not None and not np.any(np.isnan(p))
                 and math.isfinite(geo(pf, start_pos, p)))
        if k in counted and k not in excluded:
            gt_status.append('found')
        elif reach and rid != start:
            gt_status.append('missed')
        else:
            gt_status.append('excluded')
    n_reach = sum(1 for s in gt_status if s in ('found', 'missed'))
    gt = art.gt(traj, gt_xy, gt_status,
                f"GT: {cat.replace('_',' ')} in {stem}  [B={budget}]",
                f"found {exp} / {n_reach} reachable   excluded {gt_status.count('excluded')}   "
                f"off-floor {n_offfloor}   spent {budget - B}/{budget}")
    gt.save(a.out + "_gt.png")
    print(f"REPLAY {os.path.basename(a.trace)}: found={exp} [{ver}] -> {a.out}_{{cam,map}}.mp4 + _gt.png")
    sim.close()


if __name__ == '__main__':
    main()
