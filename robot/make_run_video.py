"""Run video from a --record directory: the odometry path growing in real time over the
lidar floor plan (left) and the RealSense stream (right), with step, remaining budget and
the current actions overlaid.

    python make_run_video.py --rec <dir> --out run.mp4 [--fps 10] [--speed 1.0] [--lidar-z 0.1,1.2]

The lidar cloud is drawn as an occupancy raster (5 cm cells, points within the z band,
accumulated over the run). If the recorded frame id is not an odometry frame the raster
will smear; pass --no-lidar to draw the path alone.
"""
import argparse
import json
import math
import os

import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

COL = {"forward": (13, 89, 230), "left": (220, 33, 16), "right": (252, 133, 3), "stop": (40, 40, 40)}


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def turbo(u):
    """cyan -> green -> yellow -> orange -> magenta with height, like the app's voxel map."""
    stops = [(0.0, (0, 230, 230)), (0.3, (60, 240, 90)), (0.55, (250, 240, 40)), (0.75, (255, 150, 30)), (1.0, (230, 40, 200))]
    u = min(1.0, max(0.0, u))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if u <= b:
            w = (u - a) / (b - a)
            return tuple(int(ca[i] + w * (cb[i] - ca[i])) for i in range(3))
    return stops[-1][1]


class Chase:
    """pinhole camera behind and above the dog, looking at a point ahead of it."""
    def __init__(self, W, H, hfov=70.0, back=3.0, up=2.0, ahead=2.5):
        self.W, self.H = W, H
        self.f = (W / 2) / math.tan(math.radians(hfov) / 2)
        self.back, self.up, self.ahead = back, up, ahead

    def set_pose(self, x, y, yaw):
        cx, cy = math.cos(yaw), math.sin(yaw)
        self.eye = np.array([x - self.back * cx, y - self.back * cy, self.up])
        tgt = np.array([x + self.ahead * cx, y + self.ahead * cy, 0.3])
        fwd = tgt - self.eye; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0, 0, 1.0])); right /= np.linalg.norm(right)
        upv = np.cross(right, fwd)
        self.R = np.stack([right, upv, fwd])                  # world -> camera (x right, y up, z forward)

    def project(self, P):
        """P (n,3) world -> (n,2) pixels, depth (n,); depth<=0.2 is behind the camera."""
        C = (P - self.eye) @ self.R.T
        z = C[:, 2]; zz = np.where(z > 0.2, z, np.nan)
        u = self.W / 2 + self.f * C[:, 0] / zz; v = self.H / 2 - self.f * C[:, 1] / zz
        return np.stack([u, v], axis=1), z


def draw_voxels(d, cam, vox, vsize, zmin, zmax):
    """vox: (n,3) voxel centres. Three visible faces per cube, painter's order."""
    if len(vox) == 0:
        return
    rel = vox - cam.eye; dist = np.linalg.norm(rel, axis=1)
    order = np.argsort(-dist)
    vox = vox[order]; dist = dist[order]
    keep = dist < 14.0
    vox = vox[keep]
    if len(vox) == 0:
        return
    h = vsize / 2
    # corners: 8 per voxel
    offs = np.array([[sx, sy, sz] for sz in (-h, h) for sy in (-h, h) for sx in (-h, h)])
    corners = vox[:, None, :] + offs[None, :, :]               # (n,8,3)
    px, z = cam.project(corners.reshape(-1, 3)); px = px.reshape(len(vox), 8, 2)
    top = [4, 5, 7, 6]                                          # +z face (indices with sz=h: 4..7 -> (x,y) order)
    faces = {"top": top, "xm": [0, 2, 6, 4], "xp": [1, 3, 7, 5], "ym": [0, 1, 5, 4], "yp": [2, 3, 7, 6]}
    for i in range(len(vox)):
        if np.isnan(px[i]).any():
            continue
        c = turbo((vox[i, 2] - zmin) / max(1e-6, zmax - zmin))
        side = tuple(int(v * 0.65) for v in c); edge = (20, 20, 20)
        vx, vy = vox[i, 0] - cam.eye[0], vox[i, 1] - cam.eye[1]
        for name in (("xm" if vx > 0 else "xp"), ("ym" if vy > 0 else "yp"), "top"):
            poly = [tuple(px[i, k]) for k in faces[name]]
            d.polygon(poly, fill=(c if name == "top" else side), outline=edge)


def render_3d(P, L, frames, decs, meta, a, R):
    W, H = 1280, 720
    cam = Chase(W, H)
    vsize = a.voxel
    t0, t1 = P[0, 0], P[-1, 0]
    z0, z1 = map(float, a.lidar_z.split(","))
    occ = {}                                                    # voxel key -> centre
    li = 0; ci = 0; cam_img = None; out = []
    dt = a.speed / a.fps
    f_big, f_small = font(20), font(15)
    B = meta.get("budget", decs[0]["N"] if decs else 0)
    T = t0
    while T <= t1 + 1e-6:
        while li < len(L) and L[li, 0] <= T:
            k = (int(math.floor(L[li, 1] / vsize)), int(math.floor(L[li, 2] / vsize)), int(math.floor(L[li, 3] / vsize)))
            if k not in occ:
                occ[k] = ((k[0] + 0.5) * vsize, (k[1] + 0.5) * vsize, (k[2] + 0.5) * vsize)
            li += 1
        while ci < len(frames) and frames[ci]["t"] <= T:
            try:
                cam_img = Image.open(os.path.join(R, "frames", frames[ci]["file"])).convert("RGB").resize((400, 225))
            except Exception:
                pass
            ci += 1
        sel = P[P[:, 0] <= T]; cur = sel[-1] if len(sel) else P[0]
        cam.set_pose(cur[1], cur[2], cur[3])
        im = Image.new("RGB", (W, H), (18, 18, 22)); d = ImageDraw.Draw(im)
        # floor grid, 1 m
        gx0, gx1 = math.floor(cur[1]) - 12, math.floor(cur[1]) + 12; gy0, gy1 = math.floor(cur[2]) - 12, math.floor(cur[2]) + 12
        for gx in range(gx0, gx1 + 1):
            pts, _ = cam.project(np.array([[gx, gy, 0.0] for gy in np.linspace(gy0, gy1, 25)]))
            seg = [tuple(p) for p in pts if not np.isnan(p).any()]
            if len(seg) > 1: d.line(seg, fill=(60, 60, 70), width=1)
        for gy in range(gy0, gy1 + 1):
            pts, _ = cam.project(np.array([[gx, gy, 0.0] for gx in np.linspace(gx0, gx1, 25)]))
            seg = [tuple(p) for p in pts if not np.isnan(p).any()]
            if len(seg) > 1: d.line(seg, fill=(60, 60, 70), width=1)
        vox = np.array(list(occ.values())) if occ else np.zeros((0, 3))
        draw_voxels(d, cam, vox, vsize, z0, z1)
        # path on the floor
        if len(sel) > 1:
            pts, _ = cam.project(np.column_stack([sel[:, 1], sel[:, 2], np.full(len(sel), 0.03)]))
            seg = [tuple(p) for p in pts if not np.isnan(p).any()]
            if len(seg) > 1:
                d.line(seg, fill=(255, 255, 255), width=6, joint="curve"); d.line(seg, fill=COL["forward"], width=3, joint="curve")
        # the dog: a box 0.7 x 0.3 x 0.3 at the current pose
        cx, cy, yaw = cur[1], cur[2], cur[3]; c_, s_ = math.cos(yaw), math.sin(yaw)
        body = np.array([[dx * c_ - dy * s_ + cx, dx * s_ + dy * c_ + cy, dz] for dz in (0.05, 0.35) for dy in (-0.15, 0.15) for dx in (-0.35, 0.35)])
        bp, _ = cam.project(body)
        if not np.isnan(bp).any():
            for face in ([4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [1, 3, 7, 5], [0, 2, 6, 4]):
                d.polygon([tuple(bp[k]) for k in face], fill=(200, 200, 210), outline=(30, 30, 30))
            nose, _ = cam.project(np.array([[cx + 0.5 * c_, cy + 0.5 * s_, 0.25]]))
            if not np.isnan(nose).any(): d.ellipse([nose[0][0] - 5, nose[0][1] - 5, nose[0][0] + 5, nose[0][1] + 5], fill=(255, 60, 60))
        if cam_img is not None:
            im.paste(cam_img, (20, 20)); d.rectangle([20, 20, 420, 245], outline=(255, 255, 255), width=2)
        last = None
        for e in decs:
            if e["t"] > T: break
            last = e
        hdr = f"{meta.get('category', '')}   budget {B}   t = {T - t0:5.1f} s"
        if last is not None:
            hdr += f"   remaining {last['N']:4d}   {' '.join(last['actions'])}"
        d.text((440, 24), hdr, fill=(240, 240, 240), font=f_big)
        travelled = float(np.sum(np.hypot(np.diff(sel[:, 1]), np.diff(sel[:, 2])))) if len(sel) > 1 else 0.0
        d.text((440, 52), f"path {travelled:.1f} m   voxels {len(occ)}", fill=(200, 200, 200), font=f_small)
        out.append(np.asarray(im))
        T += dt
    imageio.mimsave(a.out, out, fps=a.fps, macro_block_size=2)
    print(f"wrote {a.out}: {len(out)} frames at {a.fps} fps, 3d view, {len(occ)} voxels, run {t1 - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
    ap.add_argument("--lidar-z", default="0.1,1.2", help="z band (m) of lidar points drawn as the floor plan")
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--panel", type=int, default=720, help="map panel height in px")
    ap.add_argument("--view", choices=["2d", "3d"], default="2d", help="2d floor plan, or 3d voxel chase view like the Unitree app")
    ap.add_argument("--voxel", type=float, default=0.1, help="voxel size (m) for the 3d view")
    a = ap.parse_args()
    R = a.rec
    poses = load_jsonl(os.path.join(R, "poses.jsonl"))
    frames = load_jsonl(os.path.join(R, "frames", "index.jsonl"))
    decs = load_jsonl(os.path.join(R, "decisions.jsonl"))
    segs = load_jsonl(os.path.join(R, "segments.jsonl"))
    lid = load_jsonl(os.path.join(R, "lidar", "index.jsonl"))
    meta = json.load(open(os.path.join(R, "meta.json"))) if os.path.exists(os.path.join(R, "meta.json")) else {}
    if not poses:
        raise SystemExit("no poses recorded")
    P = np.array([[p["t"], p["x"], p["y"], p["yaw"]] for p in poses])
    t0, t1 = P[0, 0], P[-1, 0]
    B = meta.get("budget", decs[0]["N"] if decs else 0)

    # lidar occupancy raster (accumulated), 5 cm cells
    z0, z1 = map(float, a.lidar_z.split(","))
    pts, pts3 = [], []
    if lid and not a.no_lidar:
        for e in lid:
            xyz = np.load(os.path.join(R, "lidar", e["file"]))
            xyz = xyz[(xyz[:, 2] > z0) & (xyz[:, 2] < z1)]
            pts.append(np.column_stack([np.full(len(xyz), e["t"]), xyz[:, :2]]))
            pts3.append(np.column_stack([np.full(len(xyz), e["t"]), xyz]))
    L = np.concatenate(pts) if pts else np.zeros((0, 3))
    L3 = np.concatenate(pts3) if pts3 else np.zeros((0, 4))
    xs = np.concatenate([P[:, 1], L[:, 1]]) if len(L) else P[:, 1]
    ys = np.concatenate([P[:, 2], L[:, 2]]) if len(L) else P[:, 2]
    if len(L):                                   # clip the extent to the path neighbourhood
        keep = (L[:, 1] > P[:, 1].min() - 6) & (L[:, 1] < P[:, 1].max() + 6) & (L[:, 2] > P[:, 2].min() - 6) & (L[:, 2] < P[:, 2].max() + 6)
        L = L[keep]; xs, ys = np.concatenate([P[:, 1], L[:, 1]]), np.concatenate([P[:, 2], L[:, 2]])
    if a.view == "3d":
        return render_3d(P, L3, frames, decs, meta, a, R)
    m = 1.0
    x0, x1, y0, y1 = xs.min() - m, xs.max() + m, ys.min() - m, ys.max() + m
    H = a.panel; scale = H / (y1 - y0); W = int((x1 - x0) * scale)
    if W > 1400:
        scale = 1400 / (x1 - x0); W = 1400; H = int((y1 - y0) * scale)

    def px(x, y):                                # world (x forward, y left) -> image; y up on screen
        return (int((x - x0) * scale), int(H - (y - y0) * scale))

    cam_w = 640; cam_h = 360
    CW = W + cam_w + 30; CH = max(H, cam_h + 60) + 40
    f_big, f_small = font(18), font(14)
    dt = a.speed / a.fps
    out = []
    fi = 0; ci = 0; li = 0
    occ = np.zeros((H + 1, W + 1), dtype=np.uint16)
    T = t0
    cam_img = None
    while T <= t1 + 1e-6:
        while li < len(L) and L[li, 0] <= T:
            ix, iy = px(L[li, 1], L[li, 2]); occ[min(H, max(0, iy)), min(W, max(0, ix))] += 1; li += 1
        while ci < len(frames) and frames[ci]["t"] <= T:
            try:
                cam_img = Image.open(os.path.join(R, "frames", frames[ci]["file"])).convert("RGB").resize((cam_w, cam_h))
            except Exception:
                pass
            ci += 1
        base = np.full((H + 1, W + 1, 3), 252, dtype=np.uint8)
        if len(L):
            # 10 cm cells, darker: the recorded cloud is subsampled and at 2 Hz, so single hits must show
            occ2 = occ.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    occ2[max(0, dy):H + 1 + min(0, dy), max(0, dx):W + 1 + min(0, dx)] += occ[max(0, -dy):H + 1 + min(0, -dy), max(0, -dx):W + 1 + min(0, -dx)]
            shade = np.clip(235 - 70 * np.sqrt(occ2), 40, 235).astype(np.uint8)
            hit = occ2 > 0
            base[hit] = np.stack([shade[hit]] * 3, axis=1)
        im = Image.fromarray(base); d = ImageDraw.Draw(im)
        sel = P[P[:, 0] <= T]
        pts2 = [px(x, y) for x, y in sel[:, 1:3]]
        if len(pts2) > 1:
            d.line(pts2, fill=(255, 255, 255), width=9, joint="curve"); d.line(pts2, fill=COL["forward"], width=5, joint="curve")
        cur = sel[-1] if len(sel) else P[0]
        cx, cy = px(cur[1], cur[2]); hx, hy = px(cur[1] + 0.35 * math.cos(cur[3]), cur[2] + 0.35 * math.sin(cur[3]))
        d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(255, 255, 255), outline=(0, 0, 0), width=3); d.line([(cx, cy), (hx, hy)], fill=(0, 0, 0), width=4)
        sx, sy = px(P[0, 1], P[0, 2]); d.ellipse([sx - 8, sy - 8, sx + 8, sy + 8], fill=(20, 20, 20)); d.text((sx - 5, sy - 8), "S", fill=(255, 255, 255), font=f_small)
        last = None
        for e in decs:
            if e["t"] > T:
                break
            last = e
            if e.get("pose"):
                qx, qy = px(e["pose"][0], e["pose"][1])
                d.ellipse([qx - 4, qy - 4, qx + 4, qy + 4], fill=COL.get(e["actions"][0], (120, 120, 120)))
        cur_seg = None
        for sg in segs:
            if sg["t0"] > T:
                break
            done_ = sg["t1"] <= T
            sx_, sy_ = px(sg["start"][0], sg["start"][1]); tx_, ty_ = px(sg["target"][0], sg["target"][1])
            col = (60, 60, 60) if done_ else (200, 30, 30)
            if sg["kind"] == "forward":
                d.line([(sx_, sy_), (tx_, ty_)], fill=col, width=2)
            d.rectangle([tx_ - 5, ty_ - 5, tx_ + 5, ty_ + 5], outline=col, width=2)
            yaw_t = math.radians(sg["target"][2]); hx_, hy_ = px(sg["target"][0] + 0.3 * math.cos(yaw_t), sg["target"][1] + 0.3 * math.sin(yaw_t))
            d.line([(tx_, ty_), (hx_, hy_)], fill=col, width=2)
            if done_:
                ex_, ey_ = px(sg["end"][0], sg["end"][1]); d.ellipse([ex_ - 3, ey_ - 3, ex_ + 3, ey_ + 3], fill=(0, 140, 0))
            cur_seg = sg
        canvas = Image.new("RGB", (CW, CH), (252, 251, 248)); canvas.paste(im, (10, 34))
        if cam_img is not None:
            canvas.paste(cam_img, (W + 20, 34))
        dc = ImageDraw.Draw(canvas)
        elapsed = T - t0
        hdr = f"{meta.get('category', '')}  budget {B}   t = {elapsed:5.1f} s"
        if last is not None:
            hdr += f"   remaining {last['N']:4d}   acts: {' '.join(last['actions'])}"
        dc.text((10, 8), hdr, fill=(20, 20, 20), font=f_big)
        dc.text((W + 20, 34 + cam_h + 8), "RealSense, what the model sees", fill=(20, 20, 20), font=f_small)
        if cur_seg is not None:
            st_, tg_, en_ = cur_seg["start"], cur_seg["target"], cur_seg["end"]
            state = "done" if cur_seg["t1"] <= T else "moving"
            lines = [f"segment: {cur_seg['kind']} x{cur_seg['count']}  ({state})",
                     f"prev odom   x {st_[0]:6.2f}  y {st_[1]:6.2f}  yaw {st_[2]:6.1f}",
                     f"target      x {tg_[0]:6.2f}  y {tg_[1]:6.2f}  yaw {tg_[2]:6.1f}",
                     (f"end odom    x {en_[0]:6.2f}  y {en_[1]:6.2f}  yaw {en_[2]:6.1f}" if state == "done" else f"now odom    x {cur[1]:6.2f}  y {cur[2]:6.2f}  yaw {math.degrees(cur[3]):6.1f}"),
                     (f"error       {100 * cur_seg['pos_err_m']:.0f} cm, {cur_seg['yaw_err_deg']:+.1f} deg" if state == "done" else ""),
                     "squares = targets with heading tick, green dots = where each segment ended"]
            for i, ln in enumerate(lines):
                dc.text((W + 20, 34 + cam_h + 30 + 18 * i), ln, fill=(20, 20, 20), font=f_small)
        travelled = float(np.sum(np.hypot(np.diff(sel[:, 1]), np.diff(sel[:, 2])))) if len(sel) > 1 else 0.0
        dc.text((10, 34 + H + 6), f"path {travelled:.1f} m   lidar points {int(occ.sum())}", fill=(20, 20, 20), font=f_small)
        out.append(np.asarray(canvas))
        T += dt
    imageio.mimsave(a.out, out, fps=a.fps, macro_block_size=2)
    print(f"wrote {a.out}: {len(out)} frames at {a.fps} fps, {CW}x{CH}, run {t1 - t0:.0f} s, lidar frame id {meta.get('lidar_frame_id')}")


if __name__ == "__main__":
    main()
