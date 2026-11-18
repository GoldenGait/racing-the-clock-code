"""Live view of a run while it is recording: tails the --record folder the client is writing
and redraws every 0.3 s. Path over the lidar floor plan, segment targets (squares with a
heading tick) and where each segment ended (green dots), the current segment in red, the
latest RealSense frame, and the odometry readout.

    python live_view.py --rec ~/robot/rec_slow400_1234        # in a second terminal, during the run
    python live_view.py --latest                              # newest ~/robot/rec_* folder

Press q to quit. Everything here is read-only; the client is untouched.
"""
import argparse
import glob
import json
import math
import os
import time
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

COL = {"forward": (13, 89, 230), "left": (220, 33, 16), "right": (252, 133, 3), "stop": (40, 40, 40)}


def font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


class Tail:
    """incremental jsonl reader."""
    def __init__(self, path):
        self.path, self.pos, self.rows = path, 0, []

    def poll(self):
        if not os.path.exists(self.path):
            return self.rows
        with open(self.path) as f:
            f.seek(self.pos)
            for line in f:
                if line.endswith("\n"):
                    try:
                        self.rows.append(json.loads(line))
                    except Exception:
                        pass
                    self.pos = f.tell()
        return self.rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="")
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--size", type=int, default=760, help="map panel size in px")
    ap.add_argument("--lidar-z", default="0.1,1.2")
    a = ap.parse_args()
    R = a.rec
    if a.latest or not R:
        cands = sorted(glob.glob(os.path.expanduser("~/robot/rec_*")), key=os.path.getmtime)
        if not cands:
            raise SystemExit("no ~/robot/rec_* folder yet")
        R = cands[-1]
    R = os.path.expanduser(R)
    print("live view of", R)
    z0, z1 = map(float, a.lidar_z.split(","))
    poses, segs, decs, lidx, fidx = (Tail(os.path.join(R, p)) for p in ("poses.jsonl", "segments.jsonl", "decisions.jsonl", "lidar/index.jsonl", "frames/index.jsonl"))
    lid_pts = []; lid_seen = 0
    S = a.size; camw, camh = 480, 270
    root = tk.Tk(); root.title("run live view"); label = tk.Label(root); label.pack()
    root.bind("q", lambda e: root.destroy())
    f_big, f_small = font(16), font(13)
    state = {"photo": None}

    def redraw():
        nonlocal lid_seen
        P = poses.poll(); SG = segs.poll(); D = decs.poll(); LI = lidx.poll(); FI = fidx.poll()
        for e in LI[lid_seen:]:
            try:
                xyz = np.load(os.path.join(R, "lidar", e["file"]))
                xyz = xyz[(xyz[:, 2] > z0) & (xyz[:, 2] < z1)]
                lid_pts.append(xyz[:, :2])
            except Exception:
                pass
            lid_seen += 1
        canvas = Image.new("RGB", (S + camw + 30, max(S, camh + 200) + 40), (252, 251, 248))
        d = ImageDraw.Draw(canvas)
        if P:
            xs = np.array([p["x"] for p in P]); ys = np.array([p["y"] for p in P])
            L = np.concatenate(lid_pts) if lid_pts else np.zeros((0, 2))
            if len(L):
                keep = (L[:, 0] > xs.min() - 6) & (L[:, 0] < xs.max() + 6) & (L[:, 1] > ys.min() - 6) & (L[:, 1] < ys.max() + 6); L = L[keep]
            ax = np.concatenate([xs, L[:, 0]]) if len(L) else xs; ay = np.concatenate([ys, L[:, 1]]) if len(L) else ys
            x0, x1, y0, y1 = ax.min() - 1, ax.max() + 1, ay.min() - 1, ay.max() + 1
            span = max(x1 - x0, y1 - y0); sc = S / span
            cx0, cy0 = (x0 + x1) / 2, (y0 + y1) / 2
            px = lambda x, y: (int(S / 2 + (x - cx0) * sc), int(S / 2 - (y - cy0) * sc))
            m = Image.new("RGB", (S, S), (252, 251, 248)); dm = ImageDraw.Draw(m)
            if len(L):
                arr = np.zeros((S, S), dtype=np.uint16)
                u = ((L[:, 0] - cx0) * sc + S / 2).astype(int); v = (S / 2 - (L[:, 1] - cy0) * sc).astype(int)
                ok = (u >= 0) & (u < S) & (v >= 0) & (v < S); np.add.at(arr, (v[ok], u[ok]), 1)
                base = np.asarray(m).copy(); hit = arr > 0
                shade = np.clip(230 - 60 * np.sqrt(arr), 40, 230).astype(np.uint8)
                base[hit] = np.stack([shade[hit]] * 3, axis=1); m = Image.fromarray(base); dm = ImageDraw.Draw(m)
            pts = [px(x, y) for x, y in zip(xs, ys)]
            if len(pts) > 1:
                dm.line(pts, fill=(255, 255, 255), width=8, joint="curve"); dm.line(pts, fill=COL["forward"], width=4, joint="curve")
            for sg in SG:
                done_ = "end" in sg
                tx_, ty_ = px(sg["target"][0], sg["target"][1]); sx_, sy_ = px(sg["start"][0], sg["start"][1])
                col = (60, 60, 60)
                if sg["kind"] == "forward": dm.line([(sx_, sy_), (tx_, ty_)], fill=col, width=2)
                dm.rectangle([tx_ - 5, ty_ - 5, tx_ + 5, ty_ + 5], outline=col, width=2)
                yt = math.radians(sg["target"][2]); hx_, hy_ = px(sg["target"][0] + 0.3 * math.cos(yt), sg["target"][1] + 0.3 * math.sin(yt)); dm.line([(tx_, ty_), (hx_, hy_)], fill=col, width=2)
                ex_, ey_ = px(sg["end"][0], sg["end"][1]); dm.ellipse([ex_ - 3, ey_ - 3, ex_ + 3, ey_ + 3], fill=(0, 140, 0))
            for e in D:
                if e.get("pose"):
                    qx, qy = px(e["pose"][0], e["pose"][1]); dm.ellipse([qx - 4, qy - 4, qx + 4, qy + 4], fill=COL.get(e["actions"][0], (120, 120, 120)))
            cur = P[-1]; cx, cy = px(cur["x"], cur["y"]); hx, hy = px(cur["x"] + 0.35 * math.cos(cur["yaw"]), cur["y"] + 0.35 * math.sin(cur["yaw"]))
            dm.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(255, 255, 255), outline=(0, 0, 0), width=3); dm.line([(cx, cy), (hx, hy)], fill=(200, 30, 30), width=4)
            sx, sy = px(xs[0], ys[0]); dm.ellipse([sx - 7, sy - 7, sx + 7, sy + 7], fill=(20, 20, 20))
            canvas.paste(m, (10, 34))
            last_d = D[-1] if D else None; last_s = SG[-1] if SG else None
            hdr = f"t = {P[-1]['t'] - P[0]['t']:5.1f} s   odom x {cur['x']:6.2f}  y {cur['y']:6.2f}  yaw {math.degrees(cur['yaw']):6.1f}"
            if last_d is not None:
                hdr += f"   remaining {last_d['N']}   {' '.join(last_d['actions'])}"
            d.text((10, 8), hdr, fill=(20, 20, 20), font=f_big)
            lines = [f"poses {len(P)}  segments {len(SG)}  decisions {len(D)}  lidar clouds {len(LI)}  frames {len(FI)}"]
            if last_s is not None:
                lines += [f"last segment: {last_s['kind']} x{last_s['count']}",
                          f"  start  x {last_s['start'][0]:6.2f}  y {last_s['start'][1]:6.2f}  yaw {last_s['start'][2]:6.1f}",
                          f"  target x {last_s['target'][0]:6.2f}  y {last_s['target'][1]:6.2f}  yaw {last_s['target'][2]:6.1f}",
                          f"  end    x {last_s['end'][0]:6.2f}  y {last_s['end'][1]:6.2f}  yaw {last_s['end'][2]:6.1f}",
                          f"  error  {100 * last_s['pos_err_m']:.0f} cm, {last_s['yaw_err_deg']:+.1f} deg"]
            lines += ["red tick = current heading (forward); squares = targets; green = segment ends"]
            for i, ln in enumerate(lines):
                d.text((S + 20, 34 + camh + 12 + 18 * i), ln, fill=(20, 20, 20), font=f_small)
        else:
            d.text((10, 8), "waiting for poses.jsonl ...", fill=(20, 20, 20), font=f_big)
        if FI:
            try:
                im = Image.open(os.path.join(R, "frames", FI[-1]["file"])).convert("RGB").resize((camw, camh))
                canvas.paste(im, (S + 20, 34))
            except Exception:
                pass
        state["photo"] = ImageTk.PhotoImage(canvas); label.configure(image=state["photo"])
        root.after(300, redraw)

    root.after(0, redraw)
    root.mainloop()


if __name__ == "__main__":
    main()
