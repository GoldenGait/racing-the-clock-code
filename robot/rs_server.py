"""Frame server for the RealSense D435i on the Go2's expansion computer.

librealsense is not installed on the Jetson, but the D435i's colour sensor is a
plain UVC node, so OpenCV reads it directly. This serves the latest colour frame
as JPEG over HTTP so the laptop-side client (go2_client.py --cam http://...) can
use the RealSense instead of the Go2's fisheye.

    scp robot/rs_server.py unitree@192.168.123.18:/tmp/
    ssh unitree@192.168.123.18 'nohup python3 /tmp/rs_server.py --dev 4 > /tmp/rs_server.log 2>&1 &'
    curl -s http://192.168.123.18:8090/frame.jpg -o /tmp/f.jpg     # from the laptop

GET /frame.jpg   latest colour frame (1280x720 JPEG)
GET /info        json: device, width, height, fps, frames served, and front_m/left_m/right_m
                 clearance from the depth stream (metres; null when depth is unavailable)
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

state = {"jpg": None, "n": 0, "t": 0.0, "w": 0, "h": 0, "dev": None, "fps": 0.0,
         "front_m": None, "left_m": None, "right_m": None, "depth_n": 0, "depth_dev": None}


def depth_loop(dev, w, h):
    """D435i depth over UVC (Z16, mm). Reports the nearest solid thing in three sectors of a
    band around the horizon: front (central third), left and right thirds. Uses the 10th
    percentile of valid pixels so a few noisy returns do not fake an obstacle."""
    import numpy as np
    cap = cv2.VideoCapture(dev)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Z16 "))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    state["depth_dev"] = dev
    while True:
        ok, d = cap.read()
        if not ok or d is None:
            time.sleep(0.05)
            continue
        if d.dtype != np.uint16:
            d = d.view(np.uint16).reshape(d.shape[0], -1) if d.size else d
        H, W = d.shape[:2]
        band = d[int(H * 0.35):int(H * 0.75), :].astype(np.float32) / 1000.0
        out = {}
        for name, sl in (("left_m", slice(0, W // 3)), ("front_m", slice(W // 3, 2 * W // 3)),
                         ("right_m", slice(2 * W // 3, W))):
            v = band[:, sl]
            v = v[(v > 0.15) & (v < 8.0)]
            out[name] = round(float(np.percentile(v, 10)), 2) if v.size > 200 else None
        state.update(out)
        state["depth_n"] += 1


def grab_loop(dev, w, h):
    cap = cv2.VideoCapture(dev)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    state["dev"] = dev
    last = time.time()
    while True:
        ok, f = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            now = time.time()
            state.update(jpg=buf.tobytes(), n=state["n"] + 1, t=now,
                         w=f.shape[1], h=f.shape[0], fps=1.0 / max(1e-3, now - last))
            last = now


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):        # keep the log quiet
        pass

    def do_GET(self):
        if self.path.startswith("/frame"):
            jpg = state["jpg"]
            if jpg is None:
                self.send_response(503); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("X-Frame-Age", "%.3f" % (time.time() - state["t"]))
            self.end_headers()
            self.wfile.write(jpg)
        else:
            body = json.dumps({k: v for k, v in state.items() if k != "jpg"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=4, help="/dev/videoN of the colour stream")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--depth-dev", type=int, default=-1, help="/dev/videoN of the depth stream (Z16) to expose clearance in /info; -1 (default) disables")
    ap.add_argument("--depth-res", default="640x480")
    a = ap.parse_args()
    threading.Thread(target=grab_loop, args=(a.dev, a.width, a.height), daemon=True).start()
    if a.depth_dev >= 0:
        dw, dh = map(int, a.depth_res.split("x"))
        threading.Thread(target=depth_loop, args=(a.depth_dev, dw, dh), daemon=True).start()
    print("rs_server: /dev/video%d -> http://0.0.0.0:%d/frame.jpg" % (a.dev, a.port), flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
