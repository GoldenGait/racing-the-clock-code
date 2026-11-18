"""Robot inference server: the budget-VLN model behind a TCP JSON-lines API.

Wraps the bench's Model class (objectnav_uninavid) -- the exact code path every
frozen-400 number came from -- with the episode state a robot needs: the
authoritative budget countdown, the training-parity instruction from
instr.collect_task, and full sim-schema trace logging so real runs are
replayable by the existing tooling.

Protocol (newline-delimited JSON over TCP):
  {"cmd":"start", "category":"chair", "budget":150}
      -> {"ok":true, "N":150, "instruction":"..."}
  {"cmd":"act", "frames":["<b64 jpeg>", ...]}       # frames since last call, 1 fps
      -> {"actions":["forward","left",...], "N":146, "raw":"...", "latency_s":0.8}
  {"cmd":"end"}
      -> {"ok":true, "trace":"<path>"}

Run (uninavid env; no habitat anywhere near this process):
  PYTHONPATH=$RTC_DATA/Uni-NaVid:<repo>/search \
  CUDA_VISIBLE_DEVICES=<gpu> OV_CKPT=<ckpt_dir> \
  .../envs/uninavid/bin/python nav_server.py --port 8877
"""
DATA_ROOT = os.environ.get('RTC_DATA', os.path.expanduser('~/rtc_data'))
import argparse
import base64
import io
import json
import os
import socketserver
import threading
import time

import numpy as np
from PIL import Image

# checkpoint override BEFORE the module reads it
import objectnav_uninavid as ov
if os.environ.get("OV_CKPT"):
    ov.CKPT = os.environ["OV_CKPT"]
import instr

WORD2ACT = {"stop": "stop", "forward": "forward", "left": "left", "right": "right"}
TRACE_DIR = f"{DATA_ROOT}/robot/traces"


class Episode:
    def __init__(self, model, category, budget, style="budget"):
        self.model = model
        self.cat = category
        self.N = budget
        self.budget = budget
        self.style = style      # "budget" = our fine-tune contract; "objectnav" = base Uni-NaVid
        self.log = []
        self.t0 = time.strftime("%Y%m%d_%H%M%S")
        model.reset()

    def instruction(self):
        if self.style == "objectnav":
            word = ov.CAT_WORD.get(self.cat, self.cat)
            return f"Search for the {word} and stop when you reach it."
        return instr.collect_task(self.cat, self.N)

    def save_frames(self, frames_b64):
        """Keep what the model actually saw -- otherwise a run is unverifiable."""
        d = os.path.join(TRACE_DIR, f"{self.t0}_{self.cat}_B{self.budget}_frames")
        os.makedirs(d, exist_ok=True)
        for fb in frames_b64:
            n = len(os.listdir(d))
            open(os.path.join(d, f"{n:04d}.jpg"), "wb").write(base64.b64decode(fb))
        return d

    def act(self, frames_b64, executed=None, pose=None):
        # the caller executes only the first few of the 4 returned primitives
        # (receding horizon), so the countdown must follow what was actually
        # spent, not what was proposed.
        t = time.time()
        self.save_frames(frames_b64)
        if executed is not None:
            self.N = max(0, self.N - int(executed))
        for fb in frames_b64:
            rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(fb))).convert("RGB"))
            self.model.add_frame(rgb)
        goal = self.instruction()
        raw = self.model.predict(goal)
        acts = [WORD2ACT[w] for w in raw.replace(",", " ").replace(".", " ").split() if w in WORD2ACT][:4]
        if executed is None:               # caller executes all of them
            self.N = max(0, self.N - len(acts))
        self.log.append({"t": time.time(), "n_frames": len(frames_b64), "goal": goal,
                         "raw": raw, "actions": acts, "N_after": self.N,
                         "executed_reported": executed,
                         "pose": pose,            # [x, y, yaw] from the dog's odometry at this observation
                         "latency_s": round(time.time() - t, 2)})
        return acts, raw, round(time.time() - t, 2)

    def close(self):
        os.makedirs(TRACE_DIR, exist_ok=True)
        p = os.path.join(TRACE_DIR, f"{self.t0}_{self.cat}_B{self.budget}.json")
        json.dump({"category": self.cat, "budget": self.budget, "decisions": self.log},
                  open(p, "w"), indent=1)
        return p


class Handler(socketserver.StreamRequestHandler):
    # a client that dies without closing (Ctrl-C on the robot) must not wedge the
    # server: threaded accept + per-connection read timeout.
    timeout = 900

    def handle(self):
        print(f"[server] client {self.client_address}", flush=True)
        ep = None
        for line in self.rfile:
            try:
                msg = json.loads(line)
                cmd = msg.get("cmd")
                if cmd == "start":
                    if ep: ep.close()
                    ep = Episode(self.server.model, msg["category"], int(msg["budget"]),
                                 msg.get("style", "budget"))
                    out = {"ok": True, "N": ep.N, "instruction": ep.instruction()}
                elif cmd == "act":
                    if not ep:
                        out = {"error": "no episode; send start first"}
                    else:
                        with self.server.lock:      # one episode through the model at a time
                            acts, raw, lat = ep.act(msg["frames"], msg.get("executed"), msg.get("pose"))
                        out = {"actions": acts, "N": ep.N, "raw": raw, "latency_s": lat,
                               "done": ("stop" in acts) or ep.N <= 0}
                elif cmd == "end":
                    out = {"ok": True, "trace": ep.close() if ep else None}; ep = None
                else:
                    out = {"error": f"unknown cmd {cmd}"}
            except Exception as e:  # noqa: robot loop must never hang on a bad message
                out = {"error": repr(e)}
            self.wfile.write((json.dumps(out) + "\n").encode())
            self.wfile.flush()
        if ep:
            ep.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8877)
    a = ap.parse_args()
    print(f"[server] loading {ov.CKPT}", flush=True)
    model = ov.Model(0)

    class Srv(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    srv = Srv(("0.0.0.0", a.port), Handler)
    srv.model = model
    srv.lock = threading.Lock()
    print(f"[server] READY on :{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
