"""Standalone Uni-NaVid inference: run any checkpoint over a video or image folder.

No habitat, no server, no robot -- just frames in, actions out. Use it to see how a
checkpoint behaves on real footage before wiring anything to hardware.

  # base Uni-NaVid on real video, objectnav-style instruction
  python run_uninavid.py --video walk.mp4 --category chair --style objectnav

  # our budget-conditioned fine-tune, with the countdown
  python run_uninavid.py --video walk.mp4 --category chair --style budget \
      --budget 400 --ckpt $RTC_DATA/sft_ckpts/uninavid-resweep-vid

  # annotated output video (actions burned into the frames)
  python run_uninavid.py --video walk.mp4 --category chair --out annotated.mp4

Env:
  PYTHONPATH=$RTC_DATA/Uni-NaVid:<repo>/search \
  CUDA_VISIBLE_DEVICES=0 python run_uninavid.py ...   (Uni-NaVid environment)
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw


def load_frames(args):
    """Yield RGB numpy frames at ~1 fps, cropped toward the training FOV."""
    if args.images:
        paths = sorted(glob.glob(os.path.join(args.images, "*")))
        srcs = (Image.open(p).convert("RGB") for p in paths)
    else:
        import imageio.v2 as imageio
        rd = imageio.get_reader(args.video)
        fps = rd.get_meta_data().get("fps", 30) or 30
        step = max(1, int(round(fps / args.sample_fps)))
        srcs = (Image.fromarray(f).convert("RGB")
                for i, f in enumerate(rd) if i % step == 0)
    for img in srcs:
        if args.src_hfov and args.src_hfov > 79.0:
            w, h = img.size
            keep = np.tan(np.radians(79.0 / 2)) / np.tan(np.radians(args.src_hfov / 2))
            cw, ch = int(w * keep), int(h * keep)
            img = img.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
        yield np.asarray(img.resize((640, 480)))


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--images", help="folder of frames, sorted by name")
    ap.add_argument("--ckpt", default=None, help="default: base Uni-NaVid")
    ap.add_argument("--category", default="chair")
    ap.add_argument("--style", default="objectnav", choices=["objectnav", "budget"])
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--frames-per-decision", type=int, default=2)
    ap.add_argument("--executed-per-decision", type=int, default=2,
                    help="primitives assumed spent per decision (budget countdown)")
    ap.add_argument("--sample-fps", type=float, default=1.0)
    ap.add_argument("--src-hfov", type=float, default=0.0,
                    help="source camera HFOV; >79 triggers a centre crop (fisheye: ~150)")
    ap.add_argument("--max-decisions", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None, help="write an annotated mp4")
    ap.add_argument("--trace", default=None, help="write a json trace")
    a = ap.parse_args()

    import objectnav_uninavid as ov
    if a.ckpt:
        ov.CKPT = a.ckpt
    import instr

    print(f"[run] loading {ov.CKPT}", flush=True)
    model = ov.Model(a.gpu)
    model.reset()

    word = ov.CAT_WORD.get(a.category, a.category)
    N = a.budget
    log, out_frames = [], []
    buf = []
    t_start = time.time()

    for frame in load_frames(a):
        buf.append(frame)
        if a.out:
            out_frames.append(frame)
        if len(buf) < a.frames_per_decision:
            continue
        goal = (instr.collect_task(a.category, N) if a.style == "budget"
                else f"Search for the {word} and stop when you reach it.")
        for f in buf:
            model.add_frame(f)
        t0 = time.time()
        raw = model.predict(goal)
        acts = [w for w in raw.replace(",", " ").replace(".", " ").split()
                if w in ("forward", "left", "right", "stop")][:4]
        dt = round(time.time() - t0, 2)
        k = len(log)
        print(f"[{k:03d}] N={N:>5} {acts}  ({dt}s)  :: {raw[:60]}", flush=True)
        log.append({"decision": k, "N": N, "goal": goal, "raw": raw,
                    "actions": acts, "latency_s": dt})
        if a.out:                                  # burn the decision into the frames
            for i in range(len(out_frames) - len(buf), len(out_frames)):
                im = Image.fromarray(out_frames[i]); d = ImageDraw.Draw(im)
                d.rectangle([0, 0, 640, 34], fill=(0, 0, 0))
                d.text((8, 10), f"{' '.join(acts)}   N={N}", fill=(255, 210, 0))
                out_frames[i] = np.asarray(im)
        if a.style == "budget":
            N = max(0, N - a.executed_per_decision)
        buf = []
        if "stop" in acts:
            print("[run] model called STOP", flush=True)
            break
        if a.style == "budget" and N <= 0:
            print("[run] budget exhausted", flush=True)
            break
        if len(log) >= a.max_decisions:
            break

    print(f"[run] {len(log)} decisions in {round(time.time()-t_start,1)}s", flush=True)
    if a.trace:
        json.dump({"ckpt": ov.CKPT, "category": a.category, "style": a.style,
                   "budget": a.budget, "decisions": log}, open(a.trace, "w"), indent=1)
        print("[run] trace ->", a.trace, flush=True)
    if a.out and out_frames:
        import imageio.v2 as imageio
        imageio.mimsave(a.out, out_frames, fps=2, macro_block_size=2)
        print("[run] video ->", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
