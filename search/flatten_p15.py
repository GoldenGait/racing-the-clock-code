"""P15-lite records from re-rendered episodes: no per-record clips.

Each record points at the shared per-episode video and an end_frame (index into the
stride-2 frame sequence), which the patched Uni-NaVid loader reads directly. Grid,
countdown, blind fraction and gates follow flatten_uninavid.py (P4a/P4b).

    python flatten_p15.py --episodes $RTC_DATA/p15lite/out --video-root episodes \
        --out $RTC_DATA/sft_data/sft_p15_records.json [--keep-every 8] [--no-budget-frac 0.2]
"""
import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instr                                                     # noqa: E402
from flatten_uninavid import grid_points, VOCAB                  # noqa: E402


STRIDE_DEFAULT = 2


def records_for(trace_path, video_rel, keep_every, no_budget_frac):
    tr = json.load(open(trace_path))
    ep = os.path.basename(trace_path).replace("_trace.json", "")
    acts, B, stride = tr["actions"], tr["budget"], int(tr.get("stride", STRIDE_DEFAULT))
    T = len(acts)
    dec_prims = sorted({int(d["prim"]) - (int(d["prim"]) % 2) for d in tr.get("decisions", [])} |
                       {T - (T % 2)})
    out = []
    for t in grid_points(T, dec_prims, keep_every):
        if t % stride:
            continue
        ans = (acts[t:t + 4] + ["stop"] * 4)[:4]
        assert len(ans) == 4 and all(x in VOCAB for x in ans), ans
        assert B - t >= 0
        blind = (int(hashlib.md5(f"{ep}_{t}".encode()).hexdigest(), 16) % 1000) < no_budget_frac * 1000
        task = instr.collect_task(tr["cat"], None if blind else B - t)
        out.append({
            "id": f"NAV_ID_COLLECT_{ep}_{t:04d}",
            "video": video_rel,
            "end_frame": t // stride,
            "conversations": [{"from": "human", "value": instr.wrap(task)},
                              {"from": "gpt", "value": " ".join(ans)}],
            "meta": {"t": t, "N": None if blind else B - t, "budget": B, "cat": tr["cat"],
                     "scene": tr["stem"], "is_decision": t in dec_prims, "mirror": tr.get("mirror", False),
                     "collected_so_far": sum(1 for f in tr.get("found_events", []) if f <= t)},
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="dir with s0/s1/s2/<ep>_{frames.mp4,trace.json}")
    ap.add_argument("--video-root", default="episodes", help="path prefix stored in records (relative to --video_folder at training)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-every", type=int, default=8)
    ap.add_argument("--stride-default", type=int, default=2, help="frames per primitive when the trace has no stride key (1 for raw c4 traces, whose frames.mp4 has one frame per primitive)")
    ap.add_argument("--no-budget-frac", type=float, default=0.2)
    ap.add_argument("--eval-scenes", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "EVAL_SCENES.txt"))
    a = ap.parse_args()
    global STRIDE_DEFAULT
    STRIDE_DEFAULT = a.stride_default
    evals = set(l.strip() for l in open(a.eval_scenes) if l.strip()) if os.path.exists(a.eval_scenes) else set()
    recs, n_ep, n_eval, n_bad = [], 0, 0, 0
    for tp in sorted(glob.glob(os.path.join(a.episodes, "s*", "*_trace.json"))):
        ep = os.path.basename(tp).replace("_trace.json", "")
        si = os.path.basename(os.path.dirname(tp))
        mp4 = tp.replace("_trace.json", "_frames.mp4")
        if not os.path.exists(mp4):
            n_bad += 1; continue
        tr = json.load(open(tp))
        if tr["stem"] in evals:
            n_eval += 1; continue
        recs += records_for(tp, f"{a.video_root}/{si}/{ep}_frames.mp4", a.keep_every, a.no_budget_frac)
        n_ep += 1
    json.dump(recs, open(a.out, "w"))
    n_blind = sum(1 for r in recs if r["meta"]["N"] is None)
    print(f"[P15] episodes {n_ep} (eval-scene skipped {n_eval}, missing video {n_bad}) -> records {len(recs)}, "
          f"blind {n_blind/max(1,len(recs)):.0%} -> {a.out}")


if __name__ == "__main__":
    main()
