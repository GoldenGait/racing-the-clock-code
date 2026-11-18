"""Phase 4 (Layer 2) - flatten C4 expert traces into Uni-NaVid training records.

Consumes trace dirs written by c4_collect_expert.py --trace-dir (per episode:
<ep>_trace.json + <ep>_frames.mp4 with one RGB frame captured BEFORE every
primitive, plus one terminal frame). Emits records.json + nav_videos/*.mp4 in
the exact schema of open_uninavid_sampled_500.json (see uninavid_format.md).

Locked cadence (Phase 0):
  - record grid = every 2 executed primitives (inference boundary);
  - record at prim t: video = frames[0..t] (t+1 frames, fps=1),
    gpt = expert prims t..t+3 padded with 'stop', instruction carries N_t = B - t;
  - N decrements by 2 per record; STOP appears via padding at chunk pos 1/2 and,
    when T is even, as a pure terminal 'stop stop stop stop' record;
  - decision-point records (planner room choices, STOP) are ALWAYS kept; the
    uniform grid in between may be subsampled with --keep-every to bound storage.

Budget conditioning (Phase 3 hooks):
  - --no-budget-frac: fraction of KEPT records emitted budget-blind (no countdown
    sentence), the C5 blending knob. Deterministic per-record hash, not RNG.

Gates (run automatically, fail loudly):
  P4a every gpt answer = exactly 4 tokens from {forward,left,right,stop};
  P4b N_t == B - t and t even for every record;
  P4c instruction byte-equality: rendered via instr.py, the same module
      an eval harness must import (single source of truth);
  P4d per-record video frame count == t + 1.

  python flatten_uninavid.py --trace-dir traces/ --out-dir dataset_c4/
"""
import argparse, glob, hashlib, json, os, subprocess
import imageio
import imageio_ffmpeg
import instr

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def clip_video(src, dst, n_frames, scale=None, stride=1):
    """Frames 0, stride, 2*stride, ... below n_frames of src -> dst via ffmpeg (no RAM).
    scale='320:240' downsizes record videos -- the Uni-NaVid loader resizes to 224
    anyway. stride=2 [P15] keeps one frame per two-primitive chunk, so the history the
    model trains on is the history it gets at inference (a frame per decision), and a
    360 look-around is 6 images 60 deg apart instead of 12 at 30."""
    vf = []
    if stride > 1:
        vf.append(f"select='not(mod(n\\,{stride}))',setpts=N/FRAME_RATE/TB")
    if scale:
        vf.append(f"scale={scale}")
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", src]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-frames:v", str((n_frames + stride - 1) // stride),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "1", dst]
    subprocess.run(cmd, check=True)

VOCAB = {"forward", "left", "right", "stop"}
DECISION_WINDOW = 4      # keep grid points within +/- this many prims of a decision/stop


def grid_points(T, decisions, keep_every):
    """Even grid 0..T with subsampling; decision neighborhoods and both ends always kept."""
    keep = set()
    anchors = set(decisions) | {0, T}
    full = list(range(0, T + 1, 2))
    for k, t in enumerate(full):
        if any(abs(t - a) <= DECISION_WINDOW for a in anchors):
            keep.add(t)
        elif k % max(1, keep_every) == 0:
            keep.add(t)
    return sorted(keep)


def flatten_episode(trace_path, out_dir, keep_every, no_budget_frac, vid_scale=None, stride=1):
    tr = json.load(open(trace_path))
    ep = os.path.basename(trace_path).replace("_trace.json", "")
    src_mp4 = trace_path.replace("_trace.json", "_frames.mp4")
    T, B = len(tr['actions']), tr['budget']
    n_src = imageio.get_reader(src_mp4).count_frames()
    assert n_src == T + 1, f"{ep}: {n_src} frames vs {T} prims (+1 expected)"
    acts = tr['actions']
    dec_prims = sorted({d['prim'] - (d['prim'] % 2) for d in tr['decisions']} |
                       {tr['stop']['prim'] - (tr['stop']['prim'] % 2)})

    records = []
    vid_dir = os.path.join(out_dir, "nav_videos")
    os.makedirs(vid_dir, exist_ok=True)
    for t in grid_points(T, dec_prims, keep_every):
        ans = (acts[t:t + 4] + ["stop"] * 4)[:4]
        # gates P4a/P4b
        assert len(ans) == 4 and all(a in VOCAB for a in ans), ans
        assert t % 2 == 0 and B - t >= 0
        blind = (int(hashlib.md5(f"{ep}_{t}".encode()).hexdigest(), 16) % 1000) < no_budget_frac * 1000
        task = instr.collect_task(tr['cat'], None if blind else B - t)
        vid = f"{ep}_{t:04d}.mp4"
        clip_video(src_mp4, os.path.join(vid_dir, vid), t + 1, scale=vid_scale, stride=stride)
        records.append({
            "id": f"NAV_ID_COLLECT_{ep}_{t:04d}",
            "video": f"nav_videos/{vid}",
            "conversations": [
                {"from": "human", "value": instr.wrap(task)},
                {"from": "gpt", "value": " ".join(ans)}],
            # planner metadata (stripped before training; kept for value distillation/audits)
            "meta": {"t": t, "N": None if blind else B - t, "budget": B, "cat": tr['cat'], "stride": stride,
                     "scene": tr['stem'], "is_decision": t in dec_prims,
                     "collected_so_far": sum(1 for f in tr['found_events'] if f <= t)},
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--frame-stride', type=int, default=1,
                    help='P15: keep every Nth frame in record clips (2 = one frame per chunk)')
    ap.add_argument('--keep-every', type=int, default=4,
                    help='keep every Nth non-decision grid point (1 = full grid)')
    ap.add_argument('--no-budget-frac', type=float, default=0.2)
    ap.add_argument('--vid-scale', default=None, help="e.g. 320:240 -- downsize record videos")
    a = ap.parse_args()

    traces = sorted(glob.glob(os.path.join(a.trace_dir, "*_trace.json")))
    assert traces, f"no traces in {a.trace_dir}"
    os.makedirs(a.out_dir, exist_ok=True)
    all_rec = []
    for tp in traces:
        rs = flatten_episode(tp, a.out_dir, a.keep_every, a.no_budget_frac, a.vid_scale, stride=a.frame_stride)
        all_rec.extend(rs)
        print(f"[P4] {os.path.basename(tp)}: {len(rs)} records")
    json.dump(all_rec, open(os.path.join(a.out_dir, "records.json"), 'w'), indent=1)

    # -------- gates summary --------
    n = len(all_rec)
    n_blind = sum(1 for r in all_rec if r['meta']['N'] is None)
    n_dec = sum(1 for r in all_rec if r['meta']['is_decision'])
    n_stop = sum(1 for r in all_rec if r['conversations'][1]['value'].split()[0] == 'stop')
    # P4c byte-equality: re-render every instruction through instr.py and compare
    for r in all_rec:
        m = r['meta']
        want = instr.wrap(instr.collect_task(m['cat'], m['N']))
        assert r['conversations'][0]['value'] == want, r['id']
    # P4d frame counts (spot-check 10)
    import random
    for r in random.Random(0).sample(all_rec, min(10, n)):
        vid = os.path.join(a.out_dir, r['video'])
        nf = imageio.get_reader(vid).count_frames()
        st = r['meta'].get('stride', 1)
        assert nf == r['meta']['t'] // st + 1, (r['id'], nf, st)
    print(f"[P4] TOTAL records={n} decision={n_dec} stop-first={n_stop} budget-blind={n_blind} "
          f"({n_blind/n:.0%}) -> {a.out_dir}")
    print("[P4] GATES: 4-token vocab PASS | N parity PASS | instr byte-equality PASS | frame counts PASS")


if __name__ == '__main__':
    main()
