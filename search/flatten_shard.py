"""Sharded driver for flatten_uninavid.flatten_episode (v3 conventions:
shared out-dir + nav_videos/, per-shard records_part<k>.json, merge separately).

  python flatten_shard.py --trace-dir D --out-dir O --shard K --nshards N \
      --keep-every 12 --no-budget-frac 0.2 --vid-scale 240:180
"""
import argparse, glob, json, os

from flatten_uninavid import flatten_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace-dir', required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--shard', type=int, required=True); ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--keep-every', type=int, default=12)
    ap.add_argument('--no-budget-frac', type=float, default=0.2)
    ap.add_argument('--vid-scale', default='240:180')
    a = ap.parse_args()

    traces = sorted(glob.glob(os.path.join(a.trace_dir, '*_trace.json')))[a.shard::a.nshards]
    assert traces, 'empty shard'
    os.makedirs(a.out_dir, exist_ok=True)
    recs = []
    for i, tp in enumerate(traces):
        rs = flatten_episode(tp, a.out_dir, a.keep_every, a.no_budget_frac, a.vid_scale)
        recs.extend(rs)
        print(f'F{a.shard}: {i+1}/{len(traces)} ({len(recs)} records)', flush=True)
    json.dump(recs, open(os.path.join(a.out_dir, f'records_part{a.shard}.json'), 'w'), indent=1)
    print(f'FLATSHARD{a.shard}_DONE {len(recs)} records', flush=True)


if __name__ == '__main__':
    main()
