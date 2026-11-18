"""Record a fine-tuned policy rollout as a [camera | map] video.

Re-runs the exact benchmark episode (same per-episode decode seed, so the video
reproduces the benchmarked outcome) with rollout()'s record mode.

  policy_video.py <ckpt> <stem> <cat> <start_idx> <budget> <out.mp4> [label]
"""
from paths import DATA_ROOT
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get('OBJEVAL', os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get('UNINAVID', f'{DATA_ROOT}/Uni-NaVid'))


def main():
    ckpt, stem, cat, si, budget, out = sys.argv[1:7]
    label = sys.argv[7] if len(sys.argv) > 7 else os.path.basename(ckpt)
    import objectnav_uninavid as ov
    ov.CKPT = ckpt
    model = ov.Model(0)
    from rollout_gate import rollout
    import torch
    key = str((stem, cat, si, int(budget)))
    torch.manual_seed(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
    r = rollout(model, stem, cat, int(budget), 0, record=out, start_idx=int(si))
    print(f"POLICY_VIDEO {label} {key}: collected {r['collected']} spent {r['spent']} "
          f"stop {r['stop']} -> {out}")


if __name__ == '__main__':
    main()
