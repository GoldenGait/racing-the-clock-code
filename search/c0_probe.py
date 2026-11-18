"""C0 gate probe: does the fine-tuned model's FIRST action track the countdown?

Probe set: t=0 decision records from the 10 held-out EVAL scenes (never trained on),
restricted to (scene, category) pairs whose expert answers DIFFER between a low and a
high budget. At t=0 the video is the identical single start frame for every budget --
the countdown sentence is the ONLY difference between the two prompts.

For each pair we run the model on both prompts and score its first action against the
expert's first action for that budget. Reported:
  - balanced accuracy over expert-action classes (the C0 gate: clearly > 0.5 passes);
  - within-pair divergence rate (model gives different answers to the two prompts);
  - the same numbers for the BASE released model as a control (should be ~chance).

  python c0_probe.py --ckpt $RTC_DATA/sft_ckpts/uninavid-budget --dataset $RTC_DATA/dataset_eval --gpu 0
  python c0_probe.py --ckpt base --dataset $RTC_DATA/dataset_eval --gpu 0     # released-weights control
"""
from paths import DATA_ROOT
import argparse, collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{DATA_ROOT}/Uni-NaVid')

DATASET = None    # set from --dataset
EVAL_SCENES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EVAL_SCENES.txt')
BASE_CKPT = f'{DATA_ROOT}/Uni-NaVid/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2'
LOW, HIGH = 150, 800


def build_probes():
    ev = {l.strip() for l in open(EVAL_SCENES) if not l.startswith('#') and l.strip()}
    recs = json.load(open(f'{DATASET}/records.json'))
    t0 = collections.defaultdict(dict)
    for r in recs:
        m = r['meta']
        if m['scene'] in ev and m['t'] == 0 and m['N'] is not None:
            t0[(m['scene'], m['cat'])][m['budget']] = r
    probes = []
    for key, d in sorted(t0.items()):
        if LOW in d and HIGH in d:
            a_lo = d[LOW]['conversations'][1]['value'].split()[0]
            a_hi = d[HIGH]['conversations'][1]['value'].split()[0]
            if a_lo != a_hi:                       # divergent pair = informative probe
                probes.append((key, d[LOW], d[HIGH]))
    return probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help="checkpoint dir, or 'base' for released weights")
    ap.add_argument('--dataset', required=True,
                    help='dataset folder built by flatten_uninavid.py from traces of the evaluation scenes')
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    global DATASET
    DATASET = a.dataset
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(a.gpu))

    probes = build_probes()
    print(f"probe pairs (eval scenes, t=0, expert-divergent {LOW} vs {HIGH}): {len(probes)}")
    if not probes:
        return

    ckpt = BASE_CKPT if a.ckpt == 'base' else a.ckpt
    import objectnav_uninavid as ov
    ov.CKPT = ckpt                      # wrapper reads the module-level path
    import decord, numpy as np
    model = ov.Model(0)

    per_class = collections.defaultdict(lambda: [0, 0])   # expert action -> [correct, total]
    diverged = 0
    results = []
    for (scene, cat), r_lo, r_hi in probes:
        outs = {}
        for tag, rec in (('lo', r_lo), ('hi', r_hi)):
            vr = decord.VideoReader(os.path.join(DATASET, rec['video']))
            frame = vr[0].asnumpy()
            model.reset()
            model.add_frame(frame)
            task = rec['conversations'][0]['value'].split("task is: ")[1].split(". Analyze")[0].strip("'")
            txt = model.predict(task)
            first = next((w for w in txt.split() if w in ('forward', 'left', 'right', 'stop')), 'none')
            outs[tag] = first
        e_lo = r_lo['conversations'][1]['value'].split()[0]
        e_hi = r_hi['conversations'][1]['value'].split()[0]
        for tag, exp in (('lo', e_lo), ('hi', e_hi)):
            per_class[exp][1] += 1
            if outs[tag] == exp:
                per_class[exp][0] += 1
        if outs['lo'] != outs['hi']:
            diverged += 1
        results.append({'scene': scene, 'cat': cat, 'expert': [e_lo, e_hi],
                        'model': [outs['lo'], outs['hi']]})

    recalls = [c / t for c, t in per_class.values() if t]
    bal_acc = sum(recalls) / len(recalls) if recalls else 0.0
    n = len(probes)
    print(f"\n=== C0 PROBE RESULTS ({os.path.basename(ckpt)}) ===")
    print(f"balanced accuracy (first action vs expert): {bal_acc:.3f}   [gate: clearly > 0.5]")
    print(f"within-pair divergence rate (model changes answer with countdown): {diverged}/{n} = {diverged/n:.0%}")
    print("per-class recall:", {k: f"{c}/{t}" for k, (c, t) in sorted(per_class.items())})
    out = a.ckpt.rstrip('/').split('/')[-1]
    json.dump(results, open(f'{DATA_ROOT}/sft_ckpts/c0_results_{out}.json', 'w'), indent=1)
    print(f"per-pair detail -> {DATA_ROOT}/sft_ckpts/c0_results_{out}.json")


if __name__ == '__main__':
    main()
