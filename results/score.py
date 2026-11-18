"""Recompute the benchmark table from saved per-episode results.

Each results file maps an evaluation tuple "(scene, category, start_index, budget)"
to [found, reachable, steps, stop_reason]. The score is normalized collection N(B):
per tuple, min(found, reachable) / reachable, averaged over the tuples of a budget.
Tuples with no reachable target are dropped. Intervals are 95% percentile
bootstraps (10,000 resamples, seed 0), reported as half-widths.

    python results/score.py                       # every method in results/frozen400
    python results/score.py my_run/results_*.json # add your own run to the table
"""
import ast, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'frozen400')
BUDGETS = [150, 400, 800, 2000]
REACH = json.load(open(os.path.join(R, 'reachable.json')))   # denominators shared by all methods


def clean(d):
    return {k: v for k, v in d.items() if isinstance(v, list) and v and v[0] != 'ERROR'}


def load_run(paths):
    out = {}
    for p in paths:
        out.update(clean(json.load(open(p))))
    return out


def main():
    methods = {
        'Greedy frontier expert': clean(json.load(open(os.path.join(R, 'baselines_frozen.json')))['greedy']),
        'Room-level Bellman expert': load_run([os.path.join(R, 'room_level_expert.json')]),
        'Pretrained VLN': load_run([os.path.join(R, 'pretrained_vln.json')]),
        'VLN trained on room-level expert': load_run([os.path.join(R, 'room_level_vln.json')]),
    }
    if len(sys.argv) > 1:
        methods['Your run'] = load_run([f for a in sys.argv[1:] for f in glob.glob(a)])
    rng = np.random.default_rng(0)
    print(f"{'policy':<34}" + ''.join(f"{'B=' + str(B):>15}" for B in BUDGETS))
    for name, m in methods.items():
        cells = []
        for B in BUDGETS:
            ks = sorted(k for k in m if k in REACH and ast.literal_eval(k)[3] == B and REACH[k] > 0)
            if not ks:
                cells.append(f"{'-':>15}")
                continue
            r = np.array([min(m[k][0], REACH[k]) / REACH[k] for k in ks])
            bs = r[rng.integers(0, len(r), (10000, len(r)))].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cells.append(f"{100 * r.mean():>7.1f} ±{50 * (hi - lo):<4.1f}".rjust(15))
        print(f'{name:<34}' + ''.join(cells))
    print('\nN(B): mean % of reachable targets collected, ± 95% CI half-width.')


if __name__ == '__main__':
    main()
