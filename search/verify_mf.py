"""Integrity sweep over a multi-floor collection run.

Checks every manifest job produced a trace, that the trace's own metadata agrees
with the filename it was written under, that the video has exactly one frame per
primitive plus the terminal observation, and that nothing from the frozen eval
scenes leaked in. Exits non-zero on any hard failure so it can gate the flatten.
"""
from paths import DATA_ROOT
import ast
import collections
import json
import os
import sys

import imageio.v3 as iio

MANIFEST = os.environ.get('MF_MANIFEST', f'{DATA_ROOT}/mp3d_mf_jobs.txt')
FROZEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'frozen400', 'baselines_frozen.json')
_P = os.environ.get('MF_PREFIX', f'{DATA_ROOT}/mp3d_mf')
DIRS = {'bellman': _P + '_bellman', 'tsp': _P + '_tsp'}
WORDS = {'forward', 'left', 'right', 'stop'}


def key_of(base):
    stem, rest = base.split('_L', 1)
    level, rest = rest.split('_', 1)
    cat, budget = rest.rsplit('_B', 1)
    return stem, int(level), cat, int(budget)


def main(check_video=True):
    rows = [l.split() for l in open(MANIFEST) if l.strip()]
    rows = [r[1:] if len(r) == 5 else r for r in rows]      # drop the dataset column
    jobs = {(s, int(L), c, int(b)) for s, L, c, b in rows}
    evil = {ast.literal_eval(k)[0] for k in json.load(open(FROZEN))['expert']}

    hard, soft, overcount = [], [], []
    reach = collections.defaultdict(dict)
    for exp, d in DIRS.items():
        seen = set()
        for f in sorted(os.listdir(d)):
            if not f.endswith('_trace.json'):
                continue
            base = f[:-11]
            k = key_of(base)
            try:
                t = json.load(open(os.path.join(d, f)))
            except Exception as e:
                hard.append(f'{exp}/{base}: unreadable trace ({e})'); continue
            seen.add(k)
            if k not in jobs:
                soft.append(f'{exp}/{base}: not in manifest'); continue
            if (t.get('stem'), t.get('level'), t.get('cat'), t.get('budget')) != k:
                hard.append(f'{exp}/{base}: metadata disagrees with filename')
            if t['stem'] in evil:
                hard.append(f'{exp}/{base}: FROZEN EVAL SCENE')
            acts = t.get('actions')
            if acts is None:
                hard.append(f'{exp}/{base}: no actions key'); continue
            if set(acts) - WORDS:
                hard.append(f'{exp}/{base}: unknown actions {sorted(set(acts) - WORDS)}')
            if t.get('spent', 0) > k[3]:
                hard.append(f'{exp}/{base}: spent {t["spent"]} over budget {k[3]}')
            if t.get('collected', 0) > t.get('n_reachable', 0):
                # expected, not a fault: P6 drops start-room instances from the
                # denominator but the crediting rule still counts them if the
                # agent passes one. score_frozen.py clamps with min(found, reach).
                overcount.append(f'{exp}/{base}')
            if len(t.get('found_events', [])) != t.get('collected', 0):
                hard.append(f'{exp}/{base}: found_events != collected')
            reach[k[:3]][exp] = t.get('n_reachable')
            mp4 = os.path.join(d, base + '_frames.mp4')
            if not acts:
                continue                       # degenerate episode, no video expected
            if not os.path.exists(mp4):
                hard.append(f'{exp}/{base}: missing video'); continue
            if check_video:
                try:
                    n = iio.improps(mp4, plugin='pyav').shape[0]
                except Exception as e:
                    hard.append(f'{exp}/{base}: unreadable video ({e})'); continue
                if n != len(acts) + 1:
                    hard.append(f'{exp}/{base}: {n} frames for {len(acts)} actions')
        missing = jobs - seen
        print(f'{exp}: {len(seen & jobs)}/{len(jobs)} manifest episodes present'
              + (f', MISSING {len(missing)}' if missing else ''))
        for k in sorted(missing)[:10]:
            hard.append(f'{exp}: missing {k}')

    dis = {k: v for k, v in reach.items() if len(v) == 2 and v['bellman'] != v['tsp']}
    print(f'reachability definition gap: {len(dis)}/{sum(1 for v in reach.values() if len(v)==2)} '
          f'tuples differ between the two experts (known, pre-existing)')
    degen = sorted(k for k, v in reach.items() if max(v.values() or [0]) == 0)
    print(f'degenerate (zero reachable on the floor): {len(degen)} tuples -> filter at flatten')
    for k in degen:
        print('   ', k)

    if overcount:
        print(f'collected > n_reachable: {len(overcount)} traces -- expected, the P6 '
              f'start-room exclusion shrinks the denominator but not the crediting rule; '
              f'score_frozen.py already clamps with min(found, reachable)')
    for m in soft[:10]:
        print('note:', m)
    if soft:
        print(f'note: {len(soft)} off-manifest files present (parked leftovers)')
    if hard:
        print(f'\nFAILURES ({len(hard)}):')
        for m in hard[:40]:
            print('  ', m)
        return 1
    print('\nALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main(check_video='--no-video' not in sys.argv))
