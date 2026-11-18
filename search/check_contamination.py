"""Refuse to train on a dataset that overlaps the frozen evaluation scenes.

This exists because the first TSP/SPOC baseline was trained on data whose job
list was globbed from a working trace directory that had accumulated eval-scene
traces; 7% of its records came from the 10 benchmark scenes, which invalidated
the whole comparison. Run this on EVERY sft json before it is uploaded.

  python3 check_contamination.py <sft.json> [<sft.json> ...]     # exit 1 if dirty
"""
import os
from paths import DATA_ROOT
import ast
import collections
import json
import sys

FROZEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'frozen400', 'baselines_frozen.json')


def eval_scenes():
    return {ast.literal_eval(k)[0] for k in json.load(open(FROZEN))['expert']}


def main():
    ev = eval_scenes()
    bad = False
    for path in sys.argv[1:]:
        recs = json.load(open(path))
        hits = [r for r in recs if r.get('meta', {}).get('scene') in ev]
        per = collections.Counter(r['meta']['scene'] for r in hits)
        pct = 100 * len(hits) / max(len(recs), 1)
        status = 'CONTAMINATED' if hits else 'clean'
        print(f'{status}: {path}')
        print(f'  {len(recs)} records, {len({r["meta"]["scene"] for r in recs})} scenes, '
              f'{len(hits)} from eval scenes ({pct:.2f}%)')
        if hits:
            print(f'  offending scenes: {dict(per)}')
            bad = True
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
