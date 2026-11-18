"""Stage 3 of automatic scene annotation: score the pseudo-labels.

Runs the annotator's output against the human annotation that ships with the
scene, so pseudo-label quality is a measured number rather than an assumption.
Matching is greedy nearest-first within a distance gate, per category.

  python annot_eval.py --pred objects.json --stem DNWbUAJYsPy --dataset hm3d
"""
import argparse
import json
import math
import os

DG = os.path.dirname(os.path.abspath(__file__))
CATS = ['chair', 'bed', 'sofa', 'toilet', 'tv_monitor', 'plant']
GATE = 1.5          # metres between predicted and true centre to count as a hit
DUP = 3.0           # an unmatched prediction this close to a real object of its own
                    # class is a duplicate of it, not an invented object


def truth(dataset, stem):
    fn = 'room_graphs_hm3d_v2.json' if dataset == 'hm3d' else 'room_graphs_mp3d.json'
    G = json.load(open(f'{DG}/{fn}'))[stem]
    return [(o[0], (o[1], o[2], o[3])) for r in G['rooms'].values()
            for o in r.get('objects_full', []) if o[0] in CATS]


def match(pred, gt, gate=GATE):
    """greedy nearest-first, one prediction to one truth.

    Splits the misses by how much they actually cost us. A duplicate sits on a
    real object that another prediction already claimed: it inflates the
    reachable count slightly but the agent still ends up beside a real object,
    so what the model sees is correct. A phantom puts a target where nothing of
    that class exists, so the expert walks to empty space and the clip teaches
    the model that empty space is a bed. Only phantoms corrupt supervision.
    """
    pairs = sorted(((math.dist(p['center'], g[1]), i, j)
                    for i, p in enumerate(pred) for j, g in enumerate(gt)
                    if p['cat'] == g[0]),
                   key=lambda t: t[0])
    used_p, used_g, hits = set(), set(), 0
    for d, i, j in pairs:
        if d > gate or i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); hits += 1
    dup = phantom = 0
    for i, p in enumerate(pred):
        if i in used_p:
            continue
        same = [g for g in gt if g[0] == p['cat']]
        d = min((math.dist(p['center'], g[1]) for g in same), default=1e9)
        if d <= DUP:
            dup += 1
        else:
            phantom += 1
    return hits, dup, phantom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--stem', required=True)
    ap.add_argument('--dataset', choices=['mp3d', 'hm3d'], required=True)
    ap.add_argument('--gate', type=float, default=GATE)
    ap.add_argument('--json', default='', help='append a machine-readable row here')
    a = ap.parse_args()

    pred = json.load(open(a.pred))
    gt = truth(a.dataset, a.stem)
    rows = {}
    for c in CATS:
        p = [x for x in pred if x['cat'] == c]
        g = [x for x in gt if x[0] == c]
        if not p and not g:
            continue
        h, dp, ph = match(p, g, a.gate)
        rows[c] = (h, dp, ph, len(p), len(g))
    H = sum(r[0] for r in rows.values())
    D = sum(r[1] for r in rows.values())
    Ph = sum(r[2] for r in rows.values())
    P = sum(r[3] for r in rows.values())
    T = sum(r[4] for r in rows.values())
    print(f'{a.stem}:')
    for c, (h, dp, ph, p, g) in sorted(rows.items()):
        print(f'   {c:11s} pred {p:3d} true {g:3d} hit {h:3d} dup {dp:2d} phantom {ph:2d}   '
              f'P {h/p if p else 0:5.0%}  R {h/g if g else 0:5.0%}')
    print(f'   {"TOTAL":11s} pred {P:3d} true {T:3d} hit {H:3d} dup {D:2d} phantom {Ph:2d}   '
          f'P {H/P if P else 0:5.0%}  R {H/T if T else 0:5.0%}  '
          f'phantom-rate {Ph/P if P else 0:.0%}')
    if a.json:
        with open(a.json, 'a') as f:
            f.write(json.dumps({'stem': a.stem, 'dataset': a.dataset,
                                'per_cat': rows, 'hit': H, 'dup': D, 'phantom': Ph,
                                'pred': P, 'true': T}) + '\n')


if __name__ == '__main__':
    main()
