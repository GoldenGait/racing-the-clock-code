"""Tune fusion against phantom rate across every annotated scene at once.

Precision is the wrong objective. A duplicate still leaves the agent standing
next to a real object, so what the model sees is correct and only the reachable
count drifts. A phantom puts a target where nothing of that class exists, so the
expert walks to empty space and the clip teaches the model that empty space is a
bed. Those are the errors to buy down, and buying them down with a blunt merge
radius costs recall we do not need to spend.

  python annot_sweep.py --annot $RTC_DATA/annot
"""
from paths import DATA_ROOT
import argparse
import glob
import json
import os

from annot_eval import truth, DUP, GATE
from annot_fuse import fuse, FUSE_R
import math

DG = os.path.dirname(os.path.abspath(__file__))


def score(pred, gt):
    pairs = sorted(((math.dist(p['center'], g[1]), i, j)
                    for i, p in enumerate(pred) for j, g in enumerate(gt)
                    if p['cat'] == g[0]), key=lambda t: t[0])
    up, ug = set(), set()
    for d, i, j in pairs:
        if d > GATE or i in up or j in ug:
            continue
        up.add(i); ug.add(j)
    dup = ph = 0
    for i, p in enumerate(pred):
        if i in up:
            continue
        same = [g for g in gt if g[0] == p['cat']]
        d = min((math.dist(p['center'], g[1]) for g in same), default=1e9)
        if d <= DUP:
            dup += 1
        else:
            ph += 1
    return len(up), dup, ph, len(pred), len(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annot', default=f'{DATA_ROOT}/annot')
    a = ap.parse_args()
    G = json.load(open(f'{DG}/room_graphs_hm3d_v2.json'))
    G.update(json.load(open(f'{DG}/room_graphs_mp3d.json')))

    scenes = []
    for f in sorted(glob.glob(os.path.join(a.annot, 'raw', '*.json'))):
        stem = os.path.basename(f)[:-5]
        if stem not in G:
            continue
        gt = [(o[0], (o[1], o[2], o[3])) for r in G[stem]['rooms'].values()
              for o in r.get('objects_full', []) if o[0] in FUSE_R]
        if gt:
            scenes.append((stem, json.load(open(f)), gt))
    print(f'sweeping on {len(scenes)} annotated scenes\n')
    print(f"{'score':>5} {'scale':>5} {'views':>5}  {'prec':>5} {'rec':>5} {'phantom':>7}  {'pred':>5}")
    best = None
    for smin in (0.75, 0.85, 0.90, 0.95):
        for scale in (1.0, 1.3, 1.6):
            for mv in (3, 5, 8, 12):
                R = {k: v * scale for k, v in FUSE_R.items()}
                H = D = P = N = T = 0
                for stem, raw, gt in scenes:
                    d = [x for x in raw if x['score'] >= smin]
                    inst = fuse(d, R, mv)
                    h, dp, ph, n, t = score(inst, gt)
                    H += h; D += dp; P += ph; N += n; T += t
                prec, rec, phr = H / max(N, 1), H / max(T, 1), P / max(N, 1)
                print(f'{smin:5.2f} {scale:5.1f} {mv:5d}  {prec:5.0%} {rec:5.0%} {phr:7.0%}  {N:5d}')
                # want the fewest phantoms we can get without giving up recall
                if rec >= 0.70 and (best is None or phr < best[0]):
                    best = (phr, rec, prec, smin, scale, mv)
    if best:
        phr, rec, prec, smin, scale, mv = best
        print(f'\nbest with recall >= 70%: score>={smin} scale={scale} min_views={mv}'
              f'  ->  phantom {phr:.0%}, recall {rec:.0%}, precision {prec:.0%}')


if __name__ == '__main__':
    main()
