"""Fuse cached per-view detections into scene-level instances.

Split out of annot_detect so the clustering can be retuned without paying for
the segmenter again: the detector writes raw detections once, this sweeps over
them in seconds. Large furniture over-segments under a small radius (one sofa
seen from four sides lands four centroids apart), so the radius is per category
and there is a second merge pass for clusters that end up adjacent.

  python annot_fuse.py --raw raw.json --out objects.json
  python annot_fuse.py --raw raw.json --sweep --stem X --dataset hm3d
"""
import argparse
import json
import subprocess
import sys

import numpy as np

FUSE_R = {'chair': 0.6, 'sofa': 1.6, 'plant': 0.5, 'bed': 1.8,
          'toilet': 0.4, 'tv_monitor': 0.5}
# [2026-08-21, from the human review] per-class acceptance floors. tv_monitor
# produced the only repeated hallucinations (windows/pictures), so it needs a
# higher bar; plants looked noisy by the metric but were nearly all real.
CLASS_MIN = {'tv_monitor': {'score': 0.90, 'views': 5}}


def fuse(dets, fuse_r=None, min_views=3, y_gate=1.0):
    fuse_r = fuse_r or FUSE_R
    inst = []
    for cat in sorted({d['cat'] for d in dets}):
        R = fuse_r[cat]
        cl = []
        for d in sorted([x for x in dets if x['cat'] == cat], key=lambda x: -x['score']):
            c = np.asarray(d['center'], np.float64)
            for k in cl:
                if (np.hypot(*(k['c'][[0, 2]] - c[[0, 2]])) < R
                        and abs(k['c'][1] - c[1]) < y_gate):
                    k['pts'].append(c); k['r'].append(d['r']); k['s'].append(d['score'])
                    k['c'] = np.mean(k['pts'], 0)
                    break
            else:
                cl.append({'c': c, 'pts': [c], 'r': [d['r']], 's': [d['score']]})
        # second pass: clusters that drifted together are one object
        merged = True
        while merged:
            merged = False
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    if (np.hypot(*(cl[i]['c'][[0, 2]] - cl[j]['c'][[0, 2]])) < R
                            and abs(cl[i]['c'][1] - cl[j]['c'][1]) < y_gate):
                        cl[i]['pts'] += cl[j]['pts']; cl[i]['r'] += cl[j]['r']
                        cl[i]['s'] += cl[j]['s']
                        cl[i]['c'] = np.mean(cl[i]['pts'], 0)
                        cl.pop(j); merged = True
                        break
                if merged:
                    break
        for k in cl:
            need = CLASS_MIN.get(cat, {})
            if len(k['pts']) < max(min_views, need.get('views', 0)):
                continue
            if float(np.mean(k['s'])) < need.get('score', 0.0):
                continue
            inst.append({'cat': cat, 'center': [float(x) for x in k['c']],
                         'r': float(np.median(k['r'])), 'n_views': len(k['pts']),
                         'score': float(np.mean(k['s']))})
    return inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--min-views', type=int, default=3)
    ap.add_argument('--sweep', action='store_true',
                    help='grid over radius scale and min-views, scoring each against truth')
    ap.add_argument('--stem', default=''); ap.add_argument('--dataset', default='hm3d')
    a = ap.parse_args()
    dets = json.load(open(a.raw))

    if not a.sweep:
        inst = fuse(dets, min_views=a.min_views)
        json.dump(inst, open(a.out, 'w'), indent=1)
        print(f'{len(dets)} raw -> {len(inst)} instances')
        return

    import tempfile
    for scale in (1.0, 1.3, 1.6, 2.0):
        for mv in (2, 3, 5, 8):
            r = {k: v * scale for k, v in FUSE_R.items()}
            inst = fuse(dets, r, mv)
            with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
                json.dump(inst, f); p = f.name
            out = subprocess.run([sys.executable, __file__.replace('annot_fuse', 'annot_eval'),
                                  '--pred', p, '--stem', a.stem, '--dataset', a.dataset],
                                 capture_output=True, text=True).stdout
            tot = [l for l in out.splitlines() if 'TOTAL' in l]
            print(f'scale {scale:.1f} min_views {mv}: {tot[0].strip() if tot else "?"}')


if __name__ == '__main__':
    main()
