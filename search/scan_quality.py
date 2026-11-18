"""Scan-quality gate for the unlabelled-scene expansion.

Renders a small depth sweep of a scene and scores how broken the photogrammetry
is: the fraction of depth pixels that are holes (no return inside the far clip)
plus how much of the navmesh sampled at all. Bad scans seed phantom objects and
break back-projection, so the expansion set drops the worst tail -- gated on
geometry (holes), not cosmetics (texture smear), to avoid selecting for easy
scenes.

  python scan_quality.py --scenes-file F --gpu 1 --out scores.jsonl
"""
import argparse
import json
import math
import os

import numpy as np

from annot_render import make_sim, viewpoints
from c4_collect_expert import HM3D

N_VIEWS = 48
HEADINGS = 4


def score_scene(stem, rel, gpu):
    d = os.path.join(HM3D, rel)
    glb = next((p for p in (os.path.join(d, stem + ".basis.glb"),
                            os.path.join(d, stem + ".glb")) if os.path.exists(p)), None)
    if glb is None:
        return {'stem': stem, 'error': 'no glb'}
    sim = make_sim(glb, gpu)
    pf = sim.pathfinder
    pts = viewpoints(pf, spacing=2.0)
    if not pts:
        sim.close(); return {'stem': stem, 'error': 'no navmesh'}
    step = max(1, len(pts) * HEADINGS // N_VIEWS)
    st = sim.get_agent(0).get_state()
    hole, valid, views = [], 0, 0
    for p in pts[::max(1, step // HEADINGS)]:
        for h in range(HEADINGS):
            th = 2 * math.pi * h / HEADINGS
            st.position = p
            st.rotation = np.quaternion(math.cos(th / 2), 0, math.sin(th / 2), 0)
            sim.get_agent(0).set_state(st)
            dep = np.asarray(sim.get_sensor_observations()["depth"], np.float32)
            if not np.isfinite(dep).any():
                continue
            views += 1
            # holes: zero/invalid depth in the lower 3/4 of the image (sky/ceiling
            # openings in the top band are normal for scans, not defects)
            band = dep[dep.shape[0] // 4:]
            hole.append(float((band <= 0.01).mean()))
            if views >= N_VIEWS:
                break
        if views >= N_VIEWS:
            break
    area = float(pf.navigable_area)
    sim.close()
    if not hole:
        return {'stem': stem, 'error': 'no valid views'}
    return {'stem': stem, 'views': views, 'nav_area_m2': round(area, 1),
            'hole_mean': round(float(np.mean(hole)), 4),
            'hole_p90': round(float(np.percentile(hole, 90)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes-file', required=True, help='lines: <stem> <scene_rel>')
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: done.add(json.loads(l)['stem'])
            except Exception: pass
    with open(a.out, 'a') as f:
        for line in open(a.scenes_file):
            stem, rel = line.split()
            if stem in done:
                continue
            try:
                r = score_scene(stem, rel, a.gpu)
            except Exception as e:
                r = {'stem': stem, 'error': str(e)[:120]}
            f.write(json.dumps(r) + '\n'); f.flush()
            print(stem, r.get('hole_mean', r.get('error')))


if __name__ == '__main__':
    main()
