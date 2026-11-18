"""Stage 2 of automatic scene annotation: lift 2D instance masks into 3D objects.

Reads a render dump from annot_render.py, segments every view with a COCO
instance model, back-projects each mask through its own depth pixels, and fuses
the per-view detections across views into scene-level instances.

All six of our target categories are COCO classes, so a closed-vocabulary model
beats an open-vocabulary one here, and masks localise far better than boxes: a
box back-projects to a frustum, a mask back-projects to the object's own points.

  python annot_detect.py --npz /tmp/x.npz --out objects.json
"""
import argparse
import json
import os

import imageio.v3 as iio
import numpy as np
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from annot_fuse import CLASS_MIN

MODEL = 'facebook/mask2former-swin-large-coco-instance'
# COCO label -> our category. These names are exactly what this checkpoint emits.
COCO2CAT = {'chair': 'chair', 'sofa': 'sofa', 'pottedplant': 'plant',
            'bed': 'bed', 'toilet': 'toilet', 'tvmonitor': 'tv_monitor'}
# fusion radius per category, roughly half a typical instance's footprint
FUSE_R = {'chair': 0.6, 'sofa': 1.1, 'plant': 0.5, 'bed': 1.2,
          'toilet': 0.4, 'tv_monitor': 0.5}
D_MIN, D_MAX = 0.35, 6.0


def backproject(mask, depth, K, pos, yaw, cam_h):
    """mask pixels + depth -> world xyz, trimmed of depth outliers."""
    fx, fy, cx, cy = K
    v, u = np.nonzero(mask)
    d = depth[v, u].astype(np.float32)
    ok = (d > D_MIN) & (d < D_MAX)
    if ok.sum() < 40:
        return None
    u, v, d = u[ok], v[ok], d[ok]
    lo, hi = np.percentile(d, [15, 85])       # drop background bleeding past edges
    keep = (d >= lo) & (d <= hi)
    if keep.sum() < 25:
        return None
    u, v, d = u[keep], v[keep], d[keep]
    xc = (u - cx) * d / fx
    yc = -(v - cy) * d / fy
    zc = -d
    c, s = np.cos(yaw), np.sin(yaw)
    xw = xc * c + zc * s + pos[0]
    yw = yc + pos[1] + cam_h
    zw = -xc * s + zc * c + pos[2]
    return np.stack([xw, yw, zw], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--score', type=float, default=0.75)
    ap.add_argument('--min-views', type=int, default=3,
                    help='an instance must be seen from this many viewpoints')
    ap.add_argument('--crop-dir', default='',
                    help='save the clearest view of each instance here; on an '
                         'unlabelled scene the crop is the only way to review it')
    ap.add_argument('--raw-out', default='',
                    help='dump per-view detections here so fusion can be retuned '
                         'without paying for the detector again')
    a = ap.parse_args()

    z = np.load(a.npz)
    K = (float(z['fx']), float(z['fy']), float(z['cx']), float(z['cy']))
    cam_h = float(z['cam_h'])
    rgb, depth, pos, yaw = z['rgb'], z['depth'], z['pos'], z['yaw']

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL).to(dev).eval()
    id2l = model.config.id2label

    dets = []                       # (cat, centroid(3), radius, score, view, bbox)
    for i0 in range(0, len(rgb), a.batch):
        batch = [rgb[i] for i in range(i0, min(i0 + a.batch, len(rgb)))]
        inp = proc(images=batch, return_tensors='pt').to(dev)
        with torch.inference_mode():
            out = model(**inp)
        res = proc.post_process_instance_segmentation(
            out, target_sizes=[b.shape[:2] for b in batch], threshold=a.score)
        for j, r in enumerate(res):
            i = i0 + j
            seg = r['segmentation']
            if seg is None:
                continue
            seg = seg.cpu().numpy()
            dep = depth[i].astype(np.float32)
            for s in r['segments_info']:
                cat = COCO2CAT.get(id2l[s['label_id']])
                if cat is None or s['score'] < a.score:
                    continue
                m = seg == s['id']
                pts = backproject(m, dep, K, pos[i], float(yaw[i]), cam_h)
                if pts is None:
                    continue
                c = np.median(pts, 0)
                rad = float(np.percentile(np.linalg.norm(pts[:, [0, 2]] - c[[0, 2]], axis=1), 85))
                ys, xs = np.nonzero(m)
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                dets.append((cat, c, min(rad, 2.0), float(s['score']), i, bbox))

    if a.raw_out:
        json.dump([{'cat': c, 'center': [float(x) for x in ct], 'r': float(rr),
                    'score': float(sc), 'view': int(v), 'bbox': list(bb)}
                   for c, ct, rr, sc, v, bb in dets], open(a.raw_out, 'w'))

    # fuse: greedy per category, a detection joins the first cluster within its radius
    inst = []
    for cat in set(d[0] for d in dets):
        cl = []
        for _, c, rad, sc, v, bb in sorted([d for d in dets if d[0] == cat], key=lambda d: -d[3]):
            for k in cl:
                if (np.linalg.norm(k['c'][[0, 2]] - c[[0, 2]]) < FUSE_R[cat]
                        and abs(k['c'][1] - c[1]) < 1.0):
                    k['pts'].append(c); k['r'].append(rad); k['s'].append(sc)
                    k['c'] = np.mean(k['pts'], 0)
                    break
            else:
                # first member is the highest-scoring one, so it is the clearest view
                cl.append({'c': c, 'pts': [c], 'r': [rad], 's': [sc], 'view': v, 'bbox': bb})
        for j, k in enumerate(cl):
            need = CLASS_MIN.get(cat, {})
            if len(k['pts']) < max(a.min_views, need.get('views', 0)):
                continue
            if float(np.mean(k['s'])) < need.get('score', 0.0):
                continue
            o = {'cat': cat, 'center': [float(x) for x in k['c']],
                 'r': float(np.median(k['r'])), 'n_views': len(k['pts']),
                 'score': float(np.mean(k['s']))}
            if a.crop_dir:
                os.makedirs(a.crop_dir, exist_ok=True)
                x0, y0, x1, y1 = k['bbox']
                pad = 16
                im = rgb[k['view']][max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
                name = f"{str(z['stem'])}_{cat}_{j}.jpg"
                iio.imwrite(os.path.join(a.crop_dir, name), im, quality=88)
                o['crop'] = name
            inst.append(o)

    json.dump(inst, open(a.out, 'w'), indent=1)
    n = {}
    for o in inst:
        n[o['cat']] = n.get(o['cat'], 0) + 1
    print(f"DETECTED {len(rgb)} views -> {len(dets)} raw masks -> {len(inst)} instances {n}")


if __name__ == '__main__':
    main()
