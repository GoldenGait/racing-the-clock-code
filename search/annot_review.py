"""Build the visual review pages for automatic annotations.

One page per scene: a floor plan with every predicted object placed on it, and
the clearest camera crop behind each one. On an annotated scene the human labels
are overlaid too and each prediction is marked hit, duplicate or phantom; on an
unlabelled scene the crop is the only evidence there is, which is the whole
reason the detector saves one.

  python annot_review.py --annot $RTC_DATA/annot --out $RTC_DATA/docs/annot
"""
from paths import DATA_ROOT
import argparse
import collections
import glob
import html
import json
import math
import os
import shutil

DG = os.path.dirname(os.path.abspath(__file__))
CELL = 0.25
CATS = ['chair', 'bed', 'sofa', 'toilet', 'tv_monitor', 'plant']
COL = {'chair': '#5aafff', 'bed': '#ff7ac6', 'sofa': '#ffd000',
       'toilet': '#7fdca4', 'tv_monitor': '#c99bff', 'plant': '#ff8a3d'}
GATE, DUP = 1.5, 3.0

CSS = """
:root{--ink:#e8e6e1;--dim:#9a978f;--bg:#131417;--panel:#1b1d21;--line:#2b2e34;
      --hit:#7fdca4;--dup:#ffd000;--ph:#ff5c5c;--miss:#6b7280}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);padding:30px 20px 70px;
     font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:26px}
a{color:#7cc0ff}
h1{font-size:22px;margin:0;font-weight:600;text-wrap:balance}
h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--dim);
   margin:0 0 14px;font-weight:600}
.sub{color:var(--dim);font-size:14px;margin-top:4px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:20px 22px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}
th,td{text-align:right;padding:7px 11px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:600;font-size:12px;letter-spacing:.04em;text-transform:uppercase}
.scroll{overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--dim);margin-top:12px}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:-1px;margin-right:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:14px}
figure{margin:0;background:#000;border:1px solid var(--line);border-radius:6px;overflow:hidden}
figure img{width:100%;height:130px;object-fit:cover;display:block}
figcaption{padding:7px 9px;font-size:12px;color:var(--dim);line-height:1.45}
figcaption b{color:var(--ink)}
.tag{float:right;font-weight:600;font-size:11px;letter-spacing:.03em;text-transform:uppercase}
svg{width:100%;height:auto;background:#0e0f12;border-radius:6px;border:1px solid var(--line)}
"""


def truth(dataset, stem, G):
    return [(o[0], (o[1], o[2], o[3])) for r in G['rooms'].values()
            for o in r.get('objects_full', []) if o[0] in CATS]


def classify(pred, gt):
    """label every prediction hit / duplicate / phantom, and count the misses."""
    pairs = sorted(((math.dist(p['center'], g[1]), i, j)
                    for i, p in enumerate(pred) for j, g in enumerate(gt)
                    if p['cat'] == g[0]), key=lambda t: t[0])
    up, ug = set(), set()
    for d, i, j in pairs:
        if d > GATE or i in up or j in ug:
            continue
        up.add(i); ug.add(j)
    lab = []
    for i, p in enumerate(pred):
        if i in up:
            lab.append('hit'); continue
        same = [g for g in gt if g[0] == p['cat']]
        d = min((math.dist(p['center'], g[1]) for g in same), default=1e9)
        lab.append('duplicate' if d <= DUP else 'phantom')
    missed = [g for j, g in enumerate(gt) if j not in ug]
    return lab, missed


def floorplan(G, pred, lab, gt, missed):
    """inline SVG: nav cells as the plan, predictions and truth on top."""
    cells = []
    for r in G['rooms'].values():
        L = str(r['level'])
        gm = G.get('grids', {}).get(L)
        if not gm:
            continue
        for (i, j) in r.get('cells', []):
            cells.append((gm['x0'] + i * CELL, gm['z0'] + j * CELL))
    xs = [c[0] for c in cells] + [p['center'][0] for p in pred]
    zs = [c[1] for c in cells] + [p['center'][2] for p in pred]
    if not xs:
        return '<p class="sub">no plan geometry</p>'
    x0, x1, z0, z1 = min(xs) - 1, max(xs) + 1, min(zs) - 1, max(zs) + 1
    S = 900.0 / max(x1 - x0, 1e-6)
    W, H = 900, max(60, int((z1 - z0) * S))

    def P(x, z):
        return (x - x0) * S, H - (z - z0) * S

    o = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    d = max(1.5, CELL * S)
    for cx, cz in cells:
        px, py = P(cx, cz)
        o.append(f'<rect x="{px:.1f}" y="{py - d:.1f}" width="{d:.1f}" height="{d:.1f}" fill="#2a2e36"/>')
    for c, (gx, gy, gz) in missed:                       # truth we never found
        px, py = P(gx, gz)
        o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="none" '
                 f'stroke="#6b7280" stroke-width="2" stroke-dasharray="3 3"/>')
    for c, (gx, gy, gz) in gt:                           # every human label, faint
        px, py = P(gx, gz)
        o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{COL.get(c, "#888")}" opacity=".45"/>')
    ring = {'hit': '#7fdca4', 'duplicate': '#ffd000', 'phantom': '#ff5c5c', '': '#7cc0ff'}
    for idx, (p, l) in enumerate(zip(pred, lab), 1):
        px, py = P(p['center'][0], p['center'][2])
        o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.5" fill="{COL.get(p["cat"], "#888")}" '
                 f'fill-opacity=".8" stroke="{ring[l]}" stroke-width="2.5"/>')
        o.append(f'<text x="{px + 9:.1f}" y="{py - 7:.1f}" font-size="13" font-weight="700" '
                 f'fill="#e8e6e1" stroke="#0e0f12" stroke-width="3" paint-order="stroke">{idx}</text>')
    o.append('</svg>')
    return ''.join(o)


def page(stem, dataset, pred, G, crops_rel, annotated):
    gt = truth(dataset, stem, G) if annotated else []
    lab, missed = classify(pred, gt) if annotated else ([''] * len(pred), [])
    n = collections.Counter(lab)
    head = (f'{len(pred)} predicted &middot; {len(gt)} human-labelled &middot; '
            f'{n["hit"]} hit, {n["duplicate"]} duplicate, {n["phantom"]} phantom, '
            f'{len(missed)} missed' if annotated
            else f'{len(pred)} predicted &middot; no human labels &mdash; judge from the crops')
    tag = {'hit': 'var(--hit)', 'duplicate': 'var(--dup)', 'phantom': 'var(--ph)', '': 'var(--dim)'}
    figs = []
    for idx, (p, l) in enumerate(zip(pred, lab), 1):
        c = p.get('crop')
        img = (f'<img loading="lazy" src="{crops_rel}/{html.escape(c)}" alt="">' if c
               else '<div style="height:130px"></div>')
        figs.append(
            f'<figure>{img}<figcaption><b>#{idx} {p["cat"]}</b>'
            f'<span class="tag" style="color:{tag[l]}">{l or "&nbsp;"}</span><br>'
            f'{p["n_views"]} views &middot; score {p["score"]:.2f}</figcaption></figure>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(stem)} — auto-annotation review</title><style>{CSS}</style></head><body>
<div class="wrap">
<header><h1>{html.escape(stem)}</h1><div class="sub">{head} &middot; <a href="index.html">all scenes</a></div></header>
<div class="panel"><h2>Floor plan</h2>{floorplan(G, pred, lab, gt, missed)}
<div class="legend">
<span><span class="sw" style="background:#2a2e36"></span>navigable floor</span>
<span><span class="sw" style="border:2px solid var(--hit)"></span>hit</span>
<span><span class="sw" style="border:2px solid var(--dup)"></span>duplicate</span>
<span><span class="sw" style="border:2px solid var(--ph)"></span>phantom</span>
<span><span class="sw" style="border:2px dashed var(--miss)"></span>missed by us</span>
<span>fill colour is the category; small faint dots are the human labels</span>
</div></div>
<div class="panel"><h2>Every detection, clearest view</h2><div class="grid">{''.join(figs)}</div></div>
</div></body></html>"""



def gallery(rows, out, crops_rel='crops'):
    """Every unmatched detection in one grid, worst-looking first.

    The bucket is the distance to the nearest real object of the same class, and
    it is the whole story: a prediction 2m from a real sofa is that sofa,
    slightly mislocated. One with no object of its class anywhere in the scene is
    either invention or something the human annotation never labelled -- and the
    crop is the only way to tell those apart.
    """
    order = ['none', '>10m', '5-10m', '3-5m', '1.5-3m']
    blurb = {
        'none': 'No object of this class anywhere in the scene. Genuine invention, '
                'or a category HM3D never labelled here.',
        '>10m': 'Nothing of this class within 10m. Most likely invention or a '
                'category confusion.',
        '5-10m': 'Same class exists in the scene but far away.',
        '3-5m': 'Just outside the 3m duplicate gate. Often the same object, '
                'mislocated, or its neighbour in the same seating group.',
        '1.5-3m': 'Counted as duplicates: near a real object of the same class, so '
                  'the agent still ends up beside something real.'}
    by = collections.defaultdict(list)
    for r in rows:
        by[r['bucket']].append(r)
    secs = []
    for b in order:
        items = sorted(by.get(b, []), key=lambda r: -r['n_views'])
        if not items:
            continue
        figs = []
        for r in items:
            img = (f'<img loading="lazy" src="{crops_rel}/{html.escape(r["crop"])}" alt="">'
                   if r.get('crop') else '<div style="height:130px"></div>')
            dtxt = 'none in scene' if r['dist'] is None else f'{r["dist"]:.1f} m away'
            figs.append(
                f'<figure>{img}<figcaption><b>#{r["idx"]} {r["cat"]}</b>'
                f'<span class="tag" style="color:var(--dim)">{r["n_views"]}v</span><br>'
                f'<a href="{r["stem"]}.html">{html.escape(r["stem"])}</a><br>'
                f'score {r["score"]:.2f} &middot; {dtxt}</figcaption></figure>')
        secs.append(f'<div class="panel"><h2>{b} &mdash; {len(items)} detections</h2>'
                    f'<p class="sub" style="margin:-6px 0 14px">{blurb[b]}</p>'
                    f'<div class="grid">{"".join(figs)}</div></div>')
    open(out, 'w').write(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Every unmatched detection</title><style>{CSS}</style></head><body><div class="wrap">
<header><h1>Every unmatched detection</h1><div class="sub">{len(rows)} predictions that did not
match a human label, grouped by how far the nearest real object of that class actually is, and
sorted by how many viewpoints saw it. The useful question on each one is not &ldquo;is this
wrong&rdquo; but &ldquo;is this a real object HM3D never labelled&rdquo;.
&middot; <a href="index.html">all scenes</a></div></header>
{''.join(secs)}
</div></body></html>""")



def page_croponly(stem, pred, crops_rel):
    """No room graph yet (auto-annotated scene): the crops ARE the evidence."""
    figs = []
    for idx, p in enumerate(pred, 1):
        c = p.get('crop')
        img = (f'<img loading="lazy" src="{crops_rel}/{html.escape(c)}" alt="">' if c
               else '<div style="height:130px"></div>')
        figs.append(f'<figure>{img}<figcaption><b>#{idx} {p["cat"]}</b><br>'
                    f'{p["n_views"]} views &middot; score {p["score"]:.2f}</figcaption></figure>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(stem)} — auto-annotated (no ground truth)</title><style>{CSS}</style></head><body>
<div class="wrap"><header><h1>{html.escape(stem)}</h1>
<div class="sub">{len(pred)} predicted &middot; auto-annotated scene, no human labels &mdash; judge from
the crops &middot; <a href="index.html">all scenes</a> &middot; <a href="roll_review.html">review queue</a></div></header>
<div class="panel"><div class="grid">{''.join(figs)}</div></div></div></body></html>"""


def roll_review(cands, out):
    """The sampled review protocol: every tv_monitor, everything under 5 views or
    0.85 score, and a deterministic 10% spot-check of the rest."""
    import hashlib
    tv = [c for c in cands if c['cat'] == 'tv_monitor']
    low = [c for c in cands if c['cat'] != 'tv_monitor'
           and (c['n_views'] < 5 or c['score'] < 0.85)]
    rest = [c for c in cands if c not in tv and c not in low]
    spot = [c for c in rest
            if int(hashlib.md5(f"{c['stem']}_{c['idx']}".encode()).hexdigest(), 16) % 10 == 0]
    secs = []
    for title, blurb, items in (
        ('Every tv_monitor', 'the one class whose hallucinations the human review confirmed', tv),
        ('Low support', 'under 5 views or score below 0.85', low),
        ('10% spot-check', 'deterministic sample of the confident remainder', spot)):
        figs = []
        for c in sorted(items, key=lambda x: (x['stem'], x['idx'])):
            img = (f'<img loading="lazy" src="crops/{html.escape(c["crop"])}" alt="">' if c.get('crop')
                   else '<div style="height:130px"></div>')
            figs.append(f'<figure>{img}<figcaption><b>#{c["idx"]} {c["cat"]}</b><br>'
                        f'<a href="{c["stem"]}.html">{html.escape(c["stem"])}</a><br>'
                        f'{c["n_views"]}v &middot; {c["score"]:.2f}</figcaption></figure>')
        secs.append(f'<div class="panel"><h2>{title} &mdash; {len(items)}</h2>'
                    f'<p class="sub" style="margin:-6px 0 14px">{blurb}</p>'
                    f'<div class="grid">{"".join(figs)}</div></div>')
    open(out, 'w').write(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Roll review queue — {len(tv)+len(low)+len(spot)} of {len(cands)} detections</title>
<style>{CSS}</style></head><body><div class="wrap">
<header><h1>592-scene roll: your review queue</h1>
<div class="sub">{len(cands):,} detections total; this queue is the {len(tv)+len(low)+len(spot):,}
that need human eyes. Verdict shorthand as before: &ldquo;scene: bad 3,17&rdquo;.</div></header>
{''.join(secs)}</div></body></html>""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annot', default=f'{DATA_ROOT}/annot')
    ap.add_argument('--out', default=f'{DATA_ROOT}/docs/annot')
    ap.add_argument('--dataset', default='hm3d')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Gall = json.load(open(f'{DG}/room_graphs_hm3d_v2.json'))
    Gall.update(json.load(open(f'{DG}/room_graphs_mp3d.json')))

    src = os.path.join(a.annot, 'crops')
    dst = os.path.join(a.out, 'crops')
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if not os.path.exists(os.path.join(dst, f)):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    rows, unmatched, rollcands = [], [], []
    for f in sorted(glob.glob(os.path.join(a.annot, 'obj', '*.json'))):
        stem = os.path.basename(f)[:-5]
        G = Gall.get(stem)
        pred = json.load(open(f))
        if G is None:
            open(os.path.join(a.out, stem + '.html'), 'w').write(
                page_croponly(stem, pred, 'crops'))
            rows.append((stem, len(pred), 0, 0, 0, 0, 0))
            for idx, pr in enumerate(pred, 1):
                rollcands.append({'stem': stem, 'idx': idx, 'cat': pr['cat'],
                                  'n_views': pr['n_views'], 'score': pr['score'],
                                  'crop': pr.get('crop')})
            continue
        annotated = any(r.get('objects_full') for r in G['rooms'].values())
        gt = truth(a.dataset, stem, G) if annotated else []
        lab, missed = classify(pred, gt) if annotated else ([''] * len(pred), [])
        n = collections.Counter(lab)
        open(os.path.join(a.out, stem + '.html'), 'w').write(
            page(stem, a.dataset, pred, G, 'crops', annotated))
        rows.append((stem, len(pred), len(gt), n['hit'], n['duplicate'], n['phantom'], len(missed)))
        for idx, (p, l) in enumerate(zip(pred, lab), 1):
            if l == 'hit' or not annotated:
                continue
            same = [g for g in gt if g[0] == p['cat']]
            d = min((math.dist(p['center'], g[1]) for g in same), default=None)
            b = ('none' if d is None else '1.5-3m' if d <= DUP else '3-5m' if d <= 5
                 else '5-10m' if d <= 10 else '>10m')
            unmatched.append({'stem': stem, 'idx': idx, 'cat': p['cat'],
                              'n_views': p['n_views'], 'score': p['score'],
                              'dist': d, 'bucket': b, 'crop': p.get('crop')})

    tr = []
    for s, p, g, h, d, ph, m in rows:
        tr.append(f'<tr><td><a href="{s}.html">{html.escape(s)}</a></td><td>{p}</td><td>{g}</td>'
                  f'<td>{h}</td><td>{d}</td><td style="color:var(--ph)">{ph}</td><td>{m}</td>'
                  f'<td>{h/p if p else 0:.0%}</td><td>{h/g if g else 0:.0%}</td>'
                  f'<td>{ph/p if p else 0:.0%}</td></tr>')
    P = sum(r[1] for r in rows); T = sum(r[2] for r in rows)
    H = sum(r[3] for r in rows); D = sum(r[4] for r in rows); PH = sum(r[5] for r in rows)
    tot = (f'<tr style="font-weight:600"><td>{len(rows)} scenes</td><td>{P}</td><td>{T}</td>'
           f'<td>{H}</td><td>{D}</td><td style="color:var(--ph)">{PH}</td>'
           f'<td>{sum(r[6] for r in rows)}</td><td>{H/P if P else 0:.0%}</td>'
           f'<td>{H/T if T else 0:.0%}</td><td>{PH/P if P else 0:.0%}</td></tr>')
    open(os.path.join(a.out, 'index.html'), 'w').write(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Auto-annotation review</title><style>{CSS}</style></head><body><div class="wrap">
<header><h1>Auto-annotation review</h1>
<div class="sub"><a href="unmatched.html"><b>See every unmatched detection &rarr;</b></a><br>
Every scene the annotator has processed. Phantoms are the errors that corrupt
supervision &mdash; a duplicate still puts the agent beside a real object, a phantom sends it to
empty space.</div></header>
<div class="panel"><div class="scroll"><table>
<tr><th>scene</th><th>pred</th><th>true</th><th>hit</th><th>dup</th><th>phantom</th><th>missed</th>
<th>precision</th><th>recall</th><th>phantom rate</th></tr>
{''.join(tr)}{tot}</table></div></div></div></body></html>""")
    gallery(unmatched, os.path.join(a.out, 'unmatched.html'))
    if rollcands:
        roll_review(rollcands, os.path.join(a.out, 'roll_review.html'))
    print(f'wrote {len(rows)} scene pages + index + gallery of {len(unmatched)} '
          f'unmatched detections to {a.out}')


if __name__ == '__main__':
    main()
