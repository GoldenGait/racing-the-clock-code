"""P15: re-price room sweeps for the sweep-once policy without rebuilding the graphs.

Stage C1 priced a room sweep as a tour of every coverage viewpoint with a 12-turn
look-around at each (sweep_cost = 12 * n_viewpoints + tour). The P15 expert scans
once, from the best-covering viewpoint, so the price becomes one look-around plus
the walk to that viewpoint. The viewpoints themselves are not stored in the graph
json, so the walk is approximated by one leg of the old tour (tour / (n_vp - 1)),
or half the room's characteristic length when there was only one viewpoint.

    python p15_recost_graphs.py room_graphs_mp3d.json room_graphs_hm3d_v2.json ...
writes <name>_p15.json next to each input.
"""
import json
import math
import sys

TURNS_360, CELL, FORWARD_M = 12, 0.25, 0.25


def once_cost(room):
    n_vp = max(1, int(room.get('n_viewpoints', 1)))
    old = float(room.get('sweep_cost', TURNS_360))
    tour = max(0.0, old - TURNS_360 * n_vp)
    if n_vp > 1:
        walk = tour / (n_vp - 1)
    else:
        walk = 0.5 * math.sqrt(max(0.0, float(room.get('area_m2', 0.0)))) / FORWARD_M
    return int(round(TURNS_360 + walk))


for path in sys.argv[1:]:
    G = json.load(open(path))
    n, before, after = 0, 0.0, 0.0
    for stem, g in G.items():
        rooms = g['rooms'].values() if isinstance(g['rooms'], dict) else g['rooms']
        for r in rooms:
            if 'sweep_cost' not in r:
                continue
            before += r['sweep_cost']
            r['sweep_cost_tour'] = r['sweep_cost']
            r['sweep_cost'] = once_cost(r)
            after += r['sweep_cost']; n += 1
    out = path.replace('.json', '_p15.json')
    json.dump(G, open(out, 'w'))
    print(f"{out}: {n} rooms, mean sweep cost {before / max(1, n):.1f} -> {after / max(1, n):.1f} primitives")
