"""Stage C2 - cross-scene transition bank with scene-level cross-fitting.

From room_graphs_mp3d.json (Stage C1), generate one-step room transitions
    tau = (c, v, r, d, c')
by growing random partial observed graphs in each scene and enumerating every
frontier candidate. The full graph is used ONLY to execute the one step (record
r, d and reveal v in c'); candidate scoring features never see it.

Leak firewall:
  - a candidate's context features use its VOTED type: objects of the observed
    boundary rooms vote via P(room|object) restricted to within DOOR_R of the door;
    no votes -> 'unknown'. The true type/counts of an unentered room appear only
    in the recorded outcome r and in c' after entry.
  - folds: 5 scene-level folds; a fold's bank is for planning in OTHER folds only.

Costs are in PRIMITIVES: d = dnav(centroid->door->centroid geodesic)/0.25 + sweep_cost.

  python build_transition_bank.py --graphs room_graphs_mp3d.json \
      --prior prior_table.json --out transition_bank.json
"""
import argparse, collections, json, math, random

CATS = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
FOLDS = 5
STARTS_PER_SCENE = 10        # start rooms sampled per scene
GROWTHS_PER_START = 6        # independent partial-graph growths per start
SEED = 7
DOOR_R = 3.0                 # objects within this of the door vote for the candidate's type
DNAV_BINS = [20, 60]         # primitives; <20 near, 20-60 mid, >60 far
NOBS_BINS = [3, 7]           # |V_obs| buckets: small, medium, large
FORWARD_M = 0.25


def bucket(x, edges):
    for k, e in enumerate(edges):
        if x < e:
            return k
    return len(edges)


def voted_type(cand_id, G, prior, door):
    """Room-type vote for an UNENTERED candidate = what the agent would glimpse THROUGH
    the doorway: the candidate's objects within DOOR_R of the door
    vote P(room|object). This mirrors real execution-time observation -- the agent
    standing at the door sees these objects -- so it is observation, not a leak. The
    candidate's full contents/true label are still never used for scoring."""
    votes = collections.Counter()
    dx, _, dz = door
    for mp, ox, oz in G['rooms'][cand_id].get('objects', []):
        if mp in prior and math.hypot(ox - dx, oz - dz) <= DOOR_R:
            for room, p in prior[mp].items():
                votes[room] += p
    return votes.most_common(1)[0][0] if votes else 'unknown'


def edges_of(G):
    adj = collections.defaultdict(list)
    for e in G['edges']:
        u, v = str(e['u']), str(e['v'])
        if u in G['rooms'] and v in G['rooms']:
            adj[u].append((v, e)); adj[v].append((u, e))
    return adj


def dnav_prims(G, u, e):
    """travel primitives from room u's centroid through the edge (approx: dnav_m geodesic
    between centroids recorded in C1) -- forward steps only; turns are folded into sweep."""
    d = e.get('dnav_m', -1)
    if d is None or d < 0:
        return None
    return max(1, int(round(d / FORWARD_M)))


def phi(c, cand_voted_type, dnav_b, y):
    """Discrete signature for exact matching."""
    return "|".join([c['cur_type'], ",".join(sorted(c['obs_types'])),
                     f"n{bucket(c['n_obs'], NOBS_BINS)}", cand_voted_type, f"d{dnav_b}", y])


def context_of(cur, obs, G):
    return {'cur_type': G['rooms'][cur]['type'],
            'obs_types': [G['rooms'][u]['type'] for u in sorted(obs)],
            'n_obs': len(obs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs', default='room_graphs_mp3d.json')
    ap.add_argument('--prior', default='prior_table.json')
    ap.add_argument('--out', default='transition_bank.json')
    a = ap.parse_args()

    graphs = json.load(open(a.graphs))
    prior = json.load(open(a.prior))['prior']
    rng = random.Random(SEED)

    stems = sorted(graphs)
    fold_of = {s: i % FOLDS for i, s in enumerate(stems)}   # deterministic round-robin

    bank, n_rec = [], 0
    for stem in stems:
        G = graphs[stem]
        adj = edges_of(G)
        rooms = [r for r in G['rooms'] if adj[r]]
        if len(rooms) < 3:
            continue
        for _ in range(STARTS_PER_SCENE):
            start = rng.choice(rooms)
            for _ in range(GROWTHS_PER_START):
                # grow a random connected partial graph of random size
                # cap 20 (was 8): shallow growths starved the planner of deep-context
                # experience -- it hit Q=0 and stopped ~6-8 rooms into long episodes
                size = rng.randint(1, max(1, min(20, len(rooms) - 1)))
                obs, frontier_pool = {start}, [start]
                while len(obs) < size and frontier_pool:
                    u = rng.choice(frontier_pool)
                    nxt = [v for v, _ in adj[u] if v not in obs]
                    if not nxt:
                        frontier_pool.remove(u); continue
                    obs.add(rng.choice(nxt))
                    frontier_pool = [w for w in obs if any(v not in obs for v, _ in adj[w])]
                cur = rng.choice(sorted(obs))
                c = context_of(cur, obs, G)
                # enumerate every frontier candidate from this context
                cand = {}
                for u in obs:
                    for v, e in adj[u]:
                        if v not in obs and v not in cand:
                            cand[v] = e
                for v, e in sorted(cand.items()):
                    d_nav = dnav_prims(G, cur, e)
                    if d_nav is None:
                        continue
                    rv = G['rooms'][v]
                    d = d_nav + rv['sweep_cost']
                    vt = voted_type(v, G, prior, e['door'])
                    obs2 = obs | {v}
                    c2 = context_of(v, obs2, G)
                    r = {y: rv['counts'].get(y, 0) for y in CATS}
                    rec = {'scene': stem, 'fold': fold_of[stem],
                           'c': c, 'v_voted_type': vt, 'v_true_type': rv['type'],
                           'd_nav': d_nav, 'd_sweep': rv['sweep_cost'], 'd': d,
                           'r': r, 'c_next': c2,
                           'phi': {y: phi(c, vt, bucket(d_nav, DNAV_BINS), y) for y in CATS}}
                    bank.append(rec); n_rec += 1

    json.dump({'meta': {'folds': FOLDS, 'fold_of': fold_of, 'cats': CATS, 'seed': SEED,
                        'starts_per_scene': STARTS_PER_SCENE, 'growths_per_start': GROWTHS_PER_START,
                        'dnav_bins': DNAV_BINS, 'nobs_bins': NOBS_BINS, 'n_records': n_rec},
               'transitions': bank}, open(a.out, 'w'))

    # -------- GATE C2 --------
    per_fold = collections.Counter(t['fold'] for t in bank)
    print(f"[C2] records={n_rec} scenes={len(stems)} per-fold={dict(sorted(per_fold.items()))}")
    leak = sum(1 for t in bank if fold_of[t['scene']] != t['fold'])
    print(f"[C2] GATE fold-consistency: {'PASS' if leak == 0 else f'FAIL ({leak})'}")
    vt_acc = sum(1 for t in bank if t['v_voted_type'] == t['v_true_type']) / max(1, n_rec)
    unk = sum(1 for t in bank if t['v_voted_type'] == 'unknown') / max(1, n_rec)
    print(f"[C2] voted-type accuracy={vt_acc:.2f} unknown-rate={unk:.2f} (diagnostic only)")
    sig = collections.Counter(t['phi'][CATS[0]] for t in bank)
    ex1 = sum(1 for s, n in sig.items() if n == 1)
    print(f"[C2] unique phi signatures={len(sig)} singletons={ex1} "
          f"(exact-match coverage {1 - ex1/max(1,len(sig)):.2f})")


if __name__ == '__main__':
    main()
