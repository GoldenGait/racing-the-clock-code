"""Stage C3 - empirical Bellman budget planner over the cross-scene transition bank.

Implements the base spec S5/S6/S9: for a query (context c, candidate v, budget B),
    Qhat_y(c,B,v) = sum_i w_i [ r_i^y + Vhat_y(key(c'_i), B - d_i) ]      (d_i <= B)
    Vhat_y(c,B)   = max(0, max_v Qhat_y(c,B,v)),   Vhat_y(c,0) = 0
with exact phi-signature matching first, softmax-weighted K-NN fallback, and
memoization on (abstract context key, B). Costs are positive integers (primitives),
so recursion strictly decreases B and terminates.

Partial-observability firewall: the continuation value uses only the NEXT OBSERVABLE
context key(c'_i) recorded in the bank -- never any scene's hidden graph.
Cross-fitting: a Planner built for plan_fold k retrieves only transitions with fold != k.

GATE C3 (run as __main__): exact agreement with brute-force enumeration on synthetic
graphs of up to 8 rooms (unique types -> abstract context = true state -> the empirical
model IS the true deterministic MDP, so Vhat must equal the brute-force optimum).

  python budget_planner.py            # runs the synthetic-graph gate
"""
import argparse, collections, itertools, json, math, random

NOBS_BINS = [3, 7]
DNAV_BINS = [20, 60]
KNN_K = 12
TAU = 2.0


def bucket(x, edges):
    for k, e in enumerate(edges):
        if x < e:
            return k
    return len(edges)


def ckey(c):
    """Abstract context key: what the agent can observe, nothing else."""
    return c['cur_type'] + '|' + ','.join(sorted(c['obs_types'])) + f"|n{bucket(c['n_obs'], NOBS_BINS)}"


def akey(voted_type, dnav_prims):
    return voted_type + f"|d{bucket(dnav_prims, DNAV_BINS)}"


def feat(c, voted_type, dnav_prims):
    """Sparse feature dict for KNN distance (one-hot + bags, no learned encoder)."""
    f = collections.Counter()
    f['cur=' + c['cur_type']] = 1
    for t in c['obs_types']:
        f['obs=' + t] += 1
    f['nobs'] = bucket(c['n_obs'], NOBS_BINS)
    f['cand=' + voted_type] = 1
    f['dbin'] = bucket(dnav_prims, DNAV_BINS)
    return f


def dist2(f1, f2):
    keys = set(f1) | set(f2)
    return sum((f1.get(k, 0) - f2.get(k, 0)) ** 2 for k in keys)


class Planner:
    def __init__(self, transitions, plan_fold, y, knn_k=KNN_K, tau=TAU):
        self.y, self.knn_k, self.tau = y, knn_k, tau
        self.bank = [t for t in transitions if plan_fold is None or t['fold'] != plan_fold]
        # exact index: (context key, action key) -> transition idxs
        self.exact = collections.defaultdict(list)
        self.by_ckey = collections.defaultdict(list)   # context key -> idxs (action menu)
        self.feats = []
        for i, t in enumerate(self.bank):
            ck, ak = ckey(t['c']), akey(t['v_voted_type'], t['d_nav'])
            self.exact[(ck, ak)].append(i)
            self.by_ckey[ck].append(i)
            self.feats.append(feat(t['c'], t['v_voted_type'], t['d_nav']))
        self.memo = {}

    def _retrieve(self, c, voted_type, dnav_prims):
        """(idxs, weights): exact matches if any, else softmax-weighted KNN."""
        ids = self.exact.get((ckey(c), akey(voted_type, dnav_prims)), [])
        if ids:
            w = 1.0 / len(ids)
            return ids, [w] * len(ids)
        q = feat(c, voted_type, dnav_prims)
        scored = sorted(((dist2(q, f), i) for i, f in enumerate(self.feats)))[:self.knn_k]
        if not scored:
            return [], []
        ws = [math.exp(-d / self.tau) for d, _ in scored]
        s = sum(ws)
        return [i for _, i in scored], [w / s for w in ws]

    def q(self, c, voted_type, dnav_prims, B):
        """Empirical Bellman action value for one candidate under budget B."""
        ids, ws = self._retrieve(c, voted_type, dnav_prims)
        pairs = [(i, w) for i, w in zip(ids, ws) if self.bank[i]['d'] <= B]
        if not pairs:
            return None                      # infeasible under B for all evidence
        s = sum(w for _, w in pairs)
        val = 0.0
        for i, w in pairs:
            t = self.bank[i]
            val += (w / s) * (t['r'].get(self.y, 0) + self.v(ckey(t['c_next']), B - t['d']))
        return val

    def v(self, ck, B):
        """Value of an abstract context: action menu = actions seen from this context key
        in the bank (on-demand, memoized). 0 if B exhausted or context unseen."""
        if B <= 0:
            return 0.0
        m = self.memo.get((ck, B))
        if m is not None:
            return m
        self.memo[(ck, B)] = 0.0             # guard vs. cycles among abstract keys
        best = 0.0
        menu = {}
        for i in self.by_ckey.get(ck, []):
            t = self.bank[i]
            menu[akey(t['v_voted_type'], t['d_nav'])] = (t['c'], t['v_voted_type'], t['d_nav'])
        for c, vt, dn in menu.values():
            qv = self.q(c, vt, dn, B)
            if qv is not None and qv > best:
                best = qv
        self.memo[(ck, B)] = best
        return best

    def rank(self, c, candidates, B):
        """Plan-time API for C4: candidates = [{'voted_type','dnav_prims',...}] ->
        list of (Qhat or None) aligned with candidates. STOP is optimal when all are
        None (infeasible) or the best Qhat <= 0 (the zero branch)."""
        return [self.q(c, cd['voted_type'], cd['dnav_prims'], B) for cd in candidates]

    def yield_est(self, c, voted_type, dnav_prims):
        """[P10 pilot] Expected IN-ROOM yield alone (no continuation term): the same
        retrieval as q(), but summing only the immediate rewards r_i. Used by the
        early-stop sweep to price a room's remaining unswept value."""
        ids, ws = self._retrieve(c, voted_type, dnav_prims)
        if not ids:
            return 0.0
        s = sum(ws)
        return sum(w * self.bank[i]['r'].get(self.y, 0) for i, w in zip(ids, ws)) / s


# ------------------------------- GATE C3 --------------------------------------
def synth_world(n_rooms, seed):
    """Random connected graph, UNIQUE types -> abstract context == true state."""
    rng = random.Random(seed)
    types = [f"T{i}" for i in range(n_rooms)]
    edges = set()
    for i in range(1, n_rooms):
        edges.add((rng.randrange(i), i))
    for _ in range(n_rooms):
        a, b = rng.randrange(n_rooms), rng.randrange(n_rooms)
        if a != b:
            edges.add((min(a, b), max(a, b)))
    reward = [rng.randint(0, 3) for _ in range(n_rooms)]
    cost = [rng.randint(1, 4) for _ in range(n_rooms)]      # d = cost of entering room
    adj = collections.defaultdict(set)
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    return types, adj, reward, cost


def synth_bank(types, adj, reward, cost, start):
    """Enumerate ALL (partial graph, frontier) one-step transitions reachable from start."""
    bank, seen = [], set()
    def context(cur, obs):
        return {'cur_type': types[cur], 'obs_types': [types[u] for u in sorted(obs)], 'n_obs': len(obs)}
    stack = [(start, frozenset([start]))]
    while stack:
        cur, obs = stack.pop()
        if (cur, obs) in seen:
            continue
        seen.add((cur, obs))
        for u in obs:
            for v in adj[u]:
                if v in obs:
                    continue
                obs2 = obs | {v}
                bank.append({'fold': 0, 'c': context(cur, obs),
                             'v_voted_type': types[v], 'v_true_type': types[v],
                             'd_nav': cost[v], 'd_sweep': 0, 'd': cost[v],
                             'r': {'y': reward[v]}, 'c_next': context(v, obs2)})
                stack.append((v, obs2))
    return bank


def brute_force(adj, reward, cost, start, B):
    """Max total reward over all frontier-expansion sequences within budget B."""
    best = [0]
    def rec(obs, acc, rem):
        best[0] = max(best[0], acc)
        for u in obs:
            for v in adj[u]:
                if v not in obs and cost[v] <= rem:
                    rec(obs | {v}, acc + reward[v], rem - cost[v])
    rec(frozenset([start]), 0, B)
    return best[0]


def gate():
    ok = fail = 0
    for n in (3, 4, 5, 6, 7, 8):
        for seed in range(6):
            types, adj, reward, cost, = synth_world(n, seed)
            bank = synth_bank(types, adj, reward, cost, 0)
            pl = Planner(bank, plan_fold=None, y='y')
            for B in (0, 1, 2, 3, 5, 8, 12, 20):
                c0 = {'cur_type': types[0], 'obs_types': [types[0]], 'n_obs': 1}
                got = pl.v(ckey(c0), B)
                want = brute_force(adj, reward, cost, 0, B)
                if abs(got - want) < 1e-9:
                    ok += 1
                else:
                    fail += 1
                    print(f"  MISMATCH n={n} seed={seed} B={B}: planner={got} brute={want}")
    print(f"[C3] GATE synthetic brute-force agreement: {ok}/{ok+fail} {'PASS' if fail == 0 else 'FAIL'}")


if __name__ == '__main__':
    gate()
