"""Anvil benchmark client: model (this interpreter, py3.10 torch) drives episodes in
a sim_bridge server (habitat py3.9 env). Same protocol/metrics as bench_model.py.

  HABITAT_PY=<habitat env python> bench_model_bridge.py <shard> <nshards> <ckpt> <outdir> [baselines]
"""
from paths import DATA_ROOT
import collections, hashlib, json, math, os, sys

DG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DG)

# The model process must NEVER import habitat: when torch and habitat's GL stack
# land in one process (single-env hosts like the Jetstream2 nodes, where both are
# installed side by side) LLVM registers its options twice and aborts with
# "Option 'default' already exists!". Poisoning the module name makes the guarded
# import in objectnav_uninavid fall back to its model-only path; the simulator
# lives in the bridge server process instead.
for _m in ('habitat_sim', 'magnum'):
    sys.modules.setdefault(_m, None)
import numpy as np

MAX_HIST = 448
VOCAB = ('forward', 'left', 'right', 'stop')


def pick_start(G, adj, start_idx=0):
    first = next((i for i, r in sorted(G['rooms'].items())
                  if r['type'] == 'hallway' and len(adj[i]) >= 4),
                 max(G['rooms'], key=lambda i: len(adj[i])))
    if start_idx == 0:
        return first
    ranked = [i for i, r in sorted(G['rooms'].items())
              if r['type'] == 'hallway' and len(adj[i]) >= 4]
    ranked += [i for i in sorted(G['rooms'], key=lambda i: (-len(adj[i]), str(i))) if i not in ranked]
    ranked = [i for i in ranked if i != first]
    return ranked[min(start_idx - 1, len(ranked) - 1)]


def main():
    import ast
    shard, nshards = int(sys.argv[1]), int(sys.argv[2])
    ckpt, outdir = sys.argv[3], sys.argv[4]
    basef = sys.argv[5] if len(sys.argv) > 5 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'frozen400', 'baselines_frozen.json')
    sys.path.insert(0, os.environ.get('OBJEVAL', os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.environ.get('UNINAVID', f'{DATA_ROOT}/Uni-NaVid'))
    from sim_bridge import BridgeEnv
    import objectnav_uninavid as ov
    ov.CKPT = ckpt
    model = ov.Model(0)
    def fresh_env():
        return BridgeEnv(os.environ['HABITAT_PY'],
                         extra_env={'HABITAT_SIM_LOG': 'quiet', 'MAGNUM_LOG': 'quiet'})
    env = fresh_env()

    G_all = json.load(open(f'{DG}/room_graphs_hm3d_v2.json'))
    tuples = sorted(ast.literal_eval(k) for k in json.load(open(basef))['expert'])
    _sub = os.environ.get('TUPLE_SUBSET')      # see bench_model.py: campaign-tail pickup
    if _sub:
        keep = {ast.literal_eval(x) if isinstance(x, str) else tuple(x)
                for x in json.load(open(_sub))}
        tuples = [t for t in tuples if t in keep]
    tuples = tuples[shard::nshards]
    os.makedirs(outdir, exist_ok=True)
    rpath = f'{outdir}/results_shard{shard}.json'
    res = json.load(open(rpath)) if os.path.exists(rpath) else {}

    def n_reachable(stem, cat, si):
        G = G_all[stem]
        adj = collections.defaultdict(list)
        for e in G['edges']:
            adj[str(e['u'])].append(str(e['v'])); adj[str(e['v'])].append(str(e['u']))
        start = pick_start(G, adj, int(si))
        level = G['rooms'][start]['level']
        comp, fr = {start}, [start]
        while fr:
            u = fr.pop()
            for v in adj[u]:
                if v not in comp and G['rooms'][v]['level'] == level:
                    comp.add(v); fr.append(v)
        return sum(1 for rid, r in G['rooms'].items() if rid in comp and rid != start
                   for o in r.get('objects_full', []) if o[0] == cat)

    import torch, instr  # noqa: instr for parity checks if needed
    def is_err(v):
        return v == 'ERROR' or (isinstance(v, list) and v and v[0] == 'ERROR')

    for i, (stem, cat, si, B) in enumerate(tuples):
        key = str((stem, cat, si, B))
        if key in res and not is_err(res[key]):
            continue                        # retry leftover ERROR entries
        for attempt in (0, 1):              # one respawn-retry per tuple
            try:
                torch.manual_seed(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
                frame, N, task = env.reset(stem, cat, B, int(si), blind=bool(int(os.environ.get('BLIND', '0'))))
                frames = [frame]
                info = {'collected': 0, 'spent': 0, 'stop': None}
                done = False
                while not done:
                    hist = frames
                    if len(hist) > MAX_HIST:
                        idx = np.unique(np.linspace(0, len(hist) - 1, MAX_HIST).astype(int))
                        hist = [hist[j] for j in idx]
                    model.reset(); model.rgb_list = list(hist)
                    txt = model.predict(task)
                    toks = [w for w in txt.split() if w in VOCAB][:4] or ['stop']
                    frame, N, task, done, info = env.step(toks)
                    frames.append(frame)
                res[key] = [info['collected'], n_reachable(stem, cat, si), info['spent'],
                            info['stop'] or 'budget exhausted']
                break
            except Exception as ex:
                try:
                    env.close()
                except Exception:
                    pass
                env = fresh_env()           # a dead sim server must not poison later tuples
                if attempt == 1:
                    res[key] = ['ERROR', str(ex)[:150], None, None]
        # merge-write: never clobber a concurrent writer's clean value with our ERROR
        try:
            disk = json.load(open(rpath)) if os.path.exists(rpath) else {}
        except Exception:
            disk = {}
        for k, v in disk.items():
            if k not in res or (is_err(res[k]) and not is_err(v)):
                res[k] = v
        json.dump(res, open(rpath, 'w'))
        print(f'A{shard}: {i+1}/{len(tuples)} {key} -> {res[key][:3]}', flush=True)
    env.close()
    print(f'ANVILSHARD{shard}_DONE', flush=True)


if __name__ == '__main__':
    main()
