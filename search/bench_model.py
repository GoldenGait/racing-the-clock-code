"""Model-in-the-loop benchmark: the fine-tuned model on the frozen 400 tuples.

Adds the missing row to the N(B) leaderboard (random/greedy/expert-V1/V2/oracle are
already frozen). Same episodes, same crediting, same denominators (graph-BFS
n_reachable, matching the baseline files). Sharded + resumable.

  bench_model.py <shard> <nshards> <gpu> <ckpt> <outdir> [baselines_json]
"""
from paths import DATA_ROOT
import ast, collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    shard, nshards, gpu = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    ckpt, outdir = sys.argv[4], sys.argv[5]
    basef = sys.argv[6] if len(sys.argv) > 6 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'frozen400', 'baselines_frozen.json')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    # upstream Uni-NaVid decode protocol: sampled T=0.2, seeded per episode.
    # (greedy locks into revisit cycles - 92-93% revisited cells at B=2000 -
    # and stays reserved for causal probes.)
    sys.path.insert(0, os.environ.get('OBJEVAL', os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.environ.get('UNINAVID', f'{DATA_ROOT}/Uni-NaVid'))
    from rollout_gate import rollout
    from c4_collect_expert import pick_start
    from build_transition_bank import edges_of
    import objectnav_uninavid as ov
    ov.CKPT = ckpt
    model = ov.Model(0)

    DG = os.path.dirname(os.path.abspath(__file__))
    G_all = json.load(open(f'{DG}/room_graphs_hm3d_v2.json'))
    tuples = sorted(ast.literal_eval(k) for k in json.load(open(basef))['expert'])
    # TUPLE_SUBSET: a json list of tuple keys to restrict this worker to. Lets an
    # idle worker pick up the tail of a campaign (shards that finished early free
    # their GPU while other shards still hold work) without re-sharding, which
    # would mean killing in-flight long-budget rollouts.
    _sub = os.environ.get('TUPLE_SUBSET')
    if _sub:
        keep = {tuple(t) if isinstance(t, list) else t
                for t in (ast.literal_eval(x) if isinstance(x, str) else x
                          for x in json.load(open(_sub)))}
        tuples = [t for t in tuples if t in keep]
    tuples = tuples[shard::nshards]
    os.makedirs(outdir, exist_ok=True)
    rpath = f'{outdir}/results_shard{shard}.json'
    res = json.load(open(rpath)) if os.path.exists(rpath) else {}

    def n_reachable(stem, cat, si):
        G = G_all[stem]; adj = edges_of(G)
        start = pick_start(G, adj, str, int(si))
        level = G['rooms'][start]['level']
        comp, fr = {start}, [start]
        while fr:
            u = fr.pop()
            for v, _ in adj[u]:
                if v not in comp and G['rooms'][v]['level'] == level:
                    comp.add(v); fr.append(v)
        return sum(1 for rid, r in G['rooms'].items() if rid in comp and rid != start
                   for o in r.get('objects_full', []) if o[0] == cat)

    for i, (stem, cat, si, B) in enumerate(tuples):
        key = str((stem, cat, si, B))
        if key in res:
            continue
        try:
            import torch, hashlib
            torch.manual_seed(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
            r = rollout(model, stem, cat, B, 0, start_idx=int(si))
            res[key] = [r['collected'], n_reachable(stem, cat, si), r['spent'], r['stop']]
        except Exception as ex:
            res[key] = ['ERROR', str(ex)[:150], None, None]
        json.dump(res, open(rpath, 'w'))
        print(f'B{shard}: {i+1}/{len(tuples)} {key} -> {res[key]}', flush=True)
    print(f'BENCHSHARD{shard}_DONE', flush=True)


if __name__ == '__main__':
    main()
