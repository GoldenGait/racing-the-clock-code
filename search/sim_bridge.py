"""Two-process rollout bridge for hosts where habitat (py3.9 conda) and the model
stack (py3.10 torch) cannot share an interpreter (for example a cluster
with no py3.10 habitat build).

  SERVER (run under the HABITAT env python): owns the sim, crediting (P3/P6),
  budget, and episode bookkeeping. Newline-JSON protocol on stdio; frames are
  hex-encoded raw RGB (480x640x3) in the JSON — simple beats clever on a local pipe.

    {"cmd": "reset", "stem": s, "cat": c, "budget": B, "start_idx": i}
        -> {"ok": 1, "frame": hex, "N": B, "task": "..."}
    {"cmd": "step", "actions": ["forward", ...]}    (executes until stop/budget)
        -> {"ok": 1, "frame": hex, "N": n, "done": 0|1, "collected": k, "spent": s,
            "stop": reason|null}
    {"cmd": "result"} -> episode summary  |  {"cmd": "quit"}

  CLIENT helper `BridgeEnv` (imported by the py3.10 side): spawns the server with
  the habitat python, exposes reset()/step().

  python sim_bridge.py            # server mode (habitat env)
"""
import json, math, os, sys

DG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DG)

VOCAB = ('forward', 'left', 'right', 'stop')
ACT = {'forward': 'move_forward', 'left': 'turn_left', 'right': 'turn_right'}


# ----------------------------- server (habitat env) -----------------------------
def serve():
    import numpy as np
    import instr
    from c4_collect_expert import make_sim, pick_start, HM3D, COLLECT_R
    from build_transition_bank import edges_of
    graphs = json.load(open(f'{DG}/room_graphs_hm3d_v2.json'))
    state = {}

    def out(obj):
        sys.stdout.write(json.dumps(obj) + '\n')
        sys.stdout.flush()

    def frame_hex():
        rgb = np.asarray(state['sim'].get_sensor_observations()['rgb'])[..., :3]
        return rgb.astype(np.uint8).tobytes().hex()

    def credit():
        s = state
        st_ = s['sim'].get_agent(0).get_state()
        p = np.array(st_.position, np.float32)
        th = 2 * math.atan2(st_.rotation.y, st_.rotation.w)
        fx, fz = -math.sin(th), -math.cos(th)
        for k, (rid, ox, oy, oz, r) in enumerate(s['targets']):
            if k in s['counted'] or abs(oy - p[1]) > 1.5:
                continue
            dx, dz = ox - p[0], oz - p[2]
            d = math.hypot(dx, dz)
            if d - r > COLLECT_R:
                continue
            ang = math.degrees(abs(math.atan2(fx * dz - fz * dx, fx * dx + fz * dz)))
            if ang <= 39.5 or d - r <= 0.3:
                s['counted'].add(k)

    for line in sys.stdin:
        try:
            m = json.loads(line)
        except Exception:
            continue
        cmd = m.get('cmd')
        if cmd == 'quit':
            break
        if cmd == 'reset':
            if state.get('sim') is not None:
                state['sim'].close()
            stem, cat = m['stem'], m['cat']
            G = graphs[stem]; adj = edges_of(G)
            start = pick_start(G, adj, str, int(m.get('start_idx', 0)))
            targets = [(rid, o[1], o[2], o[3], o[4]) for rid, r in G['rooms'].items()
                       for o in r.get('objects_full', []) if o[0] == cat]
            sim = make_sim(os.path.join(HM3D, G['scene_rel'], stem + '.glb'), 0)
            pf = sim.pathfinder
            st = sim.get_agent(0).get_state()
            st.position = np.asarray(pf.snap_point(G['rooms'][start]['centroid']), np.float32)
            sim.get_agent(0).set_state(st)
            counted = {k for k, t in enumerate(targets) if str(t[0]) == str(start)}
            state.update(sim=sim, targets=targets, counted=counted,
                         n_start_ex=len(counted), B=int(m['budget']),
                         budget=int(m['budget']), cat=cat, stop=None)
            blind = bool(m.get('blind', 0))
            state['blind'] = blind
            out({'ok': 1, 'frame': frame_hex(), 'N': state['B'],
                 'task': instr.collect_task(cat, None if blind else state['B'])})
            continue
        if cmd == 'step':
            s = state
            toks = [w for w in m['actions'] if w in VOCAB]
            if not toks or toks[0] == 'stop':
                s['stop'] = 'model stop'
            else:
                for w in toks:
                    if w == 'stop' or s['B'] <= 0:
                        break
                    s['sim'].step(ACT[w]); s['B'] -= 1
                    credit()
            done = 1 if (s['stop'] or s['B'] <= 0) else 0
            out({'ok': 1, 'frame': frame_hex(), 'N': s['B'], 'done': done,
                 'collected': len(s['counted']) - s['n_start_ex'],
                 'spent': s['budget'] - s['B'], 'stop': s['stop'],
                 'task': instr.collect_task(s['cat'], None if s.get('blind') else s['B'])})
            continue
        if cmd == 'result':
            s = state
            out({'collected': len(s['counted']) - s['n_start_ex'],
                 'spent': s['budget'] - s['B'], 'stop': s['stop'] or 'budget exhausted'})


# ----------------------------- client (model env) ------------------------------
class BridgeEnv:
    """Spawn the habitat-env server; expose reset/step to the py3.10 model side."""
    def __init__(self, habitat_python, extra_env=None):
        import subprocess
        env = dict(os.environ)
        env.update(extra_env or {})
        self.p = subprocess.Popen([habitat_python, os.path.abspath(__file__)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  env=env, text=True, bufsize=1)

    def _rpc(self, obj):
        self.p.stdin.write(json.dumps(obj) + '\n')
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('sim server died')
            try:
                return json.loads(line)
            except Exception:
                continue                    # skip habitat log noise on stdout

    @staticmethod
    def _decode(hexs):
        import numpy as np
        return np.frombuffer(bytes.fromhex(hexs), np.uint8).reshape(480, 640, 3)

    def reset(self, stem, cat, budget, start_idx=0, blind=False):
        r = self._rpc({'cmd': 'reset', 'stem': stem, 'cat': cat,
                       'budget': budget, 'start_idx': start_idx, 'blind': int(blind)})
        return self._decode(r['frame']), r['N'], r['task']

    def step(self, actions):
        r = self._rpc({'cmd': 'step', 'actions': actions})
        return (self._decode(r['frame']), r['N'], r['task'], bool(r['done']),
                {'collected': r['collected'], 'spent': r['spent'], 'stop': r['stop']})

    def close(self):
        try:
            self.p.stdin.write('{"cmd": "quit"}\n')
            self.p.stdin.flush()
        except Exception:
            pass
        self.p.terminate()


if __name__ == '__main__':
    serve()
