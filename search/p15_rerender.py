"""P15-lite: re-render existing expert trajectories with the robot's camera.

The P14 demonstrations are kept (same start, same primitive sequence, same crediting
rule) and only three things change, so the fine-tuned model sees the deployment
distribution:
  1. camera: RealSense D435i on the Go2 (69 deg HFOV, 640x360, lens 0.40 m above the
     floor) instead of the 79 deg / 0.88 m simulator camera; no cropping, a true re-render;
  2. cadence: one stored frame per two primitives (the deployment grabs an image after
     every 2-primitive chunk), so a 360 look-around is 6 images 60 deg apart;
  3. trajectory edits that do not break continuity:
       - the re-sweep tail (if the trace has one) is cut;
       - within one room visit only the first full 360 scan is kept; later complete
         rotations (12 same-direction turns) are removed, which leaves the pose sequence
         continuous because a full rotation returns to the same heading;
       - a deterministic half of the episodes is mirrored (frames flipped, left/right
         swapped in the labels): the expert's turn tie-breaks lean left 2:1 and the
         student copied that at every episode start.
Output per episode: <ep>_frames.mp4 (frames at primitives 0,2,4,...,T; fps 1) and
<ep>_trace.json (edited actions, replayed found_events, decisions with remapped prims,
mirror flag, camera).

    python p15_rerender.py --trace <P14 _trace.json> --dataset hm3d --gpu 0 --out-dir <dir>
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import quaternion  # noqa: F401  (numpy-quaternion, used by habitat agent states)

DG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DG)
import c4_collect_expert as c4                                   # noqa: E402
from build_room_graphs import parse_house                        # noqa: E402

ACT = {"forward": "move_forward", "left": "turn_left", "right": "turn_right"}
SWAP = {"left": "right", "right": "left", "forward": "forward"}
TURNS_360 = 12


def edit_actions(tr, keep_first_scan=True):
    """Return (kept_indices, notes): indices of the original primitives that survive.
    Cuts the re-sweep tail and surplus full scans; see module docstring."""
    acts = tr["actions"]
    T = len(acts)
    end = T
    for d in tr.get("decisions", []):
        if str(d.get("room", "")).upper().startswith("RESWEEP"):
            end = min(end, int(d["prim"]))
            break
    # room visits: decisions that carry a value are room entries; markers (CLEARED,
    # APPROACH, ...) have q None
    entries = sorted({int(d["prim"]) for d in tr.get("decisions", []) if d.get("q") is not None})
    if not entries or entries[0] != 0:
        entries = [0] + entries
    bounds = entries + [end]
    remove = np.zeros(T, dtype=bool)
    remove[end:] = True
    n_scans_cut = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        i = a
        first_kept = False
        while i < b:
            j = i
            while j < b and acts[j] == acts[i] and acts[i] in ("left", "right"):
                j += 1
            L = j - i
            if L >= TURNS_360:
                if keep_first_scan and not first_kept:
                    first_kept = True
                else:
                    full = (L // TURNS_360) * TURNS_360
                    remove[i:i + full] = True
                    n_scans_cut += L // TURNS_360
            i = j if j > i else i + 1
    kept = [k for k in range(T) if not remove[k]]
    return kept, {"resweep_cut": T - end, "scans_cut": n_scans_cut}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, nargs="+", help="trace json(s); one simulator per scene, so pass a whole scene's traces")
    ap.add_argument("--dataset", choices=["mp3d", "hm3d"], default="hm3d")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cam-h", type=float, default=0.40)
    ap.add_argument("--hfov", type=float, default=69.0)
    ap.add_argument("--res", default="320x180", help="stored resolution; the loader pads to square and resizes to 224, so 320x180 loses nothing (same 69 deg FOV)")
    ap.add_argument("--stride", type=int, default=2, help="store a frame every N primitives")
    ap.add_argument("--mirror", default="auto", choices=["auto", "0", "1"],
                    help="auto: deterministic half of the episodes (by name hash)")
    ap.add_argument("--no-scan-cut", action="store_true", help="keep every full scan")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    W, H = map(int, a.res.lower().split("x"))
    c4.CAM_H, c4.HFOV, c4.W, c4.H = a.cam_h, a.hfov, W, H     # the robot's camera, in the expert's sim factory
    todo = []
    for tp in a.trace:
        ep = os.path.basename(tp).replace("_trace.json", "")
        if os.path.exists(os.path.join(a.out_dir, ep + "_trace.json")) and os.path.exists(os.path.join(a.out_dir, ep + "_frames.mp4")):
            print("exists", ep); continue
        todo.append(tp)
    if not todo:
        return
    scenes = {}
    for tp in todo:
        scenes.setdefault(json.load(open(tp))["stem"], []).append(tp)
    for stem, tps in scenes.items():
        G, targets_of, glb = scene_setup(a, stem)
        sim = c4.make_sim(glb, a.gpu)
        for tp in tps:
            try:
                render_one(a, sim, G, targets_of, tp, W, H)
            except Exception as e:                                  # one bad trace must not kill the scene batch
                print(f"FAIL {tp}: {e!r}", flush=True)
        sim.close()


def scene_setup(a, stem):
    """Graph, per-category target lookup and scene file for one scene."""
    if a.dataset == "hm3d":
        graphs = None
        for gp in (os.environ.get("GRAPHS_HM3D", ""), f"{DG}/room_graphs_hm3d_p592.json", f"{DG}/room_graphs_hm3d_v2.json"):
            if gp and os.path.exists(gp):
                g = json.load(open(gp))
                if stem in g:
                    graphs = g; break
        assert graphs is not None, f"{stem}: not in any HM3D graph file"
        G = graphs[stem]
        targets_of = lambda cat: [(rid, o[1], o[2], o[3], o[4]) for rid, r in G["rooms"].items()
                                  for o in r.get("objects_full", []) if o[0] == cat]
        glb = os.path.join(c4.HM3D, G["scene_rel"], stem + ".glb")
    else:
        G = json.load(open(os.environ.get("GRAPHS_MP3D", f"{DG}/room_graphs_mp3d.json")))[stem]
        _, objects = parse_house(f"{c4.MP3D}/{stem}/{stem}.house")
        targets_of = lambda cat: [(ridx, o[0], o[1], o[2], r) for ridx, mp, o, r in objects if mp == cat]
        glb = f"{c4.MP3D}/{stem}/{stem}.glb"
    return G, targets_of, glb


def render_one(a, sim, G, targets_of, trace_path, W, H):
    tr = json.load(open(trace_path))
    ep = os.path.basename(trace_path).replace("_trace.json", "")
    out_mp4 = os.path.join(a.out_dir, ep + "_frames.mp4")
    out_json = os.path.join(a.out_dir, ep + "_trace.json")
    stem, cat, budget, start = tr["stem"], tr["cat"], tr["budget"], str(tr["start_room"])
    mirror = (int(hashlib.md5(ep.encode()).hexdigest(), 16) % 2 == 1) if a.mirror == "auto" else a.mirror == "1"
    targets = targets_of(cat)

    kept, notes = edit_actions(tr, keep_first_scan=not a.no_scan_cut)
    kept_set = set(kept)
    new_index = {}                    # original prim index -> edited prim index
    n = 0
    for k in range(len(tr["actions"]) + 1):
        new_index[k] = n
        if k in kept_set:
            n += 1
    acts_new = [tr["actions"][k] for k in kept]
    T_new = len(acts_new)

    pf = sim.pathfinder
    st = sim.get_agent(0).get_state()
    st.position = np.asarray(pf.snap_point(G["rooms"][start]["centroid"]), np.float32)
    st.rotation = np.quaternion(1, 0, 0, 0)                          # fresh episode: identity heading, as the expert had
    sim.get_agent(0).set_state(st)

    counted, excluded = set(), set()
    for k, (ridx, ox, oy, oz, r) in enumerate(targets):            # P6: start room excluded
        if str(ridx) == start:
            counted.add(k); excluded.add(k)

    def credit(step_new):
        st_ = sim.get_agent(0).get_state(); p = np.array(st_.position, np.float32)
        th = 2 * math.atan2(st_.rotation.y, st_.rotation.w)
        fx, fz = -math.sin(th), -math.cos(th)
        hits = []
        for k, (ridx, ox, oy, oz, r) in enumerate(targets):
            if k in counted or abs(oy - p[1]) > 1.5:
                continue
            dx, dz = ox - p[0], oz - p[2]
            d = math.hypot(dx, dz)
            if d - r > c4.COLLECT_R:
                continue
            ang = math.degrees(abs(math.atan2(fx * dz - fz * dx, fx * dx + fz * dz)))
            if ang <= 39.5 or d - r <= 0.3:
                counted.add(k); hits.append(step_new)
        return hits

    def grab():
        rgb = np.asarray(sim.get_sensor_observations()["rgb"])[..., :3].copy()
        return rgb[:, ::-1] if mirror else rgb

    frames, found_events = [], []
    traj = [c4.apos(sim)[[0, 2]].tolist()]
    step_new = 0
    frames.append(grab())                                          # observation before primitive 0
    # the ORIGINAL sequence is executed (so excised scans still turn the agent back to
    # where it was); frames and credits are recorded only on kept primitives
    last_kept = max(kept) if kept else -1
    for k, w in enumerate(tr["actions"][:last_kept + 1]):
        if k not in kept_set:                                      # excised scan: execute, don't record
            sim.step(ACT[w])
            continue
        sim.step(ACT[w]); step_new += 1
        traj.append(c4.apos(sim)[[0, 2]].tolist())
        found_events += credit(step_new)
        if step_new % a.stride == 0 or step_new == T_new:
            frames.append(grab())

    import imageio
    stored = [frames[i] for i in range(len(frames))]
    # frames list holds observations at kept prims 0, stride, 2*stride, ..., and the final
    # one; drop a duplicated final if T_new happened to be a multiple of stride
    if T_new % a.stride == 0 and len(stored) > T_new // a.stride + 1:
        stored = stored[:T_new // a.stride + 1]
    imageio.mimsave(out_mp4, stored, fps=1, macro_block_size=2)

    decisions = []
    for d in tr.get("decisions", []):
        d2 = dict(d); d2["prim"] = new_index.get(int(d["prim"]), T_new)
        if mirror and isinstance(d2.get("room"), str) and d2["room"] in ("left", "right"):
            d2["room"] = SWAP[d2["room"]]
        decisions.append(d2)
    out = {
        "stem": stem, "cat": cat, "budget": budget, "level": tr.get("level"), "start_room": start,
        "actions": [SWAP[w] if mirror else w for w in acts_new],
        "spent": T_new, "stop": {"reason": tr.get("stop", {}).get("reason", ""), "prim": T_new},
        "found_events": found_events, "collected": len(counted) - len(excluded),
        "n_targets": tr.get("n_targets"), "n_reachable": tr.get("n_reachable"),
        "n_start_excluded": len(excluded), "decisions": decisions,
        "mirror": mirror, "stride": a.stride, "n_frames": len(stored),
        "camera": {"cam_h": a.cam_h, "hfov": a.hfov, "res": [W, H]},
        "edit": notes, "source": os.path.abspath(trace_path),
        "orig_spent": len(tr["actions"]), "orig_collected": tr.get("collected"),
    }
    json.dump(out, open(out_json, "w"))
    print(f"{ep}: {len(tr['actions'])}->{T_new} prims, frames {len(stored)}, "
          f"found {out['collected']} (orig {tr.get('collected')}), mirror={int(mirror)}, edit={notes}")


if __name__ == "__main__":
    main()
