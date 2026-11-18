"""Stage 1 of automatic scene annotation: render a navigable sweep of one scene.

Writes RGB, depth and camera pose for a grid of navigable viewpoints so the
detector stage can lift 2D instance masks into 3D. Kept in its own process
because torch and habitat_sim cannot share one (LLVM option clash), so this
never imports torch -- the detector reads the npz back.

  python annot_render.py --dataset hm3d --stem 00009-vLpv2VX547B --out /tmp/x.npz
"""
import argparse
import math
import os

import numpy as np
import habitat_sim
import magnum as mn

from c4_collect_expert import HM3D, MP3D, CAM_H, HFOV, W, H

SPACING = 1.5        # metres between viewpoint columns
HEADINGS = 6         # yaw samples per column
DEPTH_MAX = 8.0      # metres; beyond this a pixel carries no usable position


def make_sim(glb, gpu):
    b = habitat_sim.SimulatorConfiguration()
    b.scene_id = glb; b.gpu_device_id = gpu; b.enable_physics = False
    specs = []
    for uuid, stype in (("rgb", habitat_sim.SensorType.COLOR),
                        ("depth", habitat_sim.SensorType.DEPTH)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid; s.sensor_type = stype
        s.resolution = [H, W]; s.position = mn.Vector3(0, CAM_H, 0); s.hfov = HFOV
        specs.append(s)
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(b, [ag]))
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
    ns.agent_radius = 0.18; ns.agent_height = CAM_H
    sim.recompute_navmesh(sim.pathfinder, ns)
    return sim


def viewpoints(pf, spacing=SPACING, seed=0):
    """Navigable positions on a regular xz lattice, one per occupied cell."""
    lo, hi = pf.get_bounds()
    rng = np.random.default_rng(seed)
    seen, pts = set(), []
    n = max(4000, int(((hi[0] - lo[0]) * (hi[2] - lo[2])) / (spacing * spacing)) * 40)
    for _ in range(n):
        p = pf.get_random_navigable_point()
        if not np.all(np.isfinite(p)):
            continue
        k = (int(math.floor(p[0] / spacing)), int(math.floor(p[1] / 1.0)),
             int(math.floor(p[2] / spacing)))
        if k in seen:
            continue
        seen.add(k); pts.append(np.asarray(p, np.float32))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mp3d', 'hm3d'], required=True)
    ap.add_argument('--stem', required=True)
    ap.add_argument('--scene-rel', default='', help='hm3d only: split/00000-STEM dir')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--spacing', type=float, default=SPACING)
    ap.add_argument('--max-views', type=int, default=1800)
    a = ap.parse_args()

    if a.dataset == 'mp3d':
        glb = f"{MP3D}/{a.stem}/{a.stem}.glb"
    else:
        # HM3D ships some scenes basis-compressed and some not; only one of the two
        # names exists per scene, and habitat rejects the missing one outright.
        d = os.path.join(HM3D, a.scene_rel)
        glb = next(p for p in (os.path.join(d, a.stem + ".basis.glb"),
                               os.path.join(d, a.stem + ".glb")) if os.path.exists(p))
    sim = make_sim(glb, a.gpu)
    pts = viewpoints(sim.pathfinder, a.spacing)
    yaws = [2 * math.pi * i / HEADINGS for i in range(HEADINGS)]
    if len(pts) * HEADINGS > a.max_views:
        step = math.ceil(len(pts) * HEADINGS / a.max_views)
        pts = pts[::step]

    rgb, dep, pos, rot = [], [], [], []
    st = sim.get_agent(0).get_state()
    for p in pts:
        for th in yaws:
            st.position = p
            st.rotation = np.quaternion(math.cos(th / 2), 0, math.sin(th / 2), 0)
            sim.get_agent(0).set_state(st)
            o = sim.get_sensor_observations()
            d = np.asarray(o["depth"], np.float32)
            if not np.isfinite(d).any() or (d > 0).mean() < 0.05:
                continue                     # camera inside geometry
            rgb.append(np.asarray(o["rgb"])[..., :3].copy())
            dep.append(np.clip(d, 0, DEPTH_MAX).astype(np.float16))
            pos.append(np.asarray(p, np.float32)); rot.append(np.float32(th))

    fx = (W / 2.0) / math.tan(math.radians(HFOV) / 2.0)
    np.savez_compressed(a.out, rgb=np.stack(rgb), depth=np.stack(dep),
                        pos=np.stack(pos), yaw=np.stack(rot),
                        fx=np.float32(fx), fy=np.float32(fx),
                        cx=np.float32(W / 2.0), cy=np.float32(H / 2.0),
                        cam_h=np.float32(CAM_H), stem=a.stem)
    print(f"RENDERED {a.stem}: {len(rgb)} views from {len(pts)} columns -> {a.out}")


if __name__ == '__main__':
    main()
