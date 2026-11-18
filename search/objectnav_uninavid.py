#!/usr/bin/env python3
"""
Uni-NaVid ObjectNav eval on HM3D-Semantics v0.2 val (RECONSTRUCTED harness).

No released Uni-NaVid ObjectNav code exists, so this is our reconstruction:
 - habitat-sim 0.3.3 drives the scene (RGB render + navmesh + geodesic).
 - Uni-NaVid model reused exactly as in the VLN eval (task-agnostic control loop:
   predict up to 4 actions, execute 2, re-infer; online token merge across the episode).
 - ObjectNav goal is injected as the instruction text via the same VLN prompt wrapper.
 - Success = STOP within SUCCESS_DIST (geodesic) of any goal-object viewpoint of the
   target category (standard Habitat ObjectNav metric); SPL accordingly.

Reconstruction knobs (unknown from the release): the object-search PROMPT phrasing and
the camera. Defaults chosen to match Habitat ObjectNav standard; tunable via CLI.
"""
from paths import DATA_ROOT
import os
import argparse, glob, gzip, json, os, re, math
import numpy as np
import torch
try:
    import habitat_sim
except ImportError:
    habitat_sim = None   # model-only hosts (Anvil py3.10 env); sim runs in the bridge server
try:
    import magnum as mn
except ImportError:
    mn = None

from uninavid.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from uninavid.conversation import conv_templates, SeparatorStyle

HM3D_ROOT = f"{DATA_ROOT}/scene_datasets/versioned_data/hm3d-0.2/hm3d"
CKPT = f"{DATA_ROOT}/Uni-NaVid/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"

# category -> natural word used in the instruction
CAT_WORD = {"chair": "chair", "bed": "bed", "plant": "plant",
            "toilet": "toilet", "tv_monitor": "TV", "sofa": "sofa"}
# VLN wrapper (identical to agent_uninavid); {} filled with the object-search goal.
PROMPT_WRAP = ("Imagine you are a robot programmed for navigation tasks. You have been given a "
               "video of historical observations and an image of the current observation <image>. "
               "Your assigned task is: '{}'. Analyze this series of images to determine your next "
               "four actions. The predicted action should be one of the following: forward, left, "
               "right, or stop.")


def obj_instruction(style, word):
    if style == "search":
        return f"Search for the {word} and stop when you reach it."
    if style == "find":
        return f"Find the nearest {word}."
    if style == "goto":
        return f"Go to the {word}."
    raise ValueError(style)


def scene_glb(scene_id):
    # 'hm3d_v0.2/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb' -> local .glb
    parts = scene_id.split("/")
    return os.path.join(HM3D_ROOT, "val", parts[-2], parts[-1].replace(".basis.glb", ".glb")), parts[-1]


def make_sim(glb, gpu, hfov, cam_h, width, height):
    bcfg = habitat_sim.SimulatorConfiguration()
    bcfg.scene_id = glb; bcfg.gpu_device_id = gpu; bcfg.enable_physics = False
    s = habitat_sim.CameraSensorSpec()
    s.uuid = "rgb"; s.sensor_type = habitat_sim.SensorType.COLOR
    s.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    s.resolution = [height, width]
    s.position = mn.Vector3(0.0, cam_h, 0.0); s.orientation = mn.Vector3(0.0, 0.0, 0.0)
    s.hfov = float(hfov)
    ag = habitat_sim.agent.AgentConfiguration()
    ag.sensor_specifications = [s]
    ag.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(bcfg, [ag]))
    if not sim.pathfinder.is_loaded:
        ns = habitat_sim.NavMeshSettings(); ns.set_defaults(); ns.agent_radius = 0.18; ns.agent_height = cam_h
        sim.recompute_navmesh(sim.pathfinder, ns)
    return sim


def geodesic(pf, a, b):
    sp = habitat_sim.ShortestPath(); sp.requested_start = a; sp.requested_end = b
    return sp.geodesic_distance if pf.find_path(sp) else math.inf


def nearest_vp_dist(pf, pos, vp_positions):
    return min((geodesic(pf, pos, vp) for vp in vp_positions), default=math.inf)


class Model:
    """Minimal Uni-NaVid inference wrapper (mirrors offline_eval + agent_uninavid loop)."""
    def __init__(self, gpu):
        self.conv_mode = "vicuna_v1"
        name = get_model_name_from_path(CKPT)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(CKPT, None, name)
        self.model.config.run_type = "eval"
        self.rgb_list = []

    def reset(self):
        self.rgb_list = []
        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0

    def add_frame(self, rgb):
        self.rgb_list.append(rgb)

    @torch.inference_mode()
    def predict(self, goal_text):
        tok = self.tokenizer
        specials = {k: tok(v, return_tensors="pt").input_ids[0][1:].cuda() for k, v in {
            "vs": "<video_special>", "ve": "</video_special>", "is": "<image_special>",
            "ie": "</image_special>", "nav": "[Navigation]", "sep": "<image_sep>"}.items()}
        qs = PROMPT_WRAP.format(goal_text)
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs.replace("<image>", "")
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs); conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        tp = tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").cuda()
        idxs = torch.where(tp == -200)[0]; parts = []
        while idxs.numel() > 0:
            i = idxs[0]
            parts += [tp[:i], specials["vs"], specials["sep"], tp[i:i+1], specials["ve"],
                      specials["is"], specials["ie"], specials["nav"]]
            tp = tp[i+1:]; idxs = torch.where(tp == -200)[0]
        if tp.numel() > 0: parts.append(tp)
        input_ids = torch.cat(parts, dim=0).unsqueeze(0)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping = KeywordsStoppingCriteria([stop_str], tok, input_ids)
        batch = np.asarray(self.rgb_list)
        self.model.get_model().new_frames = len(self.rgb_list)
        video = self.image_processor.preprocess(batch, return_tensors="pt")["pixel_values"].half().cuda()
        self.rgb_list = []
        self.model.update_prompt([[PROMPT_WRAP.format(goal_text).replace("<image>", "").replace("\n", "")]])
        out = self.model.generate(input_ids, images=[video],
                                  do_sample=(os.environ.get("OV_GREEDY") != "1"), temperature=0.2,
                                  max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping])
        txt = tok.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        if txt.endswith(stop_str): txt = txt[:-len(stop_str)]
        return txt.strip()


def run(args):
    files = sorted(glob.glob(os.path.join(args.episodes, "content", "*.json.gz")))
    files = files[args.split_id::args.split_num]  # round-robin scene assignment
    os.makedirs(os.path.join(args.result, "log"), exist_ok=True)
    model = Model(args.gpu)
    ACT = {1: "move_forward", 2: "turn_left", 3: "turn_right"}
    n_done = 0

    for f in files:
        d = json.load(gzip.open(f))
        gbc = d["goals_by_category"]
        eps = d["episodes"]
        if not eps: continue
        glb, sk = scene_glb(eps[0]["scene_id"])
        if not os.path.exists(glb):
            print("MISSING scene", glb); continue
        sim = make_sim(glb, args.gpu, args.hfov, args.cam_h, args.width, args.height)
        pf = sim.pathfinder
        for ep in eps:
            cat = ep["object_category"]
            # episode_id is NOT unique within a scene (repeats per category) -> include cat
            done_path = os.path.join(args.result, "log", f"stats_{ep['episode_id']}_{cat}_{sk}.json")
            if os.path.exists(done_path): continue
            vps = [np.array(vp["agent_state"]["position"], dtype=np.float32)
                   for g in gbc.get(f"{sk}_{cat}", []) for vp in g.get("view_points", [])]
            if not vps: continue
            start = np.array(ep["start_position"], dtype=np.float32)
            st = sim.get_agent(0).get_state()
            st.position = start
            st.rotation = np.quaternion(ep["start_rotation"][3], *ep["start_rotation"][:3])
            sim.get_agent(0).set_state(st)
            shortest = nearest_vp_dist(pf, start, vps)
            model.reset()
            goal_text = obj_instruction(args.prompt_style, CAT_WORD[cat])
            pending, steps, path_len, prev = [], 0, 0.0, start.copy()
            stopped = False
            while steps < args.max_steps:
                rgb = np.asarray(sim.get_sensor_observations()["rgb"])[..., :3]
                model.add_frame(rgb)
                if not pending:
                    txt = model.predict(goal_text)
                    for w in txt.split(" "):
                        if w == "stop": pending.append(0)
                        elif w == "forward": pending.append(1)
                        elif w == "left": pending.append(2)
                        elif w == "right": pending.append(3)
                        if len(pending) == 2: break
                    if not pending: pending = [0]  # empty output -> stop
                a = pending.pop(0)
                if a == 0:
                    stopped = True; break
                sim.step(ACT[a])
                cur = np.array(sim.get_agent(0).get_state().position, dtype=np.float32)
                path_len += float(np.linalg.norm(cur - prev)); prev = cur
                steps += 1
            final = np.array(sim.get_agent(0).get_state().position, dtype=np.float32)
            dtg = nearest_vp_dist(pf, final, vps)
            success = int(stopped and dtg <= args.success_dist)
            spl = success * (shortest / max(path_len, shortest, 1e-6)) if math.isfinite(shortest) else 0.0
            rec = {"id": ep["episode_id"], "scene": sk, "category": cat, "success": success,
                   "spl": float(spl), "distance_to_goal": float(dtg) if math.isfinite(dtg) else 999.0,
                   "path_length": float(path_len), "steps": steps, "stopped": int(stopped)}
            with open(done_path, "w") as fp: json.dump(rec, fp)
            print(f"[ep {ep['episode_id']} {sk} {cat}] success={success} dtg={dtg:.2f} steps={steps} stopped={stopped}", flush=True)
            n_done += 1
            if args.limit and n_done >= args.limit:
                sim.close(); print("[objnav] limit reached", args.limit); return
        sim.close()
    print("[objnav] chunk", args.split_id, "DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=f"{DATA_ROOT}/objectnav_data/objectnav_hm3d_v2/val")
    ap.add_argument("--result", required=True)
    ap.add_argument("--split-num", type=int, default=1)
    ap.add_argument("--split-id", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--success-dist", type=float, default=1.0)
    ap.add_argument("--prompt-style", default="search", choices=["search", "find", "goto"])
    ap.add_argument("--hfov", type=float, default=79.0)
    ap.add_argument("--cam-h", type=float, default=0.88)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--limit", type=int, default=0)
    run(ap.parse_args())
