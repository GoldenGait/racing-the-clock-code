# The pipeline, step by step

All commands run from `search/` with `RTC_DATA` set (see [INSTALL.md](INSTALL.md)). Steps 1 and 2 are optional: their outputs already ship in `search/`.

## 1. Scene abstraction (optional, outputs included)

```bash
# room graphs: rooms, their types, the doorways between them, and the objects in each
python build_room_graphs.py      --out room_graphs_mp3d.json
python build_room_graphs_hm3d.py --out room_graphs_hm3d_v2.json --workers 10

# P(room type | object), counted from Matterport3D house segmentations
python build_prior.py --house-root $RTC_DATA/mp3d_semantics --out prior_table.json
```

Included: `room_graphs_mp3d.json`, `room_graphs_hm3d_v2.json`, `room_graphs_hm3d_p592.json` (the unlabelled-scene expansion, built with `annot_*.py` and `pseudo_graphs.py`), and `prior_table.json`.

## 2. Cross-scene transition bank (optional, outputs included)

```bash
python build_transition_bank.py --graphs room_graphs_mp3d.json \
    --prior prior_table.json --out transition_bank.json
```

The bank holds one-step room transitions gathered across scenes: from a partial view of a house, what entering each frontier room cost and returned. The room-level expert plans over these, never over the hidden graph of the scene it is in. The bank is split into five scene-level folds, and when the expert plans in a scene it uses only transitions from the other folds.

Included: `transition_bank.json` (MP3D), `transition_bank_hm3d_v2.json` (HM3D), and `transition_bank_hm3d_v2_evalfree.json`, a variant with the ten evaluation scenes removed entirely (pass it with `--bank-file`).

## 3. Run the experts

Each run writes `<episode>_trace.json` (the action sequence and the expert's decisions) and `<episode>_frames.mp4` (one RGB frame before every action).

```bash
# room-level Bellman expert, with the paper's settings
python c4_collect_expert.py --dataset hm3d --stem <scene> --cat chair --budget 400 \
    --approach 1 --anytime 1 --resweep 1 --trace-dir $RTC_DATA/traces/room_level

# greedy frontier baseline: same executor, greedy room choice
python c4_collect_expert.py --dataset hm3d --stem <scene> --cat chair --budget 400 \
    --approach 1 --anytime 1 --policy greedy --trace-dir $RTC_DATA/traces/greedy

# object-level Bellman expert: budgeted routing over known target locations
python tsp_oracle.py --dataset hm3d --stem <scene> --cat chair --budget 400 \
    --out $RTC_DATA/traces/object_level/<scene>_chair_B400
```

Useful options of `c4_collect_expert.py`:

| Option | Meaning |
|---|---|
| `--dataset {hm3d,mp3d}` | Scene dataset |
| `--stem`, `--cat`, `--budget` | Scene, target category, action budget |
| `--start-idx`, `--level` | Which start pose, and which floor of a multi-storey house |
| `--policy {bellman,greedy,random}` | How the next room is chosen |
| `--resweep 1` | Re-sweep visited rooms once frontiers are exhausted, so the run uses its whole budget |
| `--sweep-once 1`, `--cam-h`, `--hfov`, `--res` | Robot-matched variant: one look-around per room and the robot camera's geometry |

The paper's training budgets are 50, 150, 400, 800, 1200 and 2000 steps. Target categories are chair, bed, plant, toilet, tv_monitor and sofa.

To replay any trace as video:

```bash
python replay_render.py --trace <..._trace.json> --dataset hm3d --out /tmp/replay
python expert_duel.py --stem mv2HUxq3B53 --cat chair --budget 800 --out /tmp/duel.mp4
```

## 4. Build training records

```bash
python flatten_uninavid.py --trace-dir $RTC_DATA/traces/room_level --out-dir $RTC_DATA/dataset \
    --no-budget-frac 0.2
python check_contamination.py $RTC_DATA/dataset/records.json
```

Each record holds the video history up to a step, the expert's next four primitives, and the instruction with a countdown of the steps remaining. `--no-budget-frac 0.2` emits 20% of records without the countdown, as in the paper. `check_contamination.py` exits with an error if any record comes from an evaluation scene.

For a budget-blind control, build the same records with the countdown removed from every one (`--no-budget-frac 1.0`).

## 5. Fine-tune

```bash
TRAIN_JSON=$RTC_DATA/dataset/records.json VIDEOS=$RTC_DATA/dataset \
    bash ../training/train_uninavid.sh
```

See [`training/README.md`](../training/README.md) for the recipe and hardware.

## 6. Benchmark

The benchmark is 400 frozen episodes: ten held-out HM3D scenes, with fixed target categories, start poses and budgets of 150, 400, 800 and 2000.

`bench_model.py` loads the model and the simulator in one process, so it needs an environment that has both habitat-sim and Uni-NaVid installed. Run it from the Uni-NaVid folder, because the checkpoint refers to the vision tower by a relative path:

```bash
cd $RTC_DATA/Uni-NaVid
REPO=/path/to/racing-the-clock-code
# bench_model.py <shard> <nshards> <gpu> <checkpoint> <output dir>
python $REPO/search/bench_model.py 0 1 0 $RTC_DATA/sft_ckpts/uninavid-budget $RTC_DATA/bench_out
python $REPO/results/score.py "$RTC_DATA/bench_out/results_*.json"
```

If the two cannot share an environment, use `bench_model_bridge.py`, which runs the model in one Python and drives the simulator in another through `sim_bridge.py`.

Runs are sharded and resumable: start several shards on different GPUs and each picks up where it stopped.

To test that the budget changes behavior, and not only how long the model runs:

```bash
cd $RTC_DATA/Uni-NaVid
python $REPO/search/rollout_gate.py --ckpt $RTC_DATA/sft_ckpts/uninavid-budget --gpu 0
```

This rolls the model out on the same episodes at 150 and 2000 steps and compares the trajectories.

## 7. Deploy on a robot

See [`robot/README.md`](../robot/README.md).

## A note on names in the code

Comments and docstrings use short labels from development: `C1` to `C4` are the four stages of the room-level expert (graphs, bank, planner, executor), and labels such as `P3`, `P9` or `P15` refer to individual design decisions. They are kept because the comments explain why each choice was made.
