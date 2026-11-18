# Data

## What is released, and where

| Data | Location | Size |
|---|---|---|
| Room graphs, semantic prior, transition banks | `search/*.json` | about 60 MB |
| Evaluation scene lists | `search/EVAL_SCENES.txt`, `search/EVAL_SCENES_MP3D.txt` | |
| Benchmark episodes and per-episode results | `results/frozen400/` | under 1 MB |
| Expert traces, about 100,000 episodes | [Releases](../../../releases) page | about 35 MB compressed |

## Released trace sets

All sets exclude the ten evaluation scenes.

| Archive | Expert | Scenes | Budgets | Episodes |
|---|---|---|---|---:|
| `room_level_traces_148scenes` | Room-level Bellman | 89 HM3D and 59 Matterport3D, the paper's training scenes | 2500 | 6,766 |
| `room_level_expert_traces` | Room-level Bellman, with re-sweeping | 60 Matterport3D | 800, 1200, 2000, 2500 | 1,640 |
| `room_level_traces_hm3d_expansion` | Room-level Bellman | 566 HM3D scenes without semantic annotation | 50 to 2000 | 41,048 |
| `object_level_traces_148scenes` | Object-level Bellman | 89 HM3D and 59 Matterport3D, the same scenes as above | 2500 | 3,484 |
| `object_level_traces_multibudget` | Object-level Bellman | 95 HM3D and 63 Matterport3D | 50, 150, 400, 800, 1200, 2000, 2500 | 6,937 |
| `greedy_frontier_traces_hm3d_expansion` | Greedy frontier | 566 HM3D scenes, the same episodes as the room-level expansion set | 50 to 2000 | 41,037 |

The 2500-step runs on the 148 scenes are long single rollouts per start. A shorter budget can be read from the same trace by stopping it early, because these runs choose rooms without looking at the remaining budget.

The expansion sets use `search/room_graphs_hm3d_p592.json`. Point the expert at it with the `GRAPHS_HM3D` environment variable when regenerating or replaying those episodes.

Rendered videos are not distributed, because they are renders of HM3D and Matterport3D scenes. Every trace stores the exact action sequence from a fixed start pose, so the videos can be regenerated with your own copy of the scenes (see below).

## Expert traces

One JSON file per episode, named `<scene>[_L<floor>]_<category>_B<budget>_trace.json`:

| Field | Meaning |
|---|---|
| `stem`, `cat`, `budget`, `level` | Scene, target category, action budget, floor |
| `start_room` | Room the episode starts in. The start pose is that room's centroid, snapped to the navigation mesh |
| `actions` | The primitive actions, in order: `forward` (0.25 m), `left` and `right` (30 degrees) |
| `spent` | Actions executed |
| `collected`, `n_reachable`, `n_targets` | Targets credited, reachable targets, all targets of the category |
| `decisions` | The expert's room choices: step, chosen room, voted room type, value estimate and budget left |
| `found_events` | The step at which each target was credited |
| `stop` | Why the episode ended |
| `resweep_passes` | Number of re-sweep passes after the frontiers were exhausted |

A target is credited when the agent is within range of it and it falls inside the camera's field of view. There is no physical pickup.

### Regenerating videos from traces

```bash
cd search
python replay_render.py --trace <..._trace.json> --dataset hm3d --out /path/to/output
```

The replay re-executes the stored actions from the stored start and re-runs the crediting rule, so the result matches the recorded episode.

## Room graphs

`room_graphs_*.json` maps each scene to:

- `rooms`: for every room, its floor, centroid, type, navigable cells, and the objects it contains
- `edges`: doorways between rooms, with their positions
- `grids`: the navigation grid of each floor

`room_graphs_hm3d_v2.json` covers the annotated HM3D-Semantics scenes and `room_graphs_mp3d.json` covers Matterport3D. `room_graphs_hm3d_p592.json` covers HM3D scenes without semantic annotation: rooms come from a watershed over the navigation mesh, and objects from an automatic detector whose output was audited (`annot_*.py`, `pseudo_graphs.py`).

## Transition bank

`transition_bank*.json` holds one-step room transitions collected across scenes. Each transition records the observed context, the candidate room entered, the reward collected, the cost in primitive actions, and the context afterwards. The room-level expert estimates the value of entering a room from the transitions whose context matches its own, weighted across scenes. Transitions are split into five scene-level folds; planning in a scene uses only the other folds. `transition_bank_hm3d_v2_evalfree.json` additionally removes the ten evaluation scenes.

## Training records

`flatten_uninavid.py` writes `records.json` in Uni-NaVid's training schema, with one entry per decision point:

```json
{
  "id": "NAV_ID_COLLECT_<episode>_0088",
  "video": "nav_videos/<episode>_0088.mp4",
  "conversations": [
    {"from": "human", "value": "... Your assigned task is: 'Find as many chairs as possible. You have 312 steps remaining.' ... determine your next four actions ..."},
    {"from": "gpt",   "value": "forward forward left forward"}
  ],
  "meta": {"t": 88, "N": 312, "budget": 400, "cat": "chair"}
}
```

The instruction text is defined once, in `search/instr.py`, and both the training flattener and the benchmark import it from there, so the model sees byte-identical prompts in training and evaluation.

## Benchmark

`results/frozen400/baselines_frozen.json` lists the 400 evaluation tuples. Each key is `(scene, category, start_index, budget)` over the ten held-out scenes in `search/EVAL_SCENES.txt`, with budgets of 150, 400, 800 and 2000. See [`results/README.md`](../results/README.md).
