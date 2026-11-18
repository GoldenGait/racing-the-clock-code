# Benchmark results

The benchmark is 400 frozen episodes on ten held-out HM3D scenes: fixed scenes, target categories and start poses, at budgets of 150, 400, 800 and 2000 steps.

```bash
python results/score.py                              # the table below
python results/score.py "my_run/results_*.json"      # add your own run as an extra row
```

Output:

```
policy                                      B=150          B=400          B=800         B=2000
Greedy frontier expert                 24.4 ±6.3      52.1 ±7.1      85.8 ±4.7      93.8 ±2.9
Object-level Bellman expert            91.3 ±3.3      99.8 ±0.3     100.0 ±0.0     100.0 ±0.0
Room-level Bellman expert              45.7 ±7.6      91.6 ±3.3      97.1 ±1.9      98.0 ±1.7
Pretrained VLN                         28.3 ±6.1      35.5 ±6.3      38.8 ±6.4      44.7 ±7.1
VLN trained on room-level expert       31.6 ±6.8      58.0 ±7.5      64.5 ±7.1      72.7 ±6.9
```

The means match the paper's table to within 0.1. The intervals are bootstrap estimates and move by a few tenths between runs of different software versions. The one larger difference is the room-level expert's interval, which this script computes narrower than the paper reports.

## Files in `frozen400/`

| File | Contents |
|---|---|
| `baselines_frozen.json` | The 400 evaluation tuples and the non-learned baselines. `greedy` and `random` are the frontier baselines. `expert` and `oracle` are earlier versions of the two Bellman experts, kept because the benchmark scripts read the tuple list from this file |
| `reachable.json` | The number of reachable targets for each tuple, used as the denominator for every method |
| `object_level_expert.json` | The object-level Bellman expert |
| `room_level_expert.json` | The room-level Bellman expert |
| `pretrained_vln.json` | Uni-NaVid before fine-tuning |
| `room_level_vln.json` | Uni-NaVid fine-tuned on room-level expert demonstrations |

Each results file maps a tuple `"(scene, category, start_index, budget)"` to `[found, reachable, steps, stop_reason]`.

Per-episode results for the other fine-tuned models in the paper's table (trained on the object-level and greedy experts, and the budget-blind controls) are not included yet.

## Scoring

For each tuple, the score is `min(found, reachable) / reachable`. Tuples with no reachable target are dropped, which leaves 98 tuples per budget. The table reports the mean per budget and a 95% percentile bootstrap interval (10,000 resamples, seed 0) as a half-width.
