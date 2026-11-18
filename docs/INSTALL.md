# Installation

The pipeline uses two Python environments, because the simulator and the model need different Python versions:

| Environment | Used for | Tested with |
|---|---|---|
| Simulator | Experts, data generation, replay and visualization | Python 3.9, habitat-sim 0.3.3 |
| Model | Training, benchmarking, the robot inference server | Python 3.10, PyTorch 2.x, transformers 4.37.2 |

Benchmarking a model in simulation needs both. Either install habitat-sim into the model environment as well and use `search/bench_model.py`, or keep them apart and use the two-process bridge (`search/bench_model_bridge.py` with `search/sim_bridge.py`).

## 1. Simulator environment

```bash
conda create -n rtc-sim python=3.9 -y
conda activate rtc-sim
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y
pip install numpy scipy pillow imageio imageio-ffmpeg matplotlib
```

Rendering needs a GPU with EGL. On a multi-GPU machine, select the GPU with `CUDA_VISIBLE_DEVICES` and pass `--gpu 0`.

## 2. Model environment

Follow the installation in the [Uni-NaVid repository](https://github.com/jzhzhang/Uni-NaVid), then apply this repository's patch:

```bash
git clone https://github.com/jzhzhang/Uni-NaVid.git $RTC_DATA/Uni-NaVid
cd $RTC_DATA/Uni-NaVid
git checkout 79ef5ea            # the commit the patch was made against
git apply /path/to/racing-the-clock-code/training/uninavid_budget_sft.patch
```

Download the pretrained Uni-NaVid weights and the EVA-ViT-G vision tower into `$RTC_DATA/Uni-NaVid/model_zoo/` as that repository's README describes.

The patch adds a validation split, a cap on the navigation history frames, and support for records that point at a shared per-episode video. [`training/README.md`](../training/README.md) lists each change.

## 3. Data root

Every script reads from one directory, set with the `RTC_DATA` environment variable (default `~/rtc_data`):

```bash
export RTC_DATA=/path/to/data
```

Expected layout:

```
$RTC_DATA/
├── scene_datasets/versioned_data/hm3d-0.2/hm3d/    HM3D-Semantics v0.2 (train/ and val/)
├── mp3d/scene_datasets/mp3d/                       Matterport3D scenes
├── mp3d_semantics/                                 Matterport3D .house files (for build_prior.py)
├── objectnav_data/objectnav_hm3d_v2/               ObjectNav HM3D episodes (for sample_episodes.py)
├── Uni-NaVid/                                      patched Uni-NaVid clone with model_zoo/
├── traces/                                         expert traces you generate or download
├── dataset/                                        training records and videos
└── sft_ckpts/                                      fine-tuned checkpoints
```

Scene datasets:

- **HM3D-Semantics v0.2**: request access at https://aihabitat.org/datasets/hm3d-semantics/ and download the train and val splits with semantic annotations.
- **Matterport3D**: request access at https://niessner.github.io/Matterport/. The `.house` files come with the `house_segmentations` task data.

After downloading HM3D, run the one-time link step so the scene-dataset config resolves:

```bash
cd search
python link_hm3d_basis.py
```

## 4. Check the installation

No simulator needed:

```bash
python results/score.py           # recomputes four rows of the paper's table
cd search && python budget_planner.py    # runs the planner's built-in check on a synthetic graph
```

With the simulator and HM3D in place:

```bash
cd search
python c4_collect_expert.py --dataset hm3d --stem 5cdEh9F2hJL --cat chair --budget 150 \
    --approach 1 --anytime 1 --resweep 1 --trace-dir /tmp/rtc_check
```

This writes `5cdEh9F2hJL_chair_B150_trace.json` and a frames video into `/tmp/rtc_check`.
