# Training

`train_uninavid.sh` fine-tunes Uni-NaVid on budget-conditioned expert demonstrations. It is the recipe behind the paper's room-level student.

| Setting | Value |
|---|---|
| Base model | `uninavid-7b-full-224-video-fps-1-grid-2` |
| Trained | Language model and projector. The vision encoder is frozen |
| Learning rate | 1e-5, cosine schedule, 3% warmup |
| Epochs | 1 |
| Effective batch size | 128 (4 GPUs × 2 per GPU × 16 gradient accumulation steps) |
| Precision | bf16, gradient checkpointing on |
| Max sequence length | 2048 tokens |

In the paper this is 3,032 steps over 388,199 records and takes about 62 hours on four RTX PRO 6000 Blackwell GPUs.

```bash
export RTC_DATA=/path/to/data
TRAIN_JSON=$RTC_DATA/dataset/records.json VIDEOS=$RTC_DATA/dataset \
    OUT=$RTC_DATA/sft_ckpts/uninavid-budget bash training/train_uninavid.sh
```

If you use a different number of GPUs, keep the effective batch size at 128 by changing `--per_device_train_batch_size` or `--gradient_accumulation_steps` in the script.

## The Uni-NaVid patch

`uninavid_budget_sft.patch` applies to Uni-NaVid at commit `79ef5ea`. It changes three files:

- **`uninavid/train/train.py`**: adds a validation split (`--val_data_path`), caps the navigation history at a fixed number of frames, and reads records that point at a shared per-episode video with an end frame, so one video serves every decision point of an episode.
- **`uninavid/train/llava_trainer.py`**: evaluation support for the validation split.
- **`uninavid/model/language_model/llava_llama_vid.py`**: registers the model config under its own name, to avoid a clash with the LLaVA config built into newer versions of transformers, and stops forcing offline logging.

## Budget-blind controls

To train a budget-blind control, build the records with the countdown removed from every one (`flatten_uninavid.py --no-budget-frac 1.0`) and train with the same script. The video and action records are identical; only the countdown text differs.
