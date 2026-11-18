#!/bin/bash
# Fine-tune Uni-NaVid on budget-conditioned expert demonstrations.
#
# This is the recipe behind the paper's room-level student: Uni-NaVid stage-2
# settings, vision encoder frozen, language model and projector trained, learning
# rate 1e-5, one epoch, effective batch size 128 (4 GPUs x 2 per GPU x 16 accumulation).
#
# Before running:
#   1. clone Uni-NaVid (https://github.com/jzhzhang/Uni-NaVid) into $RTC_DATA/Uni-NaVid,
#      download its released weights into model_zoo/ as its README describes, and
#      apply the patch:   git apply /path/to/training/uninavid_budget_sft.patch
#   2. build the training records with search/flatten_uninavid.py (see docs/PIPELINE.md)
#
#   RTC_DATA   data root
#   TRAIN_JSON training records         (default: $RTC_DATA/sft_data/sft_train.json)
#   VAL_JSON   held-out records         (default: $RTC_DATA/sft_data/sft_val.json)
#   VIDEOS     folder the record video paths are relative to
#   OUT        output checkpoint folder (default: $RTC_DATA/sft_ckpts/uninavid-budget)
#   NGPU       number of GPUs           (default: 4)
set -eu
: "${RTC_DATA:?set RTC_DATA to your data root}"
TRAIN_JSON=${TRAIN_JSON:-$RTC_DATA/sft_data/sft_train.json}
VAL_JSON=${VAL_JSON:-$RTC_DATA/sft_data/sft_val.json}
VIDEOS=${VIDEOS:-$RTC_DATA/dataset}
OUT=${OUT:-$RTC_DATA/sft_ckpts/uninavid-budget}
NGPU=${NGPU:-4}

cd "$RTC_DATA/Uni-NaVid"
export PYTHONPATH="$RTC_DATA/Uni-NaVid"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

# Keep per-GPU batch x accumulation x NGPU = 128 if you change NGPU.
torchrun --nproc_per_node "$NGPU" --master_port 29519 uninavid/train/train_mem.py \
  --model_name_or_path ./model_zoo/uninavid-7b-full-224-video-fps-1-grid-2 \
  --version imgsp_v1 \
  --data_path "$TRAIN_JSON" \
  --image_folder "$VIDEOS" \
  --video_folder "$VIDEOS" \
  --vision_tower ./model_zoo/eva_vit_g.pth \
  --image_processor ./uninavid/processor/clip-patch14-224 \
  --tune_vision_encoder False \
  --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 \
  --mm_use_im_start_end False --mm_use_im_patch_token False \
  --image_aspect_ratio pad --group_by_modality_length True \
  --video_fps 1 --compress_type grid:2 --bf16 True \
  --output_dir "$OUT" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --val_data_path "$VAL_JSON" \
  --evaluation_strategy steps --eval_steps 150 \
  --save_strategy steps --save_steps 150 --save_total_limit 3 \
  --learning_rate 1e-5 --weight_decay 0. --warmup_ratio 0.03 \
  --lr_scheduler_type cosine --logging_steps 5 --tf32 True \
  --model_max_length 2048 --gradient_checkpointing True \
  --dataloader_num_workers 4 --lazy_preprocess True --report_to none
