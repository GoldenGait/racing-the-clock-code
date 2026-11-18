#!/bin/bash
# The object-level (TSP) supervised student as a robot model server on port 8883.
# Separate from the room-level server; nothing else changes.
#
#   RTC_DATA   data root (holds Uni-NaVid/ and the checkpoint)
#   PYTHON     python of the Uni-NaVid environment   (default: python)
#   GPU        GPU index                             (default: 0)
set -u
: "${RTC_DATA:?set RTC_DATA to your data root}"
R="$(cd "$(dirname "$0")" && pwd)"
SEARCH="$(cd "$R/../search" && pwd)"
cd "$RTC_DATA/Uni-NaVid"
CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OV_CKPT=${OV_CKPT:-$RTC_DATA/sft_ckpts/uninavid-budget-tsp-vid} \
PYTHONPATH="$RTC_DATA/Uni-NaVid:$SEARCH" \
exec ${PYTHON:-python} "$R/nav_server.py" --port 8883
