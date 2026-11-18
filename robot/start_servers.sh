#!/bin/bash
# Start both inference servers on the GPU host: the budget-conditioned model (:8877)
# and base Uni-NaVid (:8878). Idempotent: kills any existing instances first.
#
#   RTC_DATA   data root (holds Uni-NaVid/ and the checkpoint)
#   PYTHON     python of the Uni-NaVid environment        (default: python)
#   OURS_GPU / BASE_GPU   GPU index for each server       (default: 0 / 1)
#   OV_CKPT    checkpoint for the budget-conditioned model
set -u
: "${RTC_DATA:?set RTC_DATA to your data root}"
R="$(cd "$(dirname "$0")" && pwd)"
SEARCH="$(cd "$R/../search" && pwd)"
PY=${PYTHON:-python}
export PYTHONPATH="$RTC_DATA/Uni-NaVid:$SEARCH"
# the checkpoint config references ./model_zoo/eva_vit_g.pth relatively
cd "$RTC_DATA/Uni-NaVid"

pgrep -f "nav_serve[r].py" | while read p; do kill "$p" 2>/dev/null; done
sleep 3

CUDA_VISIBLE_DEVICES=${OURS_GPU:-0} OV_CKPT=${OV_CKPT:-$RTC_DATA/sft_ckpts/uninavid-resweep-vid} \
  setsid nohup "$PY" "$R/nav_server.py" --port 8877 > "$R/server_ours.log" 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=${BASE_GPU:-1} \
  setsid nohup "$PY" "$R/nav_server.py" --port 8878 > "$R/server_base.log" 2>&1 < /dev/null &
sleep 2
echo "launched: $(pgrep -fc "nav_serve[r].py") processes"
