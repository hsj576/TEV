#!/usr/bin/env bash
# Follow-up DTED training launcher (2 epochs, lr=1e-4, Qwen3-4B, DFlash-b16 warm start).
#
# Launched automatically by watchdog_dted_next.sh once the 1-epoch lr=1e-5
# run completes. Manually invoked the same way if you need to (re)start.
set -euo pipefail

REPO_ROOT="<WORKSPACE>/ICLR2027/SpecForge"
CONFIG="examples/configs/online/disaggregated/managed-local/qwen3-4b-dted-b16init-2ep-lr1e4.yaml"
OUT_DIR="/dockerdata/specforge_outputs/qwen3-4b-dted-b16init-2ep-lr1e4"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

source /opt/conda/etc/profile.d/conda.sh
conda activate specforge
export WANDB_MODE=offline
export WANDB_API_KEY="${WANDB_API_KEY:-offline_dummy_key}"
export SWANLAB_SYNC_WANDB=1
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

cd "${REPO_ROOT}"
nohup setsid specforge train --config "${CONFIG}" > "${OUT_DIR}/train.log" 2>&1 &
PID=$!
disown
echo "$PID" > "${OUT_DIR}/train.pid"
echo "pid=${PID}"
echo "log=${OUT_DIR}/train.log"
