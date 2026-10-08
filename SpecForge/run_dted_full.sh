#!/usr/bin/env bash
# Full DTED training launcher (1 epoch, Qwen3-4B, DFlash-b16 warm start).
#
# Post-Phase-4.5 speedup: expected ~40 h for 1 epoch (72k steps @ ~2 s/step).
# Verified 2026-08-27 via 100-step smoke @ 1.89 s/step, samples/s=14.79.
set -euo pipefail

REPO_ROOT="<WORKSPACE>/ICLR2027/SpecForge"
CONFIG="examples/configs/online/disaggregated/managed-local/qwen3-4b-dted-b16init.yaml"
OUT_DIR="/dockerdata/specforge_outputs/qwen3-4b-dted-b16init-1ep-lr1e5"

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
