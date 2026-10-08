#!/usr/bin/env bash
# Phase 5 hybrid CE + REINFORCE DTED training launcher.
#
# Config: 1 epoch, lr=1e-4, weight_type=P_tgt, ce_weight=1.0,
#         reinforce_weight=0.1. See docs/22 §5 for the rationale
#         (loss down / expected_al flat root-cause).
set -euo pipefail

REPO_ROOT="<WORKSPACE>/ICLR2027/SpecForge"
CONFIG="examples/configs/online/disaggregated/managed-local/qwen3-4b-dted-b16init-1ep-lr1e4-hybrid.yaml"
OUT_DIR="/dockerdata/specforge_outputs/qwen3-4b-dted-b16init-1ep-lr1e4-hybrid"

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
