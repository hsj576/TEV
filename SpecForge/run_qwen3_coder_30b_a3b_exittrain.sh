#!/usr/bin/env bash
# Qwen3-Coder-30B-A3B-Instruct (MoE) ExitTrain (DTED full-tree-verify)
# training launcher.
#
# 2 epochs, lr=1e-4, num_anchors=32, tree_budget=32, prefix_window=64.
# Warm-start from a Qwen3-Coder-30B-A3B-DFlash checkpoint
# (8 draft layers, target_layer_ids=[1,12,23,34,45]).
#
# Note: target is a qwen3_moe model (48 layers, 128 experts, 8 active,
# full attention). The yaml bumps sglang_mem_fraction_static to 0.65
# because the target weights are ~60 GB (vs. ~16 GB for a dense 8B model).
set -euo pipefail

# Repository root: default to the parent directory of this script.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
CONFIG="examples/configs/online/disaggregated/managed-local/qwen3-coder-30b-a3b-exittrain.yaml"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/qwen3-coder-30b-a3b-exittrain}"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

# Activate the SpecForge Python environment. Adjust CONDA_HOME / env name
# for your local setup if needed.
if [ -n "${CONDA_HOME:-}" ] && [ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_HOME}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-specforge}"
fi

# Tracking. Leave WANDB in offline mode by default; provide your own
# SWANLAB_API_KEY via the environment if you want to sync to swanlab.ai.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-offline_dummy_key}"
export SWANLAB_SYNC_WANDB="${SWANLAB_SYNC_WANDB:-1}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

cd "${REPO_ROOT}"
nohup setsid specforge train --config "${CONFIG}" > "${OUT_DIR}/train.log" 2>&1 &
PID=$!
disown
echo "$PID" > "${OUT_DIR}/train.pid"
echo "pid=${PID}"
echo "log=${OUT_DIR}/train.log"
