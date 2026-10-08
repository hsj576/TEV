#!/usr/bin/env bash
# Phase 4.5 DTED smoke launcher (post CUDA-13 environment rebuild).
#
# Simplified from the Phase 3 CUDA-12 version: the rebuilt conda env
# ships CUDA 13 (nvidia/cu13/lib/libcudart.so.13) and sglang+tilelang
# import cleanly without any LD_PRELOAD tricks -- verified 2026-08-27.
#
# We still need:
#   * HTTP(S) proxy for swanlab.cn if you are behind a firewall.
#   * SWANLAB_API_KEY for the disaggregated worker validator.
set -euo pipefail

REPO_ROOT="<WORKSPACE>/ICLR2027/SpecForge"
CONFIG="examples/configs/online/disaggregated/managed-local/qwen3-4b-dted-b16init-smoke.yaml"
OUT_DIR="/dockerdata/specforge_outputs/qwen3-4b-dted-b16init-smoke"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

source /opt/conda/etc/profile.d/conda.sh
conda activate specforge
export WANDB_MODE=offline
export WANDB_API_KEY="${WANDB_API_KEY:-offline_dummy_key}"
export SWANLAB_SYNC_WANDB=1
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

cd "${REPO_ROOT}"
nohup specforge train --config "${CONFIG}" > "${OUT_DIR}/train.log" 2>&1 &
echo "pid=$!"
