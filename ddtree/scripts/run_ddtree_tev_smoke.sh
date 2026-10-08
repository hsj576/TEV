#!/usr/bin/env bash
# Single-GPU smoke test for DDTree + TEV.
#
# Runs one dataset (default: gsm8k) at temperature 1.0 with a small sample budget,
# writes a .pt file into ${RUN_DIR} and per-run logs into ${LOG_DIR}.
#
# Usage:
#   bash scripts/run_ddtree_tev_smoke.sh
#
# Environment overrides:
#   GPU_ID          GPU index (default: 0)
#   MAX_SAMPLES     Number of prompts (default: 20)
#   TREE_BUDGET     DDTree budget (default: 64)
#   MAX_NEW_TOKENS  Max new tokens per response (default: 2048)
#   TARGET_MODEL    Target model path or HuggingFace repo id (required)
#   DRAFT_MODEL     DFlash draft model path or HuggingFace repo id (required)

set -u

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${WORKSPACE}/runs/ddtree_tev_smoke"
LOG_DIR="${WORKSPACE}/logs/ddtree_tev_smoke"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

TARGET_MODEL="${TARGET_MODEL:?TARGET_MODEL is required (path or HF repo id)}"
DRAFT_MODEL="${DRAFT_MODEL:?DRAFT_MODEL is required (path or HF repo id)}"

GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TREE_BUDGET="${TREE_BUDGET:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

TEMP_SLUG="${TEMPERATURE//./_}"
RUN_TAG="${DATASET}__t${TEMP_SLUG}__tb${TREE_BUDGET}__tev"
SAVE_PATH="${RUN_DIR}/${RUN_TAG}.pt"
LOG_PATH="${LOG_DIR}/${RUN_TAG}__gpu${GPU_ID}.log"
MASTER_PORT=$((29700 + GPU_ID))

echo "==== DDTree + TEV smoke test ===="
echo "  dataset=${DATASET} temperature=${TEMPERATURE} max_samples=${MAX_SAMPLES} tree_budget=${TREE_BUDGET} gpu=${GPU_ID}"
echo "  target=${TARGET_MODEL}"
echo "  draft =${DRAFT_MODEL}"
echo "  save  =${SAVE_PATH}"
echo "  log   =${LOG_PATH}"

t0=$(date +%s)
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" \
  "${WORKSPACE}/benchmark.py" \
  --model-name-or-path "${TARGET_MODEL}" \
  --draft-name-or-path "${DRAFT_MODEL}" \
  --dataset "${DATASET}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --tree-budget "${TREE_BUDGET}" \
  --skip-baseline-dflash \
  --save-path "${SAVE_PATH}" \
  > "${LOG_PATH}" 2>&1

rc=$?
dt=$(($(date +%s) - t0))
echo "==== DONE rc=${rc} elapsed=${dt}s  log=${LOG_PATH} ===="
exit "${rc}"
