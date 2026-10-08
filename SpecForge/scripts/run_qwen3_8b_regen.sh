#!/usr/bin/env bash
# Regenerate the perfectblend training set with Qwen3-8B answers.
#
# Prerequisites: run ``start_qwen3_8b_regen_servers.sh`` first and wait
# until all 8 sglang servers report READY. This script drives the
# ``regenerate_train_data.py`` client which parallelizes requests
# across servers.
#
# Design notes
# ------------
# * Input file is the existing Qwen3-4B regen dataset; the client
#   automatically drops the assistant turn and only sends user prompts
#   to the target, so we reuse it as the prompt source without needing
#   the original perfectblend jsonl.
# * ``--reasoning disable`` keeps Qwen3-8B out of thinking mode so the
#   assistant output matches the standard SFT format (no <think>...</think>).
# * ``--temperature 0`` matches the Qwen3-4B regen run for a clean
#   apples-to-apples training comparison.
# * Resume mode is on by default so an interrupted run can pick up
#   where it left off.
set -euo pipefail

REPO_ROOT="<WORKSPACE>/ICLR2027/SpecForge"
INPUT_FILE="${INPUT_FILE:-/dockerdata/datasets/qwen3_4b_regen/perfectblend_train_regen.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/dockerdata/datasets/qwen3_8b_regen}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/perfectblend_train_regen.jsonl}"
MODEL_PATH="${MODEL_PATH:-/dockerdata/models/Qwen3-8B}"
CONCURRENCY="${CONCURRENCY:-64}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/regen.log}"

mkdir -p "${OUTPUT_DIR}"

source /opt/conda/etc/profile.d/conda.sh
conda activate specforge

# Local-only traffic, do NOT route via corp proxy.
unset http_proxy https_proxy || true

# Build server address list (8 servers, ports 30000..30070).
SERVER_ADDRESSES=(
    "localhost:30000"
    "localhost:30010"
    "localhost:30020"
    "localhost:30030"
    "localhost:30040"
    "localhost:30050"
    "localhost:30060"
    "localhost:30070"
)

echo "Config:"
echo "  Input:  ${INPUT_FILE}"
echo "  Output: ${OUTPUT_FILE}"
echo "  Model:  ${MODEL_PATH}"
echo "  Servers: ${SERVER_ADDRESSES[@]}"
echo "  Concurrency (per server): ${CONCURRENCY}"
echo "  Max tokens: ${MAX_TOKENS}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Log: ${LOG_FILE}"

cd "${REPO_ROOT}"

nohup setsid python scripts/regenerate_train_data.py \
    --model "${MODEL_PATH}" \
    --reasoning disable \
    --temperature "${TEMPERATURE}" \
    --concurrency "${CONCURRENCY}" \
    --max-tokens "${MAX_TOKENS}" \
    --server-address "${SERVER_ADDRESSES[@]}" \
    --input-file-path "${INPUT_FILE}" \
    --output-file-path "${OUTPUT_FILE}" \
    --resume \
    > "${LOG_FILE}" 2>&1 &
PID=$!
disown

echo ""
echo "pid=${PID}"
echo "log=${LOG_FILE}"
echo ""
echo "Monitor progress with:"
echo "  tail -f ${LOG_FILE}"
echo "  wc -l ${OUTPUT_FILE}"
