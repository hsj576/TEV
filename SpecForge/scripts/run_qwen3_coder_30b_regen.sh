#!/usr/bin/env bash
# Regenerate the perfectblend training set with Qwen3-Coder-30B-A3B-Instruct answers.
#
# Prerequisites: run ``start_qwen3_coder_30b_regen_servers.sh`` first and wait
# until all 8 sglang servers report READY. This script drives the
# ``regenerate_train_data.py`` client which parallelizes requests across
# servers.
#
# Design notes
# ------------
# * Input file is the original Qwen3-4B regen dataset (1,349,858 lines,
#   the most complete/upstream source). The client automatically drops
#   the assistant turn and only sends user prompts to the target.
# * Qwen3-Coder-30B-A3B-Instruct is a non-thinking Instruct model, so we
#   do NOT pass ``--reasoning`` to the client and do NOT pass
#   ``--reasoning-parser`` to the sglang server.
# * ``--temperature 0`` matches the Qwen3-8B / Qwen3.5-35B regen runs
#   for a consistent apples-to-apples training comparison across
#   target-model families.
# * Resume mode is on by default so an interrupted run can pick up where
#   it left off.
set -euo pipefail

REPO_ROOT="<WORKSPACE>/ICLR2027/SpecForge"
INPUT_FILE="${INPUT_FILE:-<WORKSPACE>/ICLR2027/qwen3_4b_regen/perfectblend_train_regen.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-<WORKSPACE>/ICLR2027/qwen3_30b_regen}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/perfectblend_train_regen_qwen3_coder_30b.jsonl}"
MODEL_PATH="${MODEL_PATH:-/dockerdata/models/Qwen3-Coder-30B-A3B-Instruct}"
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
