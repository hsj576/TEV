#!/usr/bin/env bash
# Launch 8 SGLang servers (one per H20-3e GPU) serving Qwen3-Coder-30B-A3B-Instruct.
#
# Each server listens on port 30000 + 10*i (i.e. 30000, 30010, ...).
# Together they provide ~2000 concurrent slots for data regeneration.
#
# Notes on Qwen3-Coder-30B-A3B-Instruct:
#   * model_type: qwen3_moe (48 layers, 128 experts, 8 active) - single-card TP=1 is fine on H20-3e 143 GB.
#   * ~60 GB weights bf16; mem_fraction_static=0.85 leaves ample KV cache headroom.
#   * Non-thinking Instruct variant, so we do NOT pass --reasoning-parser
#     (the model never emits <think> tags).
#
# The servers run detached; PIDs are recorded so
# ``stop_qwen3_coder_30b_regen_servers.sh`` can shut them down cleanly.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/dockerdata/models/Qwen3-Coder-30B-A3B-Instruct}"
LOG_DIR="${LOG_DIR:-/dockerdata/qwen3_coder_30b_regen/sglang_logs}"
PID_FILE="${PID_FILE:-/dockerdata/qwen3_coder_30b_regen/sglang_pids.txt}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-128}"

mkdir -p "${LOG_DIR}"
: > "${PID_FILE}"

source /opt/conda/etc/profile.d/conda.sh
conda activate specforge

# HTTP(S) proxy for HuggingFace lookups in case the tokenizer needs to
# resolve remote metadata (rare, but harmless when set).
export http_proxy="${http_proxy:-<HTTP_PROXY_URL>}"
export https_proxy="${https_proxy:-<HTTP_PROXY_URL>}"
launch_one() {
    local gpu_id=$1
    local port=$2
    local log_path="${LOG_DIR}/server_gpu${gpu_id}.log"

    CUDA_VISIBLE_DEVICES="${gpu_id}" nohup setsid \
        python -m sglang.launch_server \
            --model-path "${MODEL_PATH}" \
            --tp 1 \
            --dtype bfloat16 \
            --mem-fraction-static "${MEM_FRACTION_STATIC}" \
            --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
            --host 0.0.0.0 \
            --port "${port}" \
            --trust-remote-code \
        > "${log_path}" 2>&1 &
    local pid=$!
    disown
    echo "gpu=${gpu_id} port=${port} pid=${pid}"
    echo "${pid}" >> "${PID_FILE}"
}

for i in 0 1 2 3 4 5 6 7; do
    port=$((30000 + i * 10))
    launch_one "${i}" "${port}"
done

echo ""
echo "All 8 SGLang servers launched. Waiting for readiness..."
echo "Log dir: ${LOG_DIR}"
echo "PID file: ${PID_FILE}"
echo ""
echo "Poll readiness with:"
echo "  for i in 0 1 2 3 4 5 6 7; do"
echo "    port=\$((30000 + i * 10))"
echo "    curl -sf http://localhost:\${port}/v1/models >/dev/null && echo \"gpu=\${i} port=\${port} READY\" || echo \"gpu=\${i} port=\${port} NOT READY\""
echo "  done"
