#!/usr/bin/env bash
# Stop all SGLang servers launched by ``start_qwen3_8b_regen_servers.sh``.
set -euo pipefail

PID_FILE="${PID_FILE:-/dockerdata/qwen3_8b_regen/sglang_pids.txt}"

if [[ -f "${PID_FILE}" ]]; then
    while IFS= read -r pid; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            echo "killing pid ${pid}"
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done < "${PID_FILE}"
    sleep 10
    # Force-kill any leftovers.
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "mooncake_master" 2>/dev/null || true
    rm -f "${PID_FILE}"
else
    echo "No PID file at ${PID_FILE}; falling back to pkill."
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "mooncake_master" 2>/dev/null || true
fi

sleep 3
echo "sglang procs alive: $(pgrep -f 'sglang' | wc -l)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8
