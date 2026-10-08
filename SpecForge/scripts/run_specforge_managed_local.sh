#!/usr/bin/env bash
# ============================================================================
# SpecForge managed-local launcher for Qwen3-4B DFlash training
# Encapsulates all env fixes discovered during smoke testing:
#   1. Conda env `specforge` activation
#   2. Star proxy for network access
#   3. LD_LIBRARY_PATH + LD_PRELOAD for libcudart.so.12 (TileLang requirement)
#   4. WANDB offline + dummy key (SpecForge validate_args requires WANDB_API_KEY)
#   5. Swanlab bridge: mirror wandb logs to swanlab cloud
#
# Usage:
#   bash scripts/run_specforge_managed_local.sh <yaml_path> [log_file]
# ============================================================================
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <yaml_path> [log_file]" >&2
  exit 1
fi

YAML="$1"
LOG_FILE="${2:-<WORKSPACE>/ICLR2027/.specforge_run.log}"

# --- 1. Activate conda env ------------------------------------------------------
source /opt/conda/etc/profile.d/conda.sh
conda activate specforge

# --- 2. Star proxy --------------------------------------------------------------
export ftp_proxy="<HTTP_PROXY_URL>"
# but do not proxy local mooncake/sglang connections
# --- 3. libcudart.so.12 for mooncake + TileLang (spec_capture_sink) -------------
CUDART_DIR=/opt/conda/envs/specforge/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH="$CUDART_DIR:${LD_LIBRARY_PATH:-}"
# Critical: TileLang needs libcudart symbols exposed *globally* (RTLD_GLOBAL).
# LD_PRELOAD achieves this for the whole process tree including sglang server.
export LD_PRELOAD="$CUDART_DIR/libcudart.so.12${LD_PRELOAD:+:$LD_PRELOAD}"

# --- 4. WANDB config (offline mode, dummy key satisfies validate_args) ----------
export WANDB_MODE=offline
export WANDB_API_KEY="${WANDB_API_KEY:-offline_dummy_key}"

# --- 5. SwanLab bridge ----------------------------------------------------------
export SWANLAB_SYNC_WANDB=1
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

# --- Info -----------------------------------------------------------------------
echo "[launcher] yaml       : $YAML"
echo "[launcher] log        : $LOG_FILE"
echo "[launcher] conda env  : $(conda info --envs | grep '\*' || echo unknown)"
echo "[launcher] python     : $(which python)"
echo "[launcher] LD_PRELOAD : $LD_PRELOAD"
echo "[launcher] WANDB_MODE : $WANDB_MODE"
echo "[launcher] SWANLAB    : SYNC_WANDB=$SWANLAB_SYNC_WANDB (key ****${SWANLAB_API_KEY: -4})"

# --- Clean stale control dir (managed_local requires a fresh one) ---------------
CONTROL_DIR=$(python - <<PY
import sys, yaml, pathlib
cfg = yaml.safe_load(open("$YAML"))
try:
    d = cfg["deployment"]["disaggregated"]["control_dir"]
except (KeyError, TypeError):
    d = ""
print(d)
PY
)
if [[ -n "$CONTROL_DIR" && -e "$CONTROL_DIR" ]]; then
  echo "[launcher] removing stale control_dir: $CONTROL_DIR"
  rm -rf "$CONTROL_DIR"
fi

# Prepare wandb dir
WANDB_DIR_PARENT=$(python - <<PY
import yaml
cfg = yaml.safe_load(open("$YAML"))
print(cfg.get("output_dir", "/tmp/specforge_run"))
PY
)
export WANDB_DIR="${WANDB_DIR_PARENT}/wandb"
mkdir -p "$WANDB_DIR"

# --- Launch ---------------------------------------------------------------------
cd <WORKSPACE>/ICLR2027/SpecForge
: > "$LOG_FILE"
echo "[launcher] launching specforge train..."
nohup setsid specforge train --config "$YAML" >> "$LOG_FILE" 2>&1 &
PID=$!
disown
echo "$PID" > <WORKSPACE>/ICLR2027/.specforge_pid.txt
echo "[launcher] PID=$PID  tail -f $LOG_FILE"
