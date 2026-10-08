#!/usr/bin/env bash
# Multi-GPU, multi-dataset, multi-temperature sweep for DDTree + TEV.
#
# Matrix: 4 datasets x 2 temperatures x (optional) N tree_budgets.
# GPU layout: 8 workers, one per (dataset, temperature) tuple, each pinned to a
# dedicated GPU. For each worker, loops over all requested tree budgets and
# repetitions serially.
#
# GPU layout (matches the sweep used in the paper):
#   0: gsm8k      T=0.6      4: mt-bench   T=0.6
#   1: gsm8k      T=1.0      5: mt-bench   T=1.0
#   2: humaneval  T=0.6      6: math500    T=0.6
#   3: humaneval  T=1.0      7: math500    T=1.0
#
# Usage:
#   bash scripts/run_ddtree_tev_multi_dataset.sh
#
# Environment overrides:
#   TARGET_MODEL    Target model path or HF repo id (required)
#   DRAFT_MODEL     DFlash draft model path or HF repo id (required)
#   TREE_BUDGETS    Space-separated tree budgets (default: "64")
#   REPS            Number of repeats per (ds, T, tb) (default: 3, mt-bench uses 2)
#   MAX_NEW_TOKENS  Max new tokens per response (default: 2048)

set -u

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${WORKSPACE}/runs/ddtree_tev_multi_dataset"
LOG_DIR="${WORKSPACE}/logs/ddtree_tev_multi_dataset"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

TARGET_MODEL="${TARGET_MODEL:?TARGET_MODEL is required (path or HF repo id)}"
DRAFT_MODEL="${DRAFT_MODEL:?DRAFT_MODEL is required (path or HF repo id)}"

TREE_BUDGETS_STR="${TREE_BUDGETS:-64}"
read -r -a TREE_BUDGETS_ARR <<< "${TREE_BUDGETS_STR}"
REPS="${REPS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

# (gpu_id, dataset, max_samples, temperature)
JOBS=(
  "0|gsm8k|128|0.6"
  "1|gsm8k|128|1.0"
  "2|humaneval|164|0.6"
  "3|humaneval|164|1.0"
  "4|mt-bench|80|0.6"
  "5|mt-bench|80|1.0"
  "6|math500|128|0.6"
  "7|math500|128|1.0"
)

reps_for_dataset() {
  local ds="$1"
  if [[ "${ds}" == "mt-bench" ]]; then
    # mt-bench prompts are multi-turn; keep repeats lower to bound wall time.
    if [[ "${REPS}" -gt 2 ]]; then echo 2; else echo "${REPS}"; fi
  else
    echo "${REPS}"
  fi
}

run_one_gpu() {
  local gpu_id="$1"
  local dataset="$2"
  local max_samples="$3"
  local temperature="$4"

  local temp_slug="${temperature//./_}"
  local worker_log="${LOG_DIR}/worker_gpu${gpu_id}__${dataset}__t${temp_slug}.log"
  local num_reps
  num_reps="$(reps_for_dataset "${dataset}")"

  {
    echo "[worker gpu=${gpu_id}] dataset=${dataset} T=${temperature} max_samples=${max_samples} reps=${num_reps} started at $(date '+%F %T')"
    for tree_budget in "${TREE_BUDGETS_ARR[@]}"; do
      for rep in $(seq 1 "${num_reps}"); do
        local run_tag="${dataset}__t${temp_slug}__tb${tree_budget}__tev__rep${rep}"
        local save_path="${RUN_DIR}/${run_tag}.pt"
        local log_path="${LOG_DIR}/${run_tag}__gpu${gpu_id}.log"
        # Unique master_port per (gpu, tb) so re-binds inside the same worker do not collide.
        local master_port=$((30300 + gpu_id + tree_budget))

        if [[ -f "${save_path}" ]]; then
          echo "  [${run_tag}] SKIP existing ${save_path}"
          continue
        fi

        echo "  [${run_tag}] START at $(date '+%F %T')  -> ${save_path}"
        local t0
        t0=$(date +%s)

        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        torchrun --nproc_per_node=1 --master_port="${master_port}" \
          "${WORKSPACE}/benchmark.py" \
          --model-name-or-path "${TARGET_MODEL}" \
          --draft-name-or-path "${DRAFT_MODEL}" \
          --dataset "${dataset}" \
          --max-samples "${max_samples}" \
          --max-new-tokens "${MAX_NEW_TOKENS}" \
          --temperature "${temperature}" \
          --tree-budget "${tree_budget}" \
          --skip-baseline-dflash \
          --save-path "${save_path}" \
          > "${log_path}" 2>&1

        local rc=$?
        local dt=$(($(date +%s) - t0))
        echo "  [${run_tag}] DONE rc=${rc} elapsed=${dt}s  log=${log_path}"
      done
    done
    echo "[worker gpu=${gpu_id}] finished at $(date '+%F %T')"
  } >> "${worker_log}" 2>&1
}

PIDS=()
for entry in "${JOBS[@]}"; do
  IFS='|' read -r gpu ds ms temp <<< "${entry}"
  run_one_gpu "${gpu}" "${ds}" "${ms}" "${temp}" &
  PIDS+=($!)
  echo "launched worker gpu=${gpu} dataset=${ds} T=${temp} pid=$!"
done

echo "All ${#JOBS[@]} workers launched. Waiting for completion..."
for pid in "${PIDS[@]}"; do
  wait "${pid}"
done
echo "ALL WORKERS DONE at $(date '+%F %T')"
echo "Results:  ${RUN_DIR}"
echo "Logs:     ${LOG_DIR}"
