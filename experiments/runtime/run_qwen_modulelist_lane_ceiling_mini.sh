#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/ziheng/backbone-first-moe_fresh_e2e}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/qwen_modulelist_lane_ceiling_mini}"

NUM_PROMPTS="${NUM_PROMPTS:-8}"
NUM_MEASURED_PAIRS="${NUM_MEASURED_PAIRS:-1}"
DEVICE_MEMORY_RATIO="${DEVICE_MEMORY_RATIO:-0.10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-1}"
STORE_CAPACITY="${STORE_CAPACITY:-1000}"
SEED="${SEED:-42}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

log_stage() {
  local stage="$1"
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${stage}"
}

run_mode() {
  local mode_name="$1"
  local no_control_flag="$2"
  local mode_out="${OUT_ROOT}/${mode_name}"
  log_stage "mode=${mode_name}:start out=${mode_out}"
  (
    cd "${REPO_ROOT}"
    env \
      OUT_ROOT="${mode_out}" \
      NO_CONTROL_MODE="${no_control_flag}" \
      NUM_PROMPTS="${NUM_PROMPTS}" \
      NUM_MEASURED_PAIRS="${NUM_MEASURED_PAIRS}" \
      DEVICE_MEMORY_RATIO="${DEVICE_MEMORY_RATIO}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      MAX_LENGTH="${MAX_LENGTH}" \
      MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
      MIN_NEW_TOKENS="${MIN_NEW_TOKENS}" \
      STORE_CAPACITY="${STORE_CAPACITY}" \
      SEED="${SEED}" \
      CUDA_DEVICE="${CUDA_DEVICE}" \
      bash experiments/runtime/run_qwen_modulelist_lane_steady_state.sh
  )
  log_stage "mode=${mode_name}:done out=${mode_out}"
}

mkdir -p "${OUT_ROOT}"

run_mode control 0
run_mode no_control 1
