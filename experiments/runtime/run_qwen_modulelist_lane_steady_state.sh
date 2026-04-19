#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/ziheng/backbone-first-moe_fresh_e2e}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/qwen_modulelist_lane_steady_state}"
NUM_MEASURED_PAIRS="${NUM_MEASURED_PAIRS:-5}"

MODEL_PATH="${MODEL_PATH:-/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat}"
OFFLOAD_PATH="${OFFLOAD_PATH:-/data/finemoe_offloads/Qwen1.5-MoE-A2.7B-Chat}"
PROMPT_FILE="${PROMPT_FILE:-/data/ziheng/FineMoE-EuroSys26/demo/states/lmsys-chat-1m~eval_prompts.json}"
STATE_FILE="${STATE_FILE:-/data/ziheng/FineMoE-EuroSys26/demo/states/Qwen1.5-MoE-A2.7B-Chat~lmsys-chat-1m~64.pkl}"

DEVICE_MEMORY_RATIO="${DEVICE_MEMORY_RATIO:-0.10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-1}"
STORE_CAPACITY="${STORE_CAPACITY:-1000}"
SEED="${SEED:-42}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NO_CONTROL_MODE="${NO_CONTROL_MODE:-0}"

cd "${REPO_ROOT}"
source /home/ziheng/miniconda3/bin/activate mxmoe

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-/data/ziheng/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data/ziheng/torch_ext_qwen_lane_steady_state}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

mkdir -p "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}" "${OUT_ROOT}"

log_stage() {
  local stage="$1"
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${stage}"
}

run_eval() {
  local output_path="$1"
  local resident_file="$2"
  local budget_override="$3"
  local extra_flags=()
  if [ "${NO_CONTROL_MODE}" = "1" ]; then
    extra_flags+=(--no-control-mode)
  fi
  python -m finemoe.entrypoints.backbone_runtime_eval \
    --model-path "${MODEL_PATH}" \
    --offload-path "${OFFLOAD_PATH}" \
    --prompt-file "${PROMPT_FILE}" \
    --output "${output_path}" \
    --device-memory-ratio "${DEVICE_MEMORY_RATIO}" \
    --prefetch-distance 0 \
    --store-prefix "" \
    --resident-expert-ids-file "${resident_file}" \
    --sparse-budget-bytes-override "${budget_override}" \
    --device cuda:0 \
    --eval-mode offline \
    --batch-size "${BATCH_SIZE}" \
    --num-prompts "${NUM_PROMPTS}" \
    --seed "${SEED}" \
    --max-length "${MAX_LENGTH}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --min-new-tokens "${MIN_NEW_TOKENS}" \
    --store-capacity "${STORE_CAPACITY}" \
    "${extra_flags[@]}" \
    --tag runtime_eval
}

read_budget() {
  python - "$1" <<'PY'
import json, sys
with open(sys.argv[1], "r") as fh:
    print(int(json.load(fh)["sparse_budget_bytes"]))
PY
}

plan_dir="${OUT_ROOT}/plan"
mkdir -p "${plan_dir}"
plan_a_json="${plan_dir}/qwen_A_plan_mem0p10.json"
resident_json="${plan_dir}/qwen_lane_mem0p10.json"

rm -f "${plan_a_json}" "${resident_json}"
log_stage "plan_a:start output=${plan_a_json}"
run_eval "${plan_a_json}" "" 0
log_stage "plan_a:done output=${plan_a_json}"
fixed_budget_bytes="$(read_budget "${plan_a_json}")"

log_stage "resident_select:start budget_bytes=${fixed_budget_bytes}"
python -m experiments.simulation.select_adaptive_resident_set \
  --state-file "${STATE_FILE}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${plan_dir}" \
  --output-prefix qwen_lane \
  --memory-ratios "${DEVICE_MEMORY_RATIO}" \
  --selection-method frontier_prefix \
  --profile-fraction 0.2 \
  --prefetch-windows 0 \
  --sparse-budget-bytes "${fixed_budget_bytes}"
log_stage "resident_select:done output=${resident_json}"

resident_capacity="$(
  python - "${resident_json}" <<'PY'
import json, sys
with open(sys.argv[1], "r") as fh:
    payload = json.load(fh)
print(int(payload.get("selected_resident_capacity", payload.get("resident_capacity", 0))))
PY
)"

warm_dir="${OUT_ROOT}/warmup"
mkdir -p "${warm_dir}"
log_stage "warmup_a:start output=${warm_dir}/qwen_A_warmup.json"
run_eval "${warm_dir}/qwen_A_warmup.json" "" "${fixed_budget_bytes}"
log_stage "warmup_a:done output=${warm_dir}/qwen_A_warmup.json"
if [ "${resident_capacity}" -eq 0 ]; then
  cp "${warm_dir}/qwen_A_warmup.json" "${warm_dir}/qwen_C_warmup.json"
  log_stage "warmup_c:skipped resident_capacity=0"
else
  log_stage "warmup_c:start output=${warm_dir}/qwen_C_warmup.json"
  run_eval "${warm_dir}/qwen_C_warmup.json" "${resident_json}" "${fixed_budget_bytes}"
  log_stage "warmup_c:done output=${warm_dir}/qwen_C_warmup.json"
fi

for pair_idx in $(seq 1 "${NUM_MEASURED_PAIRS}"); do
  pair_dir="${OUT_ROOT}/pair${pair_idx}"
  mkdir -p "${pair_dir}"
  a_json="${pair_dir}/qwen_A_mem0p10_lane_long.json"
  c_json="${pair_dir}/qwen_C_mem0p10_lane_long.json"
  rm -f "${a_json}" "${c_json}"

  if [ $((pair_idx % 2)) -eq 1 ]; then
    log_stage "pair${pair_idx}:A:start output=${a_json}"
    run_eval "${a_json}" "" "${fixed_budget_bytes}"
    log_stage "pair${pair_idx}:A:done output=${a_json}"
    if [ "${resident_capacity}" -eq 0 ]; then
      cp "${a_json}" "${c_json}"
      log_stage "pair${pair_idx}:C:skipped resident_capacity=0"
    else
      log_stage "pair${pair_idx}:C:start output=${c_json}"
      run_eval "${c_json}" "${resident_json}" "${fixed_budget_bytes}"
      log_stage "pair${pair_idx}:C:done output=${c_json}"
    fi
  else
    if [ "${resident_capacity}" -eq 0 ]; then
      log_stage "pair${pair_idx}:A:start output=${a_json}"
      run_eval "${a_json}" "" "${fixed_budget_bytes}"
      log_stage "pair${pair_idx}:A:done output=${a_json}"
      cp "${a_json}" "${c_json}"
      log_stage "pair${pair_idx}:C:skipped resident_capacity=0"
    else
      log_stage "pair${pair_idx}:C:start output=${c_json}"
      run_eval "${c_json}" "${resident_json}" "${fixed_budget_bytes}"
      log_stage "pair${pair_idx}:C:done output=${c_json}"
      log_stage "pair${pair_idx}:A:start output=${a_json}"
      run_eval "${a_json}" "" "${fixed_budget_bytes}"
      log_stage "pair${pair_idx}:A:done output=${a_json}"
    fi
  fi

  python - "${a_json}" "${c_json}" "${pair_idx}" <<'PY'
import json, sys
a_path, c_path, pair_idx = sys.argv[1:4]
with open(a_path) as fa, open(c_path) as fc:
    a=json.load(fa); c=json.load(fc)
gain = (c["generated_tokens_per_sec"] / a["generated_tokens_per_sec"] - 1.0) * 100.0
print(
    f"pair{pair_idx}: "
    f"A={a['generated_tokens_per_sec']:.6f} "
    f"C={c['generated_tokens_per_sec']:.6f} "
    f"gain={gain:.2f}%"
)
PY
done
