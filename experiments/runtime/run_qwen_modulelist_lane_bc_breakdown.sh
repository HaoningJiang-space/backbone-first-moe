#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/ziheng/backbone-first-moe_fresh_e2e}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/qwen_runtime_breakdown_bc}"

MODEL_PATH="${MODEL_PATH:-/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat}"
OFFLOAD_PATH="${OFFLOAD_PATH:-/data/finemoe_offloads/Qwen1.5-MoE-A2.7B-Chat}"
PROMPT_FILE="${PROMPT_FILE:-/data/ziheng/FineMoE-EuroSys26/demo/states/lmsys-chat-1m~eval_prompts.json}"
STATE_FILE="${STATE_FILE:-/data/ziheng/FineMoE-EuroSys26/demo/states/Qwen1.5-MoE-A2.7B-Chat~lmsys-chat-1m~64.pkl}"

DEVICE_MEMORY_RATIO="${DEVICE_MEMORY_RATIO:-0.10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-1}"
STORE_CAPACITY="${STORE_CAPACITY:-1000}"
SEED="${SEED:-42}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"

cd "${REPO_ROOT}"
source /home/ziheng/miniconda3/bin/activate mxmoe

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-/data/ziheng/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data/ziheng/torch_ext_qwen_lane_bc_breakdown}"
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
  local mode="$4"
  local extra_flags=()
  if [ "${mode}" = "B" ]; then
    extra_flags+=(--disable-backbone-lane-split)
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
    --tag "runtime_breakdown_${mode}" \
    "${extra_flags[@]}"
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
run_eval "${plan_a_json}" "" 0 "A"
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

warm_dir="${OUT_ROOT}/warmup"
mkdir -p "${warm_dir}"
log_stage "warmup_b:start"
run_eval "${warm_dir}/qwen_B_warmup.json" "${resident_json}" "${fixed_budget_bytes}" "B"
log_stage "warmup_b:done"
log_stage "warmup_c:start"
run_eval "${warm_dir}/qwen_C_warmup.json" "${resident_json}" "${fixed_budget_bytes}" "C"
log_stage "warmup_c:done"

b_json="${OUT_ROOT}/qwen_B_breakdown.json"
c_json="${OUT_ROOT}/qwen_C_breakdown.json"
rm -f "${b_json}" "${c_json}"

log_stage "measure_b:start output=${b_json}"
run_eval "${b_json}" "${resident_json}" "${fixed_budget_bytes}" "B"
log_stage "measure_b:done output=${b_json}"

log_stage "measure_c:start output=${c_json}"
run_eval "${c_json}" "${resident_json}" "${fixed_budget_bytes}" "C"
log_stage "measure_c:done output=${c_json}"

python - "${b_json}" "${c_json}" "${OUT_ROOT}/summary_bc.json" <<'PY'
import json
import sys
from pathlib import Path

b_path, c_path, out_path = sys.argv[1:4]
with open(b_path) as fb, open(c_path) as fc:
    b = json.load(fb)
    c = json.load(fc)

def pick(payload, key):
    return payload.get("runtime_profile", {}).get(key)

summary = {
    "B_tps": b["generated_tokens_per_sec"],
    "C_tps": c["generated_tokens_per_sec"],
    "B_to_C_gain_percent": (c["generated_tokens_per_sec"] / b["generated_tokens_per_sec"] - 1.0) * 100.0,
    "shared_budget_bytes": b["sparse_budget_bytes"],
    "resident_count": c["resident_count"],
    "B_breakdown": {
        "tail_group_begin_wall_time_sec": pick(b, "tail_group_begin_wall_time_sec"),
        "tail_group_tensor_begin_service_wall_time_sec": pick(b, "tail_group_tensor_begin_service_wall_time_sec"),
        "modulelist_demand_compute_wall_time_sec": pick(b, "modulelist_demand_compute_wall_time_sec"),
        "modulelist_resident_compute_wall_time_sec": pick(b, "modulelist_resident_compute_wall_time_sec"),
        "modulelist_resident_gather_wall_time_sec": pick(b, "modulelist_resident_gather_wall_time_sec"),
        "modulelist_resident_merge_wall_time_sec": pick(b, "modulelist_resident_merge_wall_time_sec"),
        "module_begin_wall_time_sec": pick(b, "module_begin_wall_time_sec"),
    },
    "C_breakdown": {
        "tail_group_begin_wall_time_sec": pick(c, "tail_group_begin_wall_time_sec"),
        "tail_group_tensor_begin_service_wall_time_sec": pick(c, "tail_group_tensor_begin_service_wall_time_sec"),
        "modulelist_demand_compute_wall_time_sec": pick(c, "modulelist_demand_compute_wall_time_sec"),
        "modulelist_resident_compute_wall_time_sec": pick(c, "modulelist_resident_compute_wall_time_sec"),
        "modulelist_resident_gather_wall_time_sec": pick(c, "modulelist_resident_gather_wall_time_sec"),
        "modulelist_resident_merge_wall_time_sec": pick(c, "modulelist_resident_merge_wall_time_sec"),
        "module_begin_wall_time_sec": pick(c, "module_begin_wall_time_sec"),
    },
}

Path(out_path).write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
