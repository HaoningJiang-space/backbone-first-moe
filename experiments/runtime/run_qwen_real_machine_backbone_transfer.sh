#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/ziheng/backbone-first-moe_fresh_e2e}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/qwen_real_machine_backbone_transfer}"

MODEL_PATH="${MODEL_PATH:-/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat}"
OFFLOAD_PATH="${OFFLOAD_PATH:-/data/finemoe_offloads/Qwen1.5-MoE-A2.7B-Chat}"
PROMPT_FILE="${PROMPT_FILE:-/data/ziheng/backbone-first-moe_fresh_e2e/results/qwen_real_machine_backbone_validation_v1/shard_1/prompts.json}"
NATIVE_RESIDENT_FILE="${NATIVE_RESIDENT_FILE:-/data/ziheng/backbone-first-moe_fresh_e2e/results/qwen_real_machine_backbone_validation_v1/shard_1/resident/resident_mem0p10.json}"
TRANSFER_RESIDENT_FILE="${TRANSFER_RESIDENT_FILE:-/data/ziheng/backbone-first-moe_fresh_e2e/results/qwen_real_machine_backbone_validation_v1/shard_0/resident/resident_mem0p10.json}"

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
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data/ziheng/torch_ext_qwen_real_machine_transfer}"
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
  local tag="$4"
  local extra_flags=()
  if [ -n "${resident_file}" ]; then
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
    --tag "${tag}" \
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
warm_dir="${OUT_ROOT}/warmup"
mkdir -p "${plan_dir}" "${warm_dir}"

plan_a_json="${plan_dir}/qwen_A_plan.json"
rm -f "${plan_a_json}"
log_stage "plan_a:start output=${plan_a_json}"
run_eval "${plan_a_json}" "" 0 "runtime_transfer_A_plan"
log_stage "plan_a:done output=${plan_a_json}"
fixed_budget_bytes="$(read_budget "${plan_a_json}")"

log_stage "warmup_a:start"
run_eval "${warm_dir}/qwen_A_warmup.json" "" "${fixed_budget_bytes}" "runtime_transfer_A_warmup"
log_stage "warmup_a:done"

log_stage "warmup_b_native:start"
run_eval "${warm_dir}/qwen_B_native_warmup.json" "${NATIVE_RESIDENT_FILE}" "${fixed_budget_bytes}" "runtime_transfer_B_native_warmup"
log_stage "warmup_b_native:done"

log_stage "warmup_b_transfer:start"
run_eval "${warm_dir}/qwen_B_transfer_warmup.json" "${TRANSFER_RESIDENT_FILE}" "${fixed_budget_bytes}" "runtime_transfer_B_transfer_warmup"
log_stage "warmup_b_transfer:done"

a_json="${OUT_ROOT}/qwen_A_heldout.json"
b_native_json="${OUT_ROOT}/qwen_B_native_heldout.json"
b_transfer_json="${OUT_ROOT}/qwen_B_transfer_heldout.json"
rm -f "${a_json}" "${b_native_json}" "${b_transfer_json}"

log_stage "measure_a:start output=${a_json}"
run_eval "${a_json}" "" "${fixed_budget_bytes}" "runtime_transfer_A"
log_stage "measure_a:done output=${a_json}"

log_stage "measure_b_native:start output=${b_native_json}"
run_eval "${b_native_json}" "${NATIVE_RESIDENT_FILE}" "${fixed_budget_bytes}" "runtime_transfer_B_native"
log_stage "measure_b_native:done output=${b_native_json}"

log_stage "measure_b_transfer:start output=${b_transfer_json}"
run_eval "${b_transfer_json}" "${TRANSFER_RESIDENT_FILE}" "${fixed_budget_bytes}" "runtime_transfer_B_transfer"
log_stage "measure_b_transfer:done output=${b_transfer_json}"

python - "${a_json}" "${b_native_json}" "${b_transfer_json}" "${OUT_ROOT}/summary_transfer.json" <<'PY'
import json
import sys
from pathlib import Path

a_path, bn_path, bt_path, out_path = sys.argv[1:5]
with open(a_path) as fa, open(bn_path) as fbn, open(bt_path) as fbt:
    a = json.load(fa)
    bn = json.load(fbn)
    bt = json.load(fbt)

def tps(x):
    return x["generated_tokens_per_sec"]

native_gain = (tps(bn) / tps(a) - 1.0) * 100.0
transfer_gain = (tps(bt) / tps(a) - 1.0) * 100.0
retained = transfer_gain / native_gain if native_gain > 0 else None

summary = {
    "A_tps": tps(a),
    "B_native_tps": tps(bn),
    "B_transfer_tps": tps(bt),
    "A_to_B_native_gain_percent": native_gain,
    "A_to_B_transfer_gain_percent": transfer_gain,
    "transfer_retained_gain_fraction": retained,
    "shared_budget_bytes": a["sparse_budget_bytes"],
    "native_resident_count": bn.get("resident_count"),
    "transfer_resident_count": bt.get("resident_count"),
    "native_resident_file": bn.get("resident_expert_ids_file"),
    "transfer_resident_file": bt.get("resident_expert_ids_file"),
  }

Path(out_path).write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
