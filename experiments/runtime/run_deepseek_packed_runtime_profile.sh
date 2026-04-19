#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/ziheng/backbone-first-moe_fresh_e2e}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/deepseek_packed_runtime_profile}"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/home/ziheng/miniconda3/bin/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mxmoe}"
DEEPSEEK_BACKEND_ROOT="${DEEPSEEK_BACKEND_ROOT:-/data/ziheng/pydeps/transformers_5_5_4}"

MODEL_PATH="${MODEL_PATH:-/data/ziheng/Efficient_AI/models/DeepSeek-V2-Lite}"
OFFLOAD_PATH="${OFFLOAD_PATH:-/data/finemoe_offloads/DeepSeek-V2-Lite}"
PROMPT_FILE="${PROMPT_FILE:-/data/ziheng/FineMoE-EuroSys26/demo/states/lmsys-chat-1m~eval_prompts.json}"
STATE_FILE="${STATE_FILE:-/data/ziheng/backbone-first-moe/results/deepseek_trace/DeepSeek-V2-Lite~lmsys-chat-1m~64.pkl}"

MEMORY_RATIO="${MEMORY_RATIO:-0.10}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_PROMPTS="${NUM_PROMPTS:-2}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-1}"
STORE_CAPACITY="${STORE_CAPACITY:-1000}"
SEED="${SEED:-42}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

RUN_C="${RUN_C:-1}"
RESIDENT_FILE="${RESIDENT_FILE:-}"
TMP_BASE="${TMP_BASE:-/data/ziheng/tmp}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data/ziheng/torch_ext_deepseek_packed_batch}"

cd "${REPO_ROOT}"
source "${CONDA_ACTIVATE}" "${CONDA_ENV_NAME}"

export PYTHONPATH="${DEEPSEEK_BACKEND_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
export TMPDIR="${TMP_BASE}"
export TMP="${TMP_BASE}"
export TEMP="${TMP_BASE}"
export TORCH_EXTENSIONS_DIR
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

mkdir -p "${TMP_BASE}" "${TORCH_EXTENSIONS_DIR}" "${OUT_ROOT}" "${OUT_ROOT}/residents"

run_eval() {
  local output_path="$1"
  local resident_file="$2"
  local budget_override="${3:-0}"
  python -m finemoe.entrypoints.backbone_runtime_eval \
    --model-path "${MODEL_PATH}" \
    --offload-path "${OFFLOAD_PATH}" \
    --prompt-file "${PROMPT_FILE}" \
    --output "${output_path}" \
    --device-memory-ratio "${MEMORY_RATIO}" \
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
    --tag runtime_eval
}

read_sparse_budget_bytes() {
  local json_path="$1"
  python - "$json_path" <<'PY'
import json, sys
with open(sys.argv[1], "r") as fh:
    print(int(json.load(fh)["sparse_budget_bytes"]))
PY
}

read_selected_resident_capacity() {
  local json_path="$1"
  python - "$json_path" <<'PY'
import json, sys
with open(sys.argv[1], "r") as fh:
    payload = json.load(fh)
print(int(payload.get("selected_resident_capacity", payload.get("resident_capacity", 0))))
PY
}

materialize_degenerate_c_result() {
  local a_json="$1"
  local c_json="$2"
  local resident_file="$3"
  python - "$a_json" "$c_json" "$resident_file" <<'PY'
import json, sys
a_path, c_path, resident_file = sys.argv[1:4]
with open(a_path, "r") as fh:
    payload = json.load(fh)
payload["resident_expert_ids_file"] = resident_file
payload["selection_degenerate_to_baseline"] = True
with open(c_path, "w") as fh:
    json.dump(payload, fh, indent=2)
PY
}

tag="${MEMORY_RATIO/./p}"
a_json="${OUT_ROOT}/deepseek_A_mem${tag}_batch.json"
c_json="${OUT_ROOT}/deepseek_C_mem${tag}_batch.json"
resident_json="${OUT_ROOT}/residents/deepseek_batch_mem${tag}.json"

rm -f "${a_json}" "${c_json}" "${resident_json}"

run_eval "${a_json}" "" 0

if [ "${RUN_C}" = "0" ]; then
  echo "A-only run complete: ${a_json}"
  exit 0
fi

fixed_budget_bytes="$(read_sparse_budget_bytes "${a_json}")"

if [ -z "${RESIDENT_FILE}" ]; then
  python -m experiments.simulation.select_adaptive_resident_set \
    --state-file "${STATE_FILE}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUT_ROOT}/residents" \
    --output-prefix deepseek_batch \
    --memory-ratios "${MEMORY_RATIO}" \
    --selection-method frontier_prefix \
    --profile-fraction 0.2 \
    --prefetch-windows 0 \
    --sparse-budget-bytes "${fixed_budget_bytes}"
else
  cp "${RESIDENT_FILE}" "${resident_json}"
fi

resident_capacity="$(read_selected_resident_capacity "${resident_json}")"
if [ "${resident_capacity}" -eq 0 ]; then
  materialize_degenerate_c_result "${a_json}" "${c_json}" "${resident_json}"
else
  run_eval "${c_json}" "${resident_json}" "${fixed_budget_bytes}"
fi

python - "${a_json}" "${c_json}" <<'PY'
import json, sys
with open(sys.argv[1]) as fa, open(sys.argv[2]) as fc:
    a=json.load(fa); c=json.load(fc)
gain=(c["generated_tokens_per_sec"]/a["generated_tokens_per_sec"]-1.0)*100.0
print(f"A={a['generated_tokens_per_sec']:.6f} C={c['generated_tokens_per_sec']:.6f} gain={gain:.2f}%")
PY
