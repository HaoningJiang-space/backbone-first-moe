#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/ziheng/backbone-first-moe_fresh_e2e}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/qwen_modulelist_lane_repeats}"
NUM_REPEATS="${NUM_REPEATS:-3}"

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

cd "${REPO_ROOT}"
source /home/ziheng/miniconda3/bin/activate mxmoe

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-/data/ziheng/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data/ziheng/torch_ext_qwen_lane_repeats}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

mkdir -p "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}" "${OUT_ROOT}"

run_eval() {
  local output_path="$1"
  local resident_file="$2"
  python -m finemoe.entrypoints.backbone_runtime_eval \
    --model-path "${MODEL_PATH}" \
    --offload-path "${OFFLOAD_PATH}" \
    --prompt-file "${PROMPT_FILE}" \
    --output "${output_path}" \
    --device-memory-ratio "${DEVICE_MEMORY_RATIO}" \
    --prefetch-distance 0 \
    --store-prefix "" \
    --resident-expert-ids-file "${resident_file}" \
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

for run_idx in $(seq 1 "${NUM_REPEATS}"); do
  run_dir="${OUT_ROOT}/run${run_idx}"
  res_dir="${run_dir}/residents"
  mkdir -p "${run_dir}" "${res_dir}"

  a_json="${run_dir}/qwen_A_mem0p10_lane_long.json"
  c_json="${run_dir}/qwen_C_mem0p10_lane_long.json"
  res_json="${res_dir}/qwen_lane_mem0p10.json"

  rm -f "${a_json}" "${c_json}" "${res_json}"

  run_eval "${a_json}" ""

  sparse_budget_bytes="$(
    python - "${a_json}" <<'PY'
import json
import sys
with open(sys.argv[1], "r") as fh:
    print(int(json.load(fh)["sparse_budget_bytes"]))
PY
  )"

  python -m experiments.simulation.select_adaptive_resident_set \
    --state-file "${STATE_FILE}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${res_dir}" \
    --output-prefix qwen_lane \
    --memory-ratios "${DEVICE_MEMORY_RATIO}" \
    --selection-method frontier_prefix \
    --profile-fraction 0.2 \
    --prefetch-windows 0 \
    --sparse-budget-bytes "${sparse_budget_bytes}"

  resident_capacity="$(
    python - "${res_json}" <<'PY'
import json
import sys
with open(sys.argv[1], "r") as fh:
    payload = json.load(fh)
print(int(payload.get("selected_resident_capacity", payload.get("resident_capacity", 0))))
PY
  )"

  if [ "${resident_capacity}" -eq 0 ]; then
    python - "${a_json}" "${c_json}" "${res_json}" <<'PY'
import json
import sys
a_path, c_path, resident_path = sys.argv[1:4]
with open(a_path, "r") as fh:
    payload = json.load(fh)
payload["resident_expert_ids_file"] = resident_path
payload["selection_degenerate_to_baseline"] = True
with open(c_path, "w") as fh:
    json.dump(payload, fh, indent=2)
PY
  else
    run_eval "${c_json}" "${res_json}"
  fi

  python - "${a_json}" "${c_json}" "${run_idx}" <<'PY'
import json
import sys
a_path, c_path, run_idx = sys.argv[1:4]
with open(a_path, "r") as fa, open(c_path, "r") as fc:
    a_payload = json.load(fa)
    c_payload = json.load(fc)
gain = (c_payload["generated_tokens_per_sec"] / a_payload["generated_tokens_per_sec"] - 1.0) * 100.0
print(
    f"run{run_idx}: "
    f"A={a_payload['generated_tokens_per_sec']:.6f} "
    f"C={c_payload['generated_tokens_per_sec']:.6f} "
    f"gain={gain:.2f}%"
)
PY
done
