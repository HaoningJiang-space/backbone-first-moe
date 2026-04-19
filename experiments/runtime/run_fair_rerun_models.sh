#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/home/ziheng/miniconda3/bin/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mxmoe}"

PROMPT_FILE="${PROMPT_FILE:-/data/ziheng/FineMoE-EuroSys26/demo/states/lmsys-chat-1m~eval_prompts.json}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/runtime_rerun_all}"
RES_DIR="${RES_DIR:-${RESULT_ROOT}/residents}"
RUN_DIR="${RUN_DIR:-${RESULT_ROOT}/runtime}"

TMP_BASE="${TMP_BASE:-/data/ziheng/tmp}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data/ziheng/torch_ext_fair_rerun}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
DEEPSEEK_BACKEND_ROOT="${DEEPSEEK_BACKEND_ROOT:-/data/ziheng/pydeps/transformers_5_5_4}"

QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat}"
QWEN_OFFLOAD_PATH="${QWEN_OFFLOAD_PATH:-/data/finemoe_offloads/Qwen1.5-MoE-A2.7B-Chat}"
QWEN_STATE_FILE="${QWEN_STATE_FILE:-/data/ziheng/FineMoE-EuroSys26/demo/states/Qwen1.5-MoE-A2.7B-Chat~lmsys-chat-1m~64.pkl}"

OLMOE_MODEL_PATH="${OLMOE_MODEL_PATH:-/data/ziheng/models/OLMoE-1B-7B-0924}"
OLMOE_OFFLOAD_PATH="${OLMOE_OFFLOAD_PATH:-/data/finemoe_offloads/OLMoE-1B-7B-0924}"
OLMOE_STATE_FILE="${OLMOE_STATE_FILE:-${REPO_ROOT}/states/OLMoE-1B-7B-0924~lmsys-chat-1m-fair~64.pkl}"
OLMOE_FAIR_MEMORY_RATIOS="${OLMOE_FAIR_MEMORY_RATIOS:-0.045 0.05}"
OLMOE_OUTPUT_PREFIX="${OLMOE_OUTPUT_PREFIX:-olmoe_current_refair}"

DEEPSEEK_MODEL_PATH="${DEEPSEEK_MODEL_PATH:-/data/ziheng/Efficient_AI/models/DeepSeek-V2-Lite}"
DEEPSEEK_OFFLOAD_PATH="${DEEPSEEK_OFFLOAD_PATH:-/data/finemoe_offloads/DeepSeek-V2-Lite}"
DEEPSEEK_STATE_FILE="${DEEPSEEK_STATE_FILE:-/data/ziheng/backbone-first-moe/results/deepseek_trace/DeepSeek-V2-Lite~lmsys-chat-1m~64.pkl}"

mkdir -p "${TMP_BASE}" "${TORCH_EXTENSIONS_DIR}" "${RES_DIR}" "${RUN_DIR}"

cd "${REPO_ROOT}"
source "${CONDA_ACTIVATE}" "${CONDA_ENV_NAME}"

export TMPDIR="${TMP_BASE}"
export TMP="${TMP_BASE}"
export TEMP="${TMP_BASE}"
export TORCH_EXTENSIONS_DIR
export CUDA_VISIBLE_DEVICES

REPO_PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
DEEPSEEK_PYTHONPATH="${DEEPSEEK_BACKEND_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

run_python() {
  local py_path="$1"
  shift
  PYTHONPATH="${py_path}" python "$@"
}

run_module() {
  local py_path="$1"
  shift
  PYTHONPATH="${py_path}" python -m "$@"
}

run_select() {
  local py_path="$1"
  shift
  run_python "${py_path}" experiments/simulation/select_adaptive_resident_set.py "$@"
}

read_sparse_budget_bytes() {
  local json_path="$1"
  python - "$json_path" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r") as fh:
    payload = json.load(fh)
value = payload.get("sparse_budget_bytes")
if value in (None, "", 0):
    raise SystemExit(f"Missing sparse_budget_bytes in {path}")
print(int(value))
PY
}

run_eval() {
  local py_path="$1"
  local model_path="$2"
  local offload_path="$3"
  local output="$4"
  shift 4
  run_module "${py_path}" finemoe.entrypoints.backbone_runtime_eval \
    --model-path "${model_path}" \
    --offload-path "${offload_path}" \
    --prompt-file "${PROMPT_FILE}" \
    --output "${output}" \
    "$@"
}

# Qwen: use repo-local backend. The newer DeepSeek backend removes GenerationMixin
# from the legacy Qwen model class, so we keep Qwen on the environment that
# previously produced the validated runtime results.
for mem in 0.07 0.10; do
  tag="${mem/./p}"
  qwen_a_output="${RUN_DIR}/qwen_A_mem${tag}_current.json"
  qwen_c_output="${RUN_DIR}/qwen_C_mem${tag}_current.json"
  run_eval "${REPO_PYTHONPATH}" "${QWEN_MODEL_PATH}" "${QWEN_OFFLOAD_PATH}" "${qwen_a_output}" \
    --device-memory-ratio "${mem}" --prefetch-distance 0 --store-prefix "" --resident-expert-ids-file "" \
    --device cuda:0 --eval-mode offline --batch-size 8 --num-prompts 16 --seed 42 \
    --max-length 256 --max-new-tokens 64 --min-new-tokens 1 --store-capacity 1000 --tag runtime_eval
  qwen_budget_bytes="$(read_sparse_budget_bytes "${qwen_a_output}")"
  run_select "${REPO_PYTHONPATH}" \
    --state-file "${QWEN_STATE_FILE}" \
    --model-path "${QWEN_MODEL_PATH}" \
    --output-dir "${RES_DIR}" \
    --output-prefix qwen_current \
    --memory-ratios "${mem}" \
    --selection-method frontier_prefix \
    --profile-fraction 0.2 \
    --prefetch-windows 0 \
    --sparse-budget-bytes "${qwen_budget_bytes}"
  run_eval "${REPO_PYTHONPATH}" "${QWEN_MODEL_PATH}" "${QWEN_OFFLOAD_PATH}" "${qwen_c_output}" \
    --device-memory-ratio "${mem}" --prefetch-distance 0 --store-prefix "" --resident-expert-ids-file "${RES_DIR}/qwen_current_mem${tag}.json" \
    --device cuda:0 --eval-mode offline --batch-size 8 --num-prompts 16 --seed 42 \
    --max-length 256 --max-new-tokens 64 --min-new-tokens 1 --store-capacity 1000 --tag runtime_eval
done

# OLMoE: same backend treatment as Qwen; this keeps modulelist runtime behavior
# on the path that produced the earlier validated results.
#
# The older 0.012/0.014/0.016 "fair" sweep assumed a much smaller expert
# footprint than the runtime actually pinned. With the corrected 12MB/expert
# accounting, those points collapse to resident=0. We therefore rerun OLMoE on
# a stricter but still non-zero-resident range.
for mem in ${OLMOE_FAIR_MEMORY_RATIOS}; do
  tag="${mem/./p}"
  olmoe_a_output="${RUN_DIR}/olmoe_A_mem${tag}_current.json"
  olmoe_c_output="${RUN_DIR}/olmoe_C_mem${tag}_current.json"
  run_eval "${REPO_PYTHONPATH}" "${OLMOE_MODEL_PATH}" "${OLMOE_OFFLOAD_PATH}" "${olmoe_a_output}" \
    --device-memory-ratio "${mem}" --prefetch-distance 0 --store-prefix "" --resident-expert-ids-file "" \
    --device cuda:0 --eval-mode offline --batch-size 2 --num-prompts 2 --seed 42 \
    --max-length 256 --max-new-tokens 8 --min-new-tokens 1 --store-capacity 1000 --tag runtime_eval
  olmoe_budget_bytes="$(read_sparse_budget_bytes "${olmoe_a_output}")"
  run_select "${REPO_PYTHONPATH}" \
    --state-file "${OLMOE_STATE_FILE}" \
    --model-path "${OLMOE_MODEL_PATH}" \
    --output-dir "${RES_DIR}" \
    --output-prefix "${OLMOE_OUTPUT_PREFIX}" \
    --memory-ratios "${mem}" \
    --selection-method frontier_prefix \
    --profile-fraction 0.2 \
    --prefetch-windows 0 \
    --sparse-budget-bytes "${olmoe_budget_bytes}"
  run_eval "${REPO_PYTHONPATH}" "${OLMOE_MODEL_PATH}" "${OLMOE_OFFLOAD_PATH}" "${olmoe_c_output}" \
    --device-memory-ratio "${mem}" --prefetch-distance 0 --store-prefix "" --resident-expert-ids-file "${RES_DIR}/${OLMOE_OUTPUT_PREFIX}_mem${tag}.json" \
    --device cuda:0 --eval-mode offline --batch-size 2 --num-prompts 2 --seed 42 \
    --max-length 256 --max-new-tokens 8 --min-new-tokens 1 --store-capacity 1000 --tag runtime_eval
done

# DeepSeek: requires the newer backend that provides transformers.models.deepseek_v2/v3.
for mem in 0.07 0.10; do
  tag="${mem/./p}"
  deepseek_a_output="${RUN_DIR}/deepseek_A_mem${tag}_current.json"
  deepseek_c_output="${RUN_DIR}/deepseek_C_mem${tag}_current.json"
  run_eval "${DEEPSEEK_PYTHONPATH}" "${DEEPSEEK_MODEL_PATH}" "${DEEPSEEK_OFFLOAD_PATH}" "${deepseek_a_output}" \
    --device-memory-ratio "${mem}" --prefetch-distance 0 --store-prefix "" --resident-expert-ids-file "" \
    --device cuda:0 --eval-mode offline --batch-size 2 --num-prompts 2 --seed 42 \
    --max-length 256 --max-new-tokens 8 --min-new-tokens 1 --store-capacity 1000 --tag runtime_eval
  deepseek_budget_bytes="$(read_sparse_budget_bytes "${deepseek_a_output}")"
  run_select "${DEEPSEEK_PYTHONPATH}" \
    --state-file "${DEEPSEEK_STATE_FILE}" \
    --model-path "${DEEPSEEK_MODEL_PATH}" \
    --output-dir "${RES_DIR}" \
    --output-prefix deepseek_current \
    --memory-ratios "${mem}" \
    --selection-method frontier_prefix \
    --profile-fraction 0.2 \
    --prefetch-windows 0 \
    --sparse-budget-bytes "${deepseek_budget_bytes}"
  run_eval "${DEEPSEEK_PYTHONPATH}" "${DEEPSEEK_MODEL_PATH}" "${DEEPSEEK_OFFLOAD_PATH}" "${deepseek_c_output}" \
    --device-memory-ratio "${mem}" --prefetch-distance 0 --store-prefix "" --resident-expert-ids-file "${RES_DIR}/deepseek_current_mem${tag}.json" \
    --device cuda:0 --eval-mode offline --batch-size 2 --num-prompts 2 --seed 42 \
    --max-length 256 --max-new-tokens 8 --min-new-tokens 1 --store-capacity 1000 --tag runtime_eval
done
