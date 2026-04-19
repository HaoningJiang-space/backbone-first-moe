#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/home/ziheng/miniconda3/bin/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mxmoe}"

PROMPT_FILE="${PROMPT_FILE:-/data/ziheng/FineMoE-EuroSys26/demo/states/lmsys-chat-1m~eval_prompts.json}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/olmoe_refair_runtime}"
RES_DIR="${RES_DIR:-${REPO_ROOT}/results/olmoe_refair}"
RUN_DIR="${RUN_DIR:-${RESULT_ROOT}}"

TMP_BASE="${TMP_BASE:-/data/ziheng/tmp}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data/ziheng/torch_ext_olmoe_refair}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OLMOE_MODEL_PATH="${OLMOE_MODEL_PATH:-/data/ziheng/models/OLMoE-1B-7B-0924}"
OLMOE_OFFLOAD_PATH="${OLMOE_OFFLOAD_PATH:-/data/finemoe_offloads/OLMoE-1B-7B-0924}"
OLMOE_STATE_FILE="${OLMOE_STATE_FILE:-${REPO_ROOT}/states/OLMoE-1B-7B-0924~lmsys-chat-1m-fair~64.pkl}"
OLMOE_FAIR_MEMORY_RATIOS="${OLMOE_FAIR_MEMORY_RATIOS:-0.045 0.05}"
OLMOE_OUTPUT_PREFIX="${OLMOE_OUTPUT_PREFIX:-olmoe_refair}"

mkdir -p "${TMP_BASE}" "${TORCH_EXTENSIONS_DIR}" "${RES_DIR}" "${RUN_DIR}"

cd "${REPO_ROOT}"
source "${CONDA_ACTIVATE}" "${CONDA_ENV_NAME}"

export TMPDIR="${TMP_BASE}"
export TMP="${TMP_BASE}"
export TEMP="${TMP_BASE}"
export TORCH_EXTENSIONS_DIR
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

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

for mem in ${OLMOE_FAIR_MEMORY_RATIOS}; do
  tag="${mem/./p}"
  a_output="${RUN_DIR}/olmoe_A_mem${tag}_refair.json"
  c_output="${RUN_DIR}/olmoe_C_mem${tag}_refair.json"
  python -m finemoe.entrypoints.backbone_runtime_eval \
    --model-path "${OLMOE_MODEL_PATH}" \
    --offload-path "${OLMOE_OFFLOAD_PATH}" \
    --prompt-file "${PROMPT_FILE}" \
    --output "${a_output}" \
    --device-memory-ratio "${mem}" \
    --prefetch-distance 0 \
    --store-prefix "" \
    --resident-expert-ids-file "" \
    --device cuda:0 \
    --eval-mode offline \
    --batch-size 2 \
    --num-prompts 2 \
    --seed 42 \
    --max-length 256 \
    --max-new-tokens 8 \
    --min-new-tokens 1 \
    --store-capacity 1000 \
    --tag runtime_eval

  budget_bytes="$(read_sparse_budget_bytes "${a_output}")"
  python experiments/simulation/select_adaptive_resident_set.py \
    --state-file "${OLMOE_STATE_FILE}" \
    --model-path "${OLMOE_MODEL_PATH}" \
    --output-dir "${RES_DIR}" \
    --output-prefix "${OLMOE_OUTPUT_PREFIX}" \
    --memory-ratios "${mem}" \
    --selection-method frontier_prefix \
    --profile-fraction 0.2 \
    --prefetch-windows 0 \
    --sparse-budget-bytes "${budget_bytes}"

  python -m finemoe.entrypoints.backbone_runtime_eval \
    --model-path "${OLMOE_MODEL_PATH}" \
    --offload-path "${OLMOE_OFFLOAD_PATH}" \
    --prompt-file "${PROMPT_FILE}" \
    --output "${c_output}" \
    --device-memory-ratio "${mem}" \
    --prefetch-distance 0 \
    --store-prefix "" \
    --resident-expert-ids-file "${RES_DIR}/${OLMOE_OUTPUT_PREFIX}_mem${tag}.json" \
    --device cuda:0 \
    --eval-mode offline \
    --batch-size 2 \
    --num-prompts 2 \
    --seed 42 \
    --max-length 256 \
    --max-new-tokens 8 \
    --min-new-tokens 1 \
    --store-capacity 1000 \
    --tag runtime_eval
done
