#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/ziheng/backbone-first-moe_fresh_e2e}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/qwen_modulelist_lane_ceiling_mini}"

MODEL_PATH="${MODEL_PATH:-/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat}"
OFFLOAD_PATH="${OFFLOAD_PATH:-/data/finemoe_offloads/Qwen1.5-MoE-A2.7B-Chat}"
PROMPT_FILE="${PROMPT_FILE:-/data/ziheng/FineMoE-EuroSys26/demo/states/lmsys-chat-1m~eval_prompts.json}"
STATE_FILE="${STATE_FILE:-/data/ziheng/FineMoE-EuroSys26/demo/states/Qwen1.5-MoE-A2.7B-Chat~lmsys-chat-1m~64.pkl}"

NUM_PROMPTS="${NUM_PROMPTS:-8}"
DEVICE_MEMORY_RATIO="${DEVICE_MEMORY_RATIO:-0.10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
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
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data/ziheng/torch_ext_qwen_lane_ceiling_singleproc}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

mkdir -p "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}" "${OUT_ROOT}"

python experiments/runtime/run_qwen_modulelist_lane_ceiling_singleproc.py \
  --model-path "${MODEL_PATH}" \
  --offload-path "${OFFLOAD_PATH}" \
  --prompt-file "${PROMPT_FILE}" \
  --state-file "${STATE_FILE}" \
  --output-root "${OUT_ROOT}" \
  --device-memory-ratio "${DEVICE_MEMORY_RATIO}" \
  --device cuda:0 \
  --batch-size "${BATCH_SIZE}" \
  --num-prompts "${NUM_PROMPTS}" \
  --max-length "${MAX_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --min-new-tokens "${MIN_NEW_TOKENS}" \
  --store-capacity "${STORE_CAPACITY}" \
  --seed "${SEED}"
