#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/ziheng/backbone-first-moe_fresh_e2e}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/demo/states/benchmarks}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/data/ziheng/hf_datasets}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DATASETS="${DATASETS:-gsm8k,mmlu,bbh,longbench}"
SAMPLE_SIZE="${SAMPLE_SIZE:-128}"
MIXED_PER_SOURCE="${MIXED_PER_SOURCE:-32}"
SEED="${SEED:-42}"

cd "${REPO_ROOT}"
source /home/ziheng/miniconda3/bin/activate mxmoe

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data/ziheng/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_DIR}}"
export HF_ENDPOINT
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

mkdir -p "${OUT_ROOT}" "${HF_CACHE_DIR}" "${HF_HOME}"

python demo/download_serving_benchmark_datasets.py \
  --output-dir "${OUT_ROOT}" \
  --cache-dir "${HF_CACHE_DIR}" \
  --datasets "${DATASETS}" \
  --sample-size "${SAMPLE_SIZE}" \
  --mixed-per-source "${MIXED_PER_SOURCE}" \
  --seed "${SEED}"
