#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/ziheng/backbone-first-moe_fresh_e2e}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/qwen_modulelist_lane_regime_compare}"

SEMI_COLD_REPEATS="${SEMI_COLD_REPEATS:-3}"
STEADY_STATE_PAIRS="${STEADY_STATE_PAIRS:-5}"

mkdir -p "${OUT_ROOT}"

echo "[1/2] semi-cold / mixed-state repeats"
OUT_ROOT="${OUT_ROOT}/semi_cold" \
NUM_REPEATS="${SEMI_COLD_REPEATS}" \
bash "${REPO_ROOT}/experiments/runtime/run_qwen_modulelist_lane_repeats.sh"

echo "[2/2] warm steady-state alternating pairs"
OUT_ROOT="${OUT_ROOT}/warm_steady_state" \
NUM_MEASURED_PAIRS="${STEADY_STATE_PAIRS}" \
bash "${REPO_ROOT}/experiments/runtime/run_qwen_modulelist_lane_steady_state.sh"

echo "done:"
echo "  semi-cold:     ${OUT_ROOT}/semi_cold"
echo "  warm_steady:   ${OUT_ROOT}/warm_steady_state"
