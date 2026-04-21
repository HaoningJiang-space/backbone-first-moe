#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/ziheng/backbone-first-moe_fresh_e2e}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/qwen_modulelist_lane_abc}"
NUM_MEASURED_PAIRS="${NUM_MEASURED_PAIRS:-3}"

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

if [ $((NUM_MEASURED_PAIRS % 3)) -ne 0 ]; then
  echo "NUM_MEASURED_PAIRS must be a multiple of 3 for balanced A/B/C order, got ${NUM_MEASURED_PAIRS}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
source /home/ziheng/miniconda3/bin/activate mxmoe

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-/data/ziheng/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data/ziheng/torch_ext_qwen_lane_abc}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
fi

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
    --tag "runtime_eval_${mode}" \
    "${extra_flags[@]}"
}

read_budget() {
  python - "$1" <<'PY'
import json, sys
with open(sys.argv[1], "r") as fh:
    print(int(json.load(fh)["sparse_budget_bytes"]))
PY
}

pair_order() {
  local pair_idx="$1"
  case $(((pair_idx - 1) % 3)) in
    0) printf 'A B C\n' ;;
    1) printf 'B C A\n' ;;
    2) printf 'C A B\n' ;;
  esac
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
log_stage "warmup_a:start"
run_eval "${warm_dir}/qwen_A_warmup.json" "" "${fixed_budget_bytes}" "A"
log_stage "warmup_a:done"

if [ "${resident_capacity}" -eq 0 ]; then
  cp "${warm_dir}/qwen_A_warmup.json" "${warm_dir}/qwen_B_warmup.json"
  cp "${warm_dir}/qwen_A_warmup.json" "${warm_dir}/qwen_C_warmup.json"
  log_stage "warmup_b_c:skipped resident_capacity=0"
else
  log_stage "warmup_b:start"
  run_eval "${warm_dir}/qwen_B_warmup.json" "${resident_json}" "${fixed_budget_bytes}" "B"
  log_stage "warmup_b:done"
  log_stage "warmup_c:start"
  run_eval "${warm_dir}/qwen_C_warmup.json" "${resident_json}" "${fixed_budget_bytes}" "C"
  log_stage "warmup_c:done"
fi

for pair_idx in $(seq 1 "${NUM_MEASURED_PAIRS}"); do
  pair_dir="${OUT_ROOT}/pair${pair_idx}"
  mkdir -p "${pair_dir}"
  read -r first second third <<<"$(pair_order "${pair_idx}")"
  for mode in "${first}" "${second}" "${third}"; do
    output_json="${pair_dir}/qwen_${mode}_mem0p10_lane_long.json"
    rm -f "${output_json}"
    log_stage "pair${pair_idx}:${mode}:start output=${output_json}"
    if [ "${mode}" = "A" ] || [ "${resident_capacity}" -eq 0 ]; then
      run_eval "${output_json}" "" "${fixed_budget_bytes}" "${mode}"
    else
      run_eval "${output_json}" "${resident_json}" "${fixed_budget_bytes}" "${mode}"
    fi
    log_stage "pair${pair_idx}:${mode}:done output=${output_json}"
  done

  python - "${pair_dir}/qwen_A_mem0p10_lane_long.json" "${pair_dir}/qwen_B_mem0p10_lane_long.json" "${pair_dir}/qwen_C_mem0p10_lane_long.json" "${pair_idx}" <<'PY'
import json, sys
a_path, b_path, c_path, pair_idx = sys.argv[1:5]
with open(a_path) as fa, open(b_path) as fb, open(c_path) as fc:
    a = json.load(fa)
    b = json.load(fb)
    c = json.load(fc)
def gain(x, y):
    return (y["generated_tokens_per_sec"] / x["generated_tokens_per_sec"] - 1.0) * 100.0
print(
    f"pair{pair_idx}: "
    f"A={a['generated_tokens_per_sec']:.6f} "
    f"B={b['generated_tokens_per_sec']:.6f} "
    f"C={c['generated_tokens_per_sec']:.6f} "
    f"A->B={gain(a, b):.2f}% "
    f"B->C={gain(b, c):.2f}% "
    f"A->C={gain(a, c):.2f}%"
)
PY
done

python - "${OUT_ROOT}" "${NUM_MEASURED_PAIRS}" <<'PY'
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
num_pairs = int(sys.argv[2])

rows = []
for pair_idx in range(1, num_pairs + 1):
    pair_dir = out_root / f"pair{pair_idx}"
    with open(pair_dir / "qwen_A_mem0p10_lane_long.json") as fa, \
         open(pair_dir / "qwen_B_mem0p10_lane_long.json") as fb, \
         open(pair_dir / "qwen_C_mem0p10_lane_long.json") as fc:
        a = json.load(fa)
        b = json.load(fb)
        c = json.load(fc)
    rows.append({
        "pair": pair_idx,
        "A_tps": a["generated_tokens_per_sec"],
        "B_tps": b["generated_tokens_per_sec"],
        "C_tps": c["generated_tokens_per_sec"],
    })

def summarize(src, dst):
    vals = [(row[dst] / row[src] - 1.0) * 100.0 for row in rows]
    return {
        "mean_gain_percent": sum(vals) / len(vals),
        "min_gain_percent": min(vals),
        "max_gain_percent": max(vals),
    }

summary = {
    "num_pairs": num_pairs,
    "pairs": rows,
    "A_to_B": summarize("A_tps", "B_tps"),
    "B_to_C": summarize("B_tps", "C_tps"),
    "A_to_C": summarize("A_tps", "C_tps"),
}

summary_path = out_root / "summary_abc.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
