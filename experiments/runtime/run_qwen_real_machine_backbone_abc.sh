#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/ziheng/backbone-first-moe_fresh_e2e}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/qwen_real_machine_backbone_abc}"
NUM_MEASURED_PAIRS="${NUM_MEASURED_PAIRS:-3}"

if [ "${1:-}" != "" ]; then
  OUT_ROOT="$1"
fi

MODEL_PATH="${MODEL_PATH:-/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat}"
OFFLOAD_PATH="${OFFLOAD_PATH:-/data/finemoe_offloads/Qwen1.5-MoE-A2.7B-Chat}"
PROMPT_FILE="${PROMPT_FILE:-/data/ziheng/backbone-first-moe_fresh_e2e/results/qwen_real_machine_backbone_validation_v1/shard_1/prompts.json}"
STATE_FILE="${STATE_FILE:-}"
RESIDENT_FILE="${RESIDENT_FILE:-}"

DEVICE_MEMORY_RATIO="${DEVICE_MEMORY_RATIO:-0.10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-1}"
STORE_CAPACITY="${STORE_CAPACITY:-1000}"
SEED="${SEED:-42}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"

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
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
fi
cuda_tag="$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | tr -c '[:alnum:]' '_')"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data/ziheng/torch_ext_qwen_real_machine_abc_${cuda_tag}}"

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
    --tag "runtime_real_machine_${mode}" \
    "${extra_flags[@]}"
}

read_budget() {
  python - "$1" <<'PY'
import json, sys
with open(sys.argv[1], "r") as fh:
    print(int(json.load(fh)["sparse_budget_bytes"]))
PY
}

resolve_state_file() {
  python - "${PROMPT_FILE}" "${STATE_FILE}" <<'PY'
from pathlib import Path
import sys

prompt_file = Path(sys.argv[1]).resolve()
explicit = sys.argv[2].strip()
if explicit:
    path = Path(explicit).resolve()
    if not path.exists():
        raise SystemExit(f"STATE_FILE does not exist: {path}")
    print(path)
    raise SystemExit(0)

candidates = []
states_dir = prompt_file.parent / "states"
if states_dir.is_dir():
    candidates.extend(sorted(states_dir.glob("*.pkl")))

if prompt_file.name == "prompts.json":
    shard_dir = prompt_file.parent
    sibling_states = shard_dir / "states"
    if sibling_states.is_dir():
        candidates.extend(sorted(sibling_states.glob("*.pkl")))

dedup = []
seen = set()
for candidate in candidates:
    key = str(candidate.resolve())
    if key in seen:
        continue
    seen.add(key)
    dedup.append(candidate.resolve())

if len(dedup) != 1:
    pretty = ", ".join(str(p) for p in dedup) if dedup else "<none>"
    raise SystemExit(
        "Unable to resolve unique STATE_FILE from PROMPT_FILE. "
        f"Candidates: {pretty}. Set STATE_FILE explicitly."
    )
print(dedup[0])
PY
}

resolve_selected_resident_file() {
  python - "${plan_dir}" "${DEVICE_MEMORY_RATIO}" <<'PY'
from pathlib import Path
import sys

plan_dir = Path(sys.argv[1])
mem_ratio = float(sys.argv[2])
text = f"{mem_ratio:.6f}"
whole, frac = text.split(".")
frac = frac.rstrip("0")
if len(frac) < 2:
    frac = frac.ljust(2, "0")
path = plan_dir / f"resident_mem{whole}p{frac}.json"
print(path)
PY
}

read_selected_resident_capacity() {
  python - "${1}" <<'PY'
import json, sys
with open(sys.argv[1], "r") as fh:
    payload = json.load(fh)
print(int(payload.get("selected_resident_capacity", payload.get("resident_capacity", 0))))
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
warm_dir="${OUT_ROOT}/warmup"
mkdir -p "${plan_dir}" "${warm_dir}"

plan_a_json="${plan_dir}/qwen_A_plan.json"
rm -f "${plan_a_json}"
log_stage "plan_a:start output=${plan_a_json}"
run_eval "${plan_a_json}" "" 0 "A"
log_stage "plan_a:done output=${plan_a_json}"
fixed_budget_bytes="$(read_budget "${plan_a_json}")"

resident_source="explicit_file"
resident_json="${RESIDENT_FILE}"
if [ -z "${resident_json}" ]; then
  resident_source="runtime_budget_reselected"
  state_file="$(resolve_state_file)"
  resident_json="$(resolve_selected_resident_file)"
  rm -f "${resident_json}"
  log_stage "resident_select:start state_file=${state_file} budget_bytes=${fixed_budget_bytes}"
  python -m experiments.simulation.select_adaptive_resident_set \
    --state-file "${state_file}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${plan_dir}" \
    --output-prefix resident \
    --memory-ratios "${DEVICE_MEMORY_RATIO}" \
    --selection-method frontier_prefix \
    --profile-fraction 0.2 \
    --prefetch-windows 0 \
    --sparse-budget-bytes "${fixed_budget_bytes}"
  log_stage "resident_select:done output=${resident_json}"
fi

resident_capacity="$(read_selected_resident_capacity "${resident_json}")"

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
    output_json="${pair_dir}/qwen_${mode}_heldout.json"
    rm -f "${output_json}"
    log_stage "pair${pair_idx}:${mode}:start output=${output_json}"
    if [ "${mode}" = "A" ] || [ "${resident_capacity}" -eq 0 ]; then
      run_eval "${output_json}" "" "${fixed_budget_bytes}" "${mode}"
    else
      run_eval "${output_json}" "${resident_json}" "${fixed_budget_bytes}" "${mode}"
    fi
    log_stage "pair${pair_idx}:${mode}:done output=${output_json}"
  done

  python - "${pair_dir}/qwen_A_heldout.json" "${pair_dir}/qwen_B_heldout.json" "${pair_dir}/qwen_C_heldout.json" "${pair_idx}" <<'PY'
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

python - "${OUT_ROOT}" "${NUM_MEASURED_PAIRS}" "${resident_json}" "${resident_source}" <<'PY'
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
num_pairs = int(sys.argv[2])
resident_file = sys.argv[3]
resident_source = sys.argv[4]

with open(resident_file) as fr:
    resident_payload = json.load(fr)

rows = []
def profile_subset(payload):
    rp = payload.get("runtime_profile", {})
    return {
        "resident_expert_blocks": int(rp.get("modulelist_resident_expert_blocks", 0)),
        "demand_expert_blocks": int(rp.get("modulelist_demand_expert_blocks", 0)),
        "resident_token_assignments": int(rp.get("modulelist_resident_token_assignments", 0)),
        "demand_token_assignments": int(rp.get("modulelist_demand_token_assignments", 0)),
        "resident_compute_wall_time_sec": float(rp.get("modulelist_resident_compute_wall_time_sec", 0.0)),
        "demand_compute_wall_time_sec": float(rp.get("modulelist_demand_compute_wall_time_sec", 0.0)),
        "resident_gather_wall_time_sec": float(rp.get("modulelist_resident_gather_wall_time_sec", 0.0)),
        "resident_merge_wall_time_sec": float(rp.get("modulelist_resident_merge_wall_time_sec", 0.0)),
        "tail_group_begin_wall_time_sec": float(rp.get("tail_group_begin_wall_time_sec", 0.0)),
    }

for pair_idx in range(1, num_pairs + 1):
    pair_dir = out_root / f"pair{pair_idx}"
    with open(pair_dir / "qwen_A_heldout.json") as fa, \
         open(pair_dir / "qwen_B_heldout.json") as fb, \
         open(pair_dir / "qwen_C_heldout.json") as fc:
        a = json.load(fa)
        b = json.load(fb)
        c = json.load(fc)
    rows.append({
        "pair": pair_idx,
        "A_tps": a["generated_tokens_per_sec"],
        "B_tps": b["generated_tokens_per_sec"],
        "C_tps": c["generated_tokens_per_sec"],
        "A_profile": profile_subset(a),
        "B_profile": profile_subset(b),
        "C_profile": profile_subset(c),
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
    "resident_file": resident_file,
    "resident_source": resident_source,
    "resident_selection_budget_bytes": resident_payload.get("selection_budget_bytes"),
    "resident_selection_budget_source": resident_payload.get("selection_budget_source"),
    "resident_capacity": int(resident_payload.get("selected_resident_capacity", resident_payload.get("resident_capacity", 0))),
    "pairs": rows,
    "A_to_B": summarize("A_tps", "B_tps"),
    "B_to_C": summarize("B_tps", "C_tps"),
    "A_to_C": summarize("A_tps", "C_tps"),
}

summary_path = out_root / "summary_abc.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
