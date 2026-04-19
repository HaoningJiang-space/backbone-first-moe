# Backbone-First MoE Serving

Resident backbone extraction and demand-only tail for MoE expert offloading.

This project builds on top of [FineMoE](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26) and adds a runtime-native resident backbone path.

There are currently two supported tracks:
- `v0.3.1` tag / `v0.3.0-stable` branch: stable runtime for `Qwen1.5-MoE-A2.7B-Chat` and `OLMoE-1B-7B-0924`
- `multi-model-runtime` branch: adds packed-MoE runtime enablement for `DeepSeek-V2-Lite`, `Mixtral`, and `DeepSeek-V3`; this branch is validated by fresh-clone unit tests and targeted runtime probes, but it is still the active integration branch

## Which Branch Should You Use?

Use **`multi-model-runtime`** unless you explicitly need the frozen stable release.

- **Recommended default branch: `multi-model-runtime`**
  - this is the current active integration branch
  - this is the branch that should be used for ongoing development and new experiments
  - this is the only branch that currently contains the packed-MoE runtime path (`DeepSeek`, `Mixtral`, `DeepSeek-V3`)
- **Conservative frozen release: `v0.3.1` / `v0.3.0-stable`**
  - use this only if you want the narrowest validated scope
  - this line is intentionally limited to `Qwen + OLMoE`

In other words:

- if you are continuing the project, use `multi-model-runtime`
- if you are reproducing the older stable release only, use `v0.3.1`

Older historical branches should not be treated as the main working branch.

## Quick Start

Stable release:

```bash
git clone --branch v0.3.1 https://github.com/HaoningJiang-space/backbone-first-moe.git
cd backbone-first-moe
pip install -e ".[runtime]"
pytest -q
```

Packed-MoE integration branch:

```bash
git clone --branch multi-model-runtime https://github.com/HaoningJiang-space/backbone-first-moe.git
cd backbone-first-moe
pip install -e ".[runtime]"
pytest -q
```

DeepSeek note for `multi-model-runtime`:
- `DeepSeek-V2` / `DeepSeek-V3` are **not** fully vendored inside this repo.
- The local wrappers still require a transformers backend that provides:
  - `transformers.models.deepseek_v2`
  - `transformers.models.deepseek_v3`
- `pip install -e ".[runtime]"` on this branch now requires `transformers>=5.5.0`, which satisfies that condition on a normal fresh clone.
- If your environment is pinned to an older transformers build, `Qwen`, `OLMoE`, and `Mixtral` can still run, but `DeepSeek` will not.
- On a clean checkout on `10.16.52.172`, the normal `pip install -e ".[runtime]"` path is sufficient. The older fallback below is only needed if you intentionally keep an older shared environment and do not upgrade `transformers`:

```bash
export PYTHONPATH=/data/ziheng/pydeps/transformers_5_5_4:/data/ziheng/backbone-first-moe_lb:$PYTHONPATH
```

Fresh-clone validation that has already been exercised:
- `v0.3.1`: local fresh clone install + full unit test pass + `Qwen/OLMoE` runtime smoke on `10.16.52.172`
- `multi-model-runtime`: local fresh clone install + full unit test pass with `transformers>=5.5.0`; clean-checkout runtime smoke now runs on `10.16.52.172` for:
  - `Qwen A/C @ mem=0.07`
  - `OLMoE A/C @ mem=0.05`
  - `DeepSeek-V2-Lite A/C @ mem=0.07`

Fresh-clone smoke numbers from `/data/ziheng/backbone-first-moe_fresh_e2e/results/fresh_smoke` on `10.16.52.172`:

| Model | mem | A gen tok/s | C gen tok/s | Notes |
|---|---:|---:|---:|---|
| Qwen1.5-MoE-A2.7B-Chat | 0.07 | 0.4436 | 0.5960 | `resident_count=325`, `admitted_bytes=5.62 GB`, `budget_bytes=5.96 GB` |
| OLMoE-1B-7B-0924 | 0.05 | 0.1786 | 0.1911 | `resident_count=58`, `admitted_bytes=0.73 GB`, `budget_bytes=4.25 GB` |
| DeepSeek-V2-Lite | 0.07 | 0.0825 | 0.0828 | current selector yields `resident_count=0`, so `C≈A` |

## What This Repository Claims

The serving path that is currently validated is:
- resident backbone
- demand-only tail fallback
- no speculative prefetch on the critical path

Runtime support that is validated today depends on the branch:
- `v0.3.1` / `v0.3.0-stable`
  - `Qwen1.5-MoE-A2.7B-Chat`
  - `OLMoE-1B-7B-0924`
- `multi-model-runtime`
  - `Qwen1.5-MoE-A2.7B-Chat`
  - `OLMoE-1B-7B-0924`
  - `DeepSeek-V2-Lite`
  - `Mixtral` / `DeepSeek-V3` packed runtime tiny probes

The selector is not a ratio sweep. It ranks experts by profiling utility, then chooses the largest resident prefix that remains feasible under a burst-aware tail frontier constraint.

Formally, given resident prefix `R_k` and total cache budget `B`, the selector chooses the largest feasible `k` such that:

```text
k + F_H(k) <= B
```

where `F_H(k)` is the burst-aware residual demand frontier after pinning the top-`k` resident experts, computed directly from the routed trace.

This also means the project should not be read as "every MoE should benefit equally".
The right applicability questions are:
- does the model expose a compact resident core?
- does the available budget leave enough tail slack after pinning that core?

For a paper-facing runtime table that already resolves the `OLMoE` fairness issue, see:
- [results/runtime_formal/paper_runtime_table.md](results/runtime_formal/paper_runtime_table.md)

For a paper-facing note that explicitly separates `Qwen`'s idealized structural headroom from its current fair-runtime realization, see:
- [results/runtime_formal/qwen/qwen_realization_gap.md](results/runtime_formal/qwen/qwen_realization_gap.md)

Use the `warm steady-state` number in that note as the primary quantitative `Qwen` runtime claim.
The larger mixed/semi-cold gains in the same note should be treated as regime sensitivity, not as the default headline throughput result.

## Historical Fixed-Runtime Results

This section is kept as historical fixed-runtime reference material.

For the current paper-facing `Qwen` claim under fair budget and the newer two-lane modulelist runtime, use:
- [results/runtime_formal/qwen/qwen_realization_gap.md](results/runtime_formal/qwen/qwen_realization_gap.md)

Qwen1.5-MoE-A2.7B-Chat, historical fixed-runtime results, `batch=8`, same GPU (`cuda:0`), `prefetch_distance=0`, 16 prompts, 64 new tokens.

| Config | mem=0.07 | mem=0.10 | Notes |
|---|---:|---:|---|
| A demand-only | 2.9266 | 3.0946 | No resident backbone |
| C backbone-only | 3.2331 | 3.6162 | Burst-aware frontier selector |

Backbone-only improves generation throughput by:
- `+10.5%` at `mem=0.07`
- `+16.9%` at `mem=0.10`

These numbers were validated on real hardware after fixing sparse-node default-device propagation, but they should now be read as a historical unified/fixed-runtime reference rather than the current paper-facing `Qwen` truth.

OLMoE-1B-7B-0924, same GPU (`cuda:0`), `prefetch_distance=0`, 2 prompts, 8 new tokens.

| Config | mem=0.07 | mem=0.10 | Notes |
|---|---:|---:|---|
| A demand-only | 0.3067 | 0.3157 | No resident backbone |
| C backbone-only | 2.5799 | 1.7778 | Resident set from offline adaptive selector |

Backbone-only improves generation throughput by:
- `+741.2%` at `mem=0.07`
- `+463.2%` at `mem=0.10`

Fairness note:
- `OLMoE` experts are much smaller than `Qwen/DeepSeek`, so fixed `device_memory_ratio` places `OLMoE` in a near-full-fit regime.
- Keep the fixed-mem numbers for the system-budget view, but use the coverage-matched fair sweep below for cross-model comparison.

OLMoE-1B-7B-0924, coverage-matched fair sweep, same GPU (`cuda:0`), `prefetch_distance=0`, 2 prompts, 8 new tokens.

| Config | mem=0.012 | mem=0.014 | mem=0.016 | Notes |
|---|---:|---:|---:|---|
| A demand-only | 0.3332 | 0.3336 | 0.3339 | Fair low-budget baseline |
| C backbone-only | 0.4988 | 0.5469 | 0.6053 | Frontier-selected resident backbone |

Coverage-matched backbone-only improves generation throughput by:
- `+49.7%` at `mem=0.012`
- `+63.9%` at `mem=0.014`
- `+81.3%` at `mem=0.016`

DeepSeek-V2-Lite, same GPU (`cuda:0`), `prefetch_distance=0`, 2 prompts, 8 new tokens. These numbers are currently validated on `multi-model-runtime`.

| Config | mem=0.07 | mem=0.10 | Notes |
|---|---:|---:|---|
| A demand-only | 0.1597 | 0.1583 | No resident backbone |
| C backbone-only | 0.1782 | 0.1776 | Packed runtime, resident demand-only tail |

Backbone-only improves generation throughput by:
- `+11.6%` at `mem=0.07`
- `+12.2%` at `mem=0.10`

## Applicability and Boundary Cases

Positive runtime cases should satisfy both:
- **concentration**: a compact backbone core exists
- **budget sufficiency**: the residual tail frontier fits in the remaining slack

The repository now includes an explicit applicability diagnostic:

```bash
python experiments/simulation/analyze_applicability.py \
  --state-file data/your_trace.pkl \
  --output-dir results/applicability \
  --memory-ratios 0.07,0.10 \
  --resident-policy profile_freq \
  --resident-profile-ratio 0.2 \
  --frontier-percentile 1.0
```

This outputs, for each memory budget:
- stall/access top-k coverage
- knee/core fraction
- burst-aware frontier size
- slack utilization under the frontier-feasible resident prefix

Packed-MoE architectures (`Mixtral`, `DeepSeek-V2`, `DeepSeek-V3`) are runtime-enabled on the `multi-model-runtime` branch. `DeepSeek-V2-Lite` has full-model `A/C` probes; `Mixtral` and `DeepSeek-V3` are currently validated via tiny end-to-end probes and should enter the formal runtime table only if the applicability diagnostics indicate a compact backbone under the target budget.

Current examples from the applicability diagnostics:
- `Qwen1.5-MoE-A2.7B-Chat`: positive case
  - top-20% access coverage `~0.77`
  - burst-feasible resident ratio `~0.83-0.91`
- `DeepSeek-V2-Lite`: weak positive / boundary case under current budgets
  - lower concentration than `Qwen`
  - transferable hotspots exist
  - real-hardware gains are positive but small (`~11-12%`)
- `Mixtral`: applicability / boundary case
  - simulation shows retained gain `~0.96-0.97` at `mem=0.07/0.10`
  - current tiny packed-runtime probe is essentially at parity (`A 7.73 -> C 7.74 gen tok/s`, `+0.1%`)
  - full-model runtime evidence is still missing, so it stays out of the main runtime table

## Current Throughput Priorities

The main throughput question is no longer whether backbone pinning helps. That has already been established for:
- `Qwen`
- `OLMoE`
- `DeepSeek-V2-Lite`
- `Mixtral` tiny packed-runtime probes

The current engineering question is how to make the gain larger and more stable.

Priority order:

1. **Finish full-model packed runtime assets**
   - `Mixtral` is currently limited by missing full-model assets, not by a known runtime correctness bug.
   - Until full-model checkpoints and offloads are in place, the packed path cannot produce a formal throughput table.

2. **Reduce packed-runtime control overhead**
   - The next likely bottleneck is not backbone selection, but packed expert indirection:
     - synthetic slice lookup
     - packed-to-expert dispatch bookkeeping
     - hook overhead on packed MoE blocks
   - This should be treated as a runtime optimization problem, not a selector-tuning problem.

3. **Prebuild and reuse offload artifacts**
   - Throughput measurements should not pay repeated setup costs:
     - JIT compilation
     - offload index construction
     - cold-start initialization
   - The highest-signal comparisons are warm-path `A` vs `C` runs on the same GPU.

4. **Keep the method fixed while optimizing the runtime**
   - The project should not go back to ratio sweeps or prefetch tuning.
   - The current method remains:
     - utility-ranked resident backbone
     - burst-aware frontier-feasible prefix
     - demand-only tail fallback

In short:

> The best way to improve throughput now is to make the runtime cheaper, not to make the selector more heuristic.

## Method Summary

1. Rank experts by profiling-prefix utility.
2. Build batch-step expert union demand from the routed trace.
3. Infer a burst horizon `H` from transfer time vs per-layer compute time.
4. Compute the residual demand frontier `F_H(k)` under the resident prefix.
5. Choose the largest feasible resident prefix.
6. Pin that prefix on GPU and serve the tail through demand-only loading.

Current implementation details:
- utility ranking: `profile_freq`
- feasibility model: burst-aware frontier
- runtime mode: backbone-only (`prefetch_distance=0`)

Legacy `capacity_search` / `ratio_grid` code paths remain in the repo for comparison and ablation, but they are not the main method.

## Reproducing the Selector

```bash
pip install -e .

python experiments/simulation/select_adaptive_resident_set.py \
  --state-file data/your_trace.pkl \
  --output-dir results/frontier_qwen \
  --output-prefix qwen_frontier \
  --memory-ratios 0.07,0.10 \
  --selection-method frontier_prefix \
  --frontier-percentile 1.0 \
  --profile-fraction 0.2 \
  --resident-policy profile_freq \
  --resident-profile-ratio 0.2 \
  --expert-size-mb 17.2 \
  --h2d-bandwidth-gbps 16.0 \
  --gpu-compute-time-ms 2.0
```

This produces one resident JSON per memory budget plus a summary JSON.

## Reproducing Real-Hardware Runtime Evaluation

Supported runtime models:
- `v0.3.1`
  - `Qwen1.5-MoE-A2.7B-Chat`
  - `OLMoE-1B-7B-0924`
- `multi-model-runtime`
  - `Qwen1.5-MoE-A2.7B-Chat`
  - `OLMoE-1B-7B-0924`
  - `DeepSeek-V2-Lite`

For normal fresh-clone use on `multi-model-runtime`, `pip install -e ".[runtime]"` should install a DeepSeek-capable transformers backend directly.

On `10.16.52.172`, the validated environment is still:

```bash
source /home/ziheng/miniconda3/bin/activate mxmoe
export PYTHONPATH=/data/ziheng/pydeps/transformers_5_5_4:/data/ziheng/backbone-first-moe_lb:$PYTHONPATH
export TMPDIR=/data/ziheng/tmp
export TMP=/data/ziheng/tmp
export TEMP=/data/ziheng/tmp
export TORCH_EXTENSIONS_DIR=/data/ziheng/torch_ext_deepseek_v2
export CUDA_VISIBLE_DEVICES=0
```

Backbone-only (`C`) example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m finemoe.entrypoints.backbone_runtime_eval \
  --model-path /path/to/Qwen1.5-MoE-A2.7B-Chat \
  --prompt-file /path/to/eval_prompts.json \
  --output results/qwen_C_mem0p10.json \
  --offload-path /path/to/offloads \
  --device-memory-ratio 0.10 \
  --prefetch-distance 0 \
  --store-prefix "" \
  --resident-expert-ids-file results/frontier_qwen/qwen_frontier_mem0p10.json \
  --device cuda:0 \
  --batch-size 8 \
  --num-prompts 16 \
  --seed 42 \
  --max-length 256 \
  --max-new-tokens 64 \
  --tag qwen_C_mem0p10
```

Demand-only baseline (`A`) example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m finemoe.entrypoints.backbone_runtime_eval \
  --model-path /path/to/Qwen1.5-MoE-A2.7B-Chat \
  --prompt-file /path/to/eval_prompts.json \
  --output results/qwen_A_mem0p10.json \
  --offload-path /path/to/offloads \
  --device-memory-ratio 0.10 \
  --prefetch-distance 0 \
  --store-prefix "" \
  --resident-expert-ids-file "" \
  --device cuda:0 \
  --batch-size 8 \
  --num-prompts 16 \
  --seed 42 \
  --max-length 256 \
  --max-new-tokens 64 \
  --tag qwen_A_mem0p10
```

## Project Structure

```text
backbone_moe/
  evaluation.py        # standalone selector / analysis helpers
  metrics.py
  workload.py

finemoe/
  backbone/
  entrypoints/
  memory/
  runtime/

core/
  model/
  prefetch/
  python/

experiments/
  simulation/
  runtime/
  plotting/

tests/
```

## Minimum Validation Checklist

Before tagging a release, this repository should satisfy all of the following:
- selector unit tests pass
- evaluation helper tests pass
- fresh clone supports `pip install -e .`
- selector CLI runs from a clean checkout
- `Qwen` runtime smoke runs from a clean checkout
- `OLMoE` runtime smoke runs from a clean checkout

For `multi-model-runtime`, the additional expectation is:
- packed-runtime unit tests pass from a fresh clone
- `Qwen`, `OLMoE`, and `DeepSeek-V2-Lite` runtime smoke runs from a clean checkout on `10.16.52.172`

## Base Runtime Modifications vs FineMoE

Key runtime changes include:
- sparse-node default-device propagation
- resident pinning support in C++ topology/prefetch runtime
- resident-aware runtime entrypoints

## License

Apache-2.0
