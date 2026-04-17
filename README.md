# Backbone-First MoE Serving

Resident backbone extraction and demand-only tail for MoE expert offloading.

This project builds on top of [FineMoE](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26) and adds a runtime-native resident backbone path. The current stable line is the `v0.3.0-stable` branch.

## What This Repository Claims

The serving path that is currently validated is:
- resident backbone
- demand-only tail fallback
- no speculative prefetch on the critical path

Runtime support that is validated today:
- `Qwen1.5-MoE-A2.7B-Chat`
- `OLMoE-1B-7B-0924`

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

## Validated Real-Hardware Results

Qwen1.5-MoE-A2.7B-Chat, `batch=8`, same GPU (`cuda:0`), `prefetch_distance=0`, 16 prompts, 64 new tokens.

| Config | mem=0.07 | mem=0.10 | Notes |
|---|---:|---:|---|
| A demand-only | 2.9266 | 3.0946 | No resident backbone |
| C backbone-only | 3.2331 | 3.6162 | Burst-aware frontier selector |

Backbone-only improves generation throughput by:
- `+10.5%` at `mem=0.07`
- `+16.9%` at `mem=0.10`

These numbers were validated on real hardware after fixing sparse-node default-device propagation.

OLMoE-1B-7B-0924, same GPU (`cuda:0`), `prefetch_distance=0`, 2 prompts, 8 new tokens.

| Config | mem=0.07 | mem=0.10 | Notes |
|---|---:|---:|---|
| A demand-only | 0.3067 | 0.3157 | No resident backbone |
| C backbone-only | 2.5799 | 1.7778 | Resident set from offline adaptive selector |

Backbone-only improves generation throughput by:
- `+741.2%` at `mem=0.07`
- `+463.2%` at `mem=0.10`

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

Packed-MoE architectures (`Mixtral`, `DeepSeek-V2`, `DeepSeek-V3`) are runtime-enabled on the `multi-model-runtime` branch and currently validated via tiny end-to-end smokes. They should enter the formal runtime table only if the applicability diagnostics indicate a compact backbone under the target budget.

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
- `Qwen1.5-MoE-A2.7B-Chat`
- `OLMoE-1B-7B-0924`

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

## Base Runtime Modifications vs FineMoE

Key runtime changes include:
- sparse-node default-device propagation
- resident pinning support in C++ topology/prefetch runtime
- resident-aware runtime entrypoints

## License

Apache-2.0
