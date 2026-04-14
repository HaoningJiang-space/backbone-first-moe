# Backbone-First MoE Serving

**Resident backbone extraction and demand-only tail for MoE expert offloading.**

This project is built on top of [FineMoE](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26) with modifications for runtime-native backbone pinning (C++ `is_resident` flag, `PinResidentNodes`, eviction exemption).

## Key Finding

MoE serving decomposes into a **stable resident backbone** and a **demand-only tail**.
The backbone is extracted automatically via throughput-sweep selection.
No manual ratio tuning is required.

### Reproducible Real-Hardware Results (batch=8, Qwen1.5-MoE-A2.7B-Chat, A800 80GB)

| Config | mem=0.07 | mem=0.10 | Description |
|---|---:|---:|---|
| A demand-only | 2.356 | 2.499 | LRU cache baseline |
| B LRU+prefetch | 1.691 | 1.702 | Per-sequence prefetch (harmful) |
| **C_optimal backbone-only** | **2.643** | **3.537** | **Throughput-sweep backbone, no prefetch** |

C_optimal uses automatically selected ratio (0.96 at mem=0.10, 0.83 at mem=0.07) with `prefetch_distance=0`.
Backbone-first achieves **+41.6%** throughput over demand-only at mem=0.10 (reproducibly verified).

### Oracle Analysis (simulator)

| Config | mem=0.07 | mem=0.10 |
|---|---:|---:|
| C backbone-only | 177.1 | 343.4 |
| D backbone + oracle prefetch | 209.6 | 498.8 |

Oracle prefetch adds +18-45% on top of backbone. Current causal prefetch adds nothing (CPU overhead cancels benefit). Batch-aware lightweight prefetch is future work.

## Reproducing Results

### Step 1: Backbone Extraction (simulator, no GPU)

```bash
pip install -e .

# Throughput-sweep backbone extraction
# Selects optimal resident ratio automatically for each memory budget
# 自动为每个内存预算选择最优 resident ratio
python experiments/simulation/select_adaptive_resident_set.py \
    --state-file data/your_trace.pkl \
    --output-dir results/adaptive_sweep \
    --output-prefix adaptive_sweep \
    --memory-ratios 0.05,0.07,0.10 \
    --selection-method capacity_search \
    --prefetch-windows 0

# Example output:
#   mem=0.05: optimal_k=149 (ratio=0.63)
#   mem=0.07: optimal_k=276 (ratio=0.83)
#   mem=0.10: optimal_k=456 (ratio=0.96)
```

### Step 2: Real-Hardware Evaluation (requires GPU)

```bash
# Best config: C_optimal (backbone-only, throughput-sweep ratio, no prefetch)
# 最优配置：纯 backbone，自动选 ratio，无 prefetch
CUDA_VISIBLE_DEVICES=0 python finemoe/entrypoints/backbone_runtime_eval.py \
    --model-path /path/to/Qwen1.5-MoE-A2.7B-Chat \
    --prompt-file /path/to/eval_prompts.json \
    --offload-path /path/to/offloads \
    --store-prefix /path/to/offline_store \
    --device-memory-ratio 0.10 \
    --prefetch-distance 0 \
    --resident-expert-ids-file results/adaptive_sweep/adaptive_sweep_mem0p10.json \
    --eval-mode offline \
    --batch-size 8 \
    --num-prompts 16 \
    --max-new-tokens 64 \
    --tag C_optimal_mem0p10 \
    --output results/C_optimal_mem0p10.json
```

Key parameters:
- `--prefetch-distance 0`: No speculative prefetch (pure backbone + demand)
- `--resident-expert-ids-file`: Use the JSON from Step 1 (throughput-sweep optimal)
- `--batch-size 8`: Batched inference

### Step 3: Baseline Comparison

```bash
# A: demand-only baseline (no backbone, no prefetch)
CUDA_VISIBLE_DEVICES=0 python finemoe/entrypoints/backbone_runtime_eval.py \
    --model-path /path/to/Qwen1.5-MoE-A2.7B-Chat \
    --prompt-file /path/to/eval_prompts.json \
    --offload-path /path/to/offloads \
    --store-prefix /path/to/offline_store \
    --device-memory-ratio 0.10 \
    --prefetch-distance 0 \
    --resident-expert-ids-file "" \
    --eval-mode offline \
    --batch-size 8 \
    --num-prompts 16 \
    --max-new-tokens 64 \
    --tag A_demand_mem0p10 \
    --output results/A_demand_mem0p10.json
```

### Other Simulation Experiments

```bash
# Backbone generalization CV (Section 3.1)
python experiments/simulation/backbone_generalization.py \
    --state-file data/your_trace.pkl --memory-ratios 0.07,0.10

# Oracle vs backbone analysis (Section 3.2)
python experiments/simulation/oracle_vs_backbone.py \
    --state-file data/your_trace.pkl --memory-ratios 0.07,0.10

# Export resident set (alternative to throughput sweep)
python experiments/simulation/export_resident_set.py \
    --state-file data/your_trace.pkl --device-memory-ratio 0.10 \
    --output results/resident_set.json
```

## Project Structure

```
backbone_moe/          # Core analysis library (standalone, no GPU needed)
  simulator.py         # Event-driven MoE offloading simulator
  evaluation.py        # Backbone extraction, knee detection, throughput sweep
  metrics.py           # Statistical utilities
  workload.py          # Trace loading and fold splitting

finemoe/               # Modified FineMoE runtime (based on FineMoE-EuroSys26)
  runtime/             # model_offload.py with pin_resident_experts()
  memory/              # expert_prefetcher.py with resident filtering + batch prefetch
  entrypoints/         # backbone_runtime_eval.py, backbone_section5.py
  backbone/            # runtime_eval.py, section5.py

core/                  # Modified C++ engine
  model/               # model_topology.h with is_resident flag
  prefetch/            # PinResidentNodes(), eviction exemption
  python/              # pin_resident_nodes, enqueue_prefetch_batch bindings

experiments/           # Experiment scripts
  simulation/          # Simulator experiments (standalone)
  runtime/             # Real-hardware experiments (requires GPU)
  plotting/            # Visualization

results/               # Archived experiment JSONs (48 files)
docs/                  # Roadmap and documentation
tests/                 # Unit tests (16 passing)
```

## Method

1. **Expert ranking**: Rank experts by profiling-prefix frequency
2. **Knee detection**: Find structural backbone core from cumulative utility curve
3. **Throughput sweep**: Search from knee to cache capacity for throughput-optimal resident size
4. **Pin and serve**: Pin selected backbone on GPU via C++ `is_resident` flag, serve tail via demand-only loading

The optimal ratio adapts automatically to memory budget:
- mem=0.05 → ratio=0.63 (more demand buffer needed)
- mem=0.07 → ratio=0.83
- mem=0.10 → ratio=0.96 (pin almost everything)

## C++ Modifications (vs base FineMoE)

| File | Change |
|---|---|
| `core/model/model_topology.h` | `bool is_resident` flag on Node |
| `core/prefetch/task_scheduler.cpp` | Skip `is_resident` nodes in eviction |
| `core/prefetch/archer_prefetch_handle.cpp` | `PinResidentNodes()`, `EnqueuePrefetchBatch()` |
| `core/python/py_archer_prefetch.cpp` | `pin_resident_nodes`, `enqueue_prefetch_batch` bindings |

## Acknowledgments

This project is built on [FineMoE](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26) by the IntelliSys Lab / TorchMoE team.

## Citation

```bibtex
@inproceedings{backbone-first-moe,
    title={Backbone-First MoE Serving},
    year={2026},
}
```

## License

Apache-2.0
