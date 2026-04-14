# Backbone-First MoE Serving

**Resident backbone extraction and demand-only tail for MoE expert offloading.**

This project is built on top of [FineMoE](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26) with modifications for runtime-native backbone pinning (C++ `is_resident` flag, `PinResidentNodes`, eviction exemption).

## Key Finding

MoE serving decomposes into a **stable resident backbone** and a **demand-only tail**.
The backbone is extracted automatically via throughput-sweep selection.
Speculative prefetch is unnecessary and harmful in the current runtime.

### Real-Hardware Results (batch=8, Qwen1.5-MoE-A2.7B-Chat, A800 80GB)

| Config | mem=0.07 | mem=0.10 | Description |
|---|---:|---:|---|
| A demand-only | 2.356 | 2.499 | LRU cache baseline |
| B LRU+prefetch | 1.691 | 1.702 | Speculative prefetch (harmful) |
| **C backbone-only** | **2.643** | **3.585** | **Our method (throughput-sweep)** |

Backbone-first achieves **+43.5%** throughput over demand-only at mem=0.10.

### Oracle Analysis (simulator)

| Config | mem=0.07 | mem=0.10 |
|---|---:|---:|
| C backbone-only | 177.1 | 343.4 |
| D backbone + oracle prefetch | 209.6 | 498.8 |

Oracle prefetch adds +18-45% on top of backbone. Current causal prefetch adds nothing (overhead cancels benefit). Batch-aware lightweight prefetch is future work.

## Project Structure

```
backbone_moe/          # Core analysis library (standalone, no GPU needed)
  simulator.py         # Event-driven MoE offloading simulator
  evaluation.py        # Backbone extraction, knee detection, throughput sweep
  metrics.py           # Statistical utilities
  workload.py          # Trace loading and fold splitting

finemoe/               # Modified FineMoE runtime (based on FineMoE-EuroSys26)
  runtime/             # model_offload.py with pin_resident_experts()
  memory/              # expert_prefetcher.py with resident filtering
  entrypoints/         # backbone_runtime_eval.py, backbone_section5.py
  backbone/            # runtime_eval.py, section5.py

core/                  # Modified C++ engine
  model/               # model_topology.h with is_resident flag
  prefetch/            # PinResidentNodes(), eviction exemption
  python/              # pin_resident_nodes Python binding

experiments/           # Experiment scripts
  simulation/          # Simulator experiments (standalone)
  runtime/             # Real-hardware experiments (requires GPU)
  plotting/            # Visualization

demo/                  # FineMoE demo scripts and configs
tests/                 # Unit tests
docs/                  # Roadmap and documentation
results/               # Result summaries
```

## Quick Start

### Simulation Only (no GPU needed)

```bash
pip install -e .

# Backbone generalization CV
python experiments/simulation/backbone_generalization.py \
    --state-file data/your_trace.pkl --memory-ratios 0.07,0.10

# Throughput-sweep backbone extraction
python experiments/simulation/select_adaptive_resident_set.py \
    --state-file data/your_trace.pkl --memory-ratios 0.07,0.10

# Oracle vs backbone comparison
python experiments/simulation/oracle_vs_backbone.py \
    --state-file data/your_trace.pkl --memory-ratios 0.07,0.10
```

### Real-Hardware Evaluation (requires GPU)

```bash
# 1. Install with runtime dependencies
pip install -e ".[runtime]"

# 2. Prepare offload store
python demo/prepare_custom_data.py --prompt-file data/prompts.json --dataset-name your_dataset

# 3. Export resident set
python experiments/simulation/export_resident_set.py \
    --state-file data/your_trace.pkl --device-memory-ratio 0.10 \
    --output results/resident_set.json

# 4. Run backbone-only evaluation (C config)
CUDA_VISIBLE_DEVICES=0 python finemoe/entrypoints/backbone_runtime_eval.py \
    --model-path /path/to/Qwen1.5-MoE-A2.7B-Chat \
    --offload-path /path/to/offloads \
    --resident-expert-ids-file results/resident_set.json \
    --device-memory-ratio 0.10 --prefetch-distance 0 \
    --batch-size 8 --num-prompts 16 --max-new-tokens 64 \
    --output results/eval_result.json
```

## Method

1. **Expert ranking**: Rank experts by profiling-prefix frequency
2. **Knee detection**: Find structural backbone core from cumulative utility curve
3. **Throughput sweep**: Search from knee to cache capacity for throughput-optimal resident size
4. **Pin and serve**: Pin selected backbone on GPU via C++ `is_resident` flag, serve tail via demand-only loading

## C++ Modifications (vs base FineMoE)

| File | Change |
|---|---|
| `core/model/model_topology.h` | `bool is_resident` flag on Node |
| `core/prefetch/task_scheduler.cpp` | Skip `is_resident` nodes in eviction |
| `core/prefetch/archer_prefetch_handle.cpp` | `PinResidentNodes()` method |
| `core/python/py_archer_prefetch.cpp` | `pin_resident_nodes` Python binding |

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
