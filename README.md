# Backbone-First MoE Serving

**Resident backbone extraction and demand-only tail for MoE expert offloading.**

## Key Finding

MoE serving decomposes into a **stable resident backbone** and a **demand-only tail**.
The backbone is extracted automatically via throughput-sweep selection.
Speculative prefetch is unnecessary and harmful in the current runtime.

| Config | mem=0.07 | mem=0.10 | Description |
|---|---:|---:|---|
| A demand-only | 2.356 | 2.499 | LRU cache baseline |
| B LRU+prefetch | 1.691 | 1.702 | Speculative prefetch (harmful) |
| **C backbone-only** | **2.643** | **3.585** | **Our method** |

Backbone-first achieves **+43.5%** throughput over demand-only at mem=0.10, with zero manual ratio tuning.

## Quick Start (Simulation Only)

```bash
pip install -e .

# Run backbone generalization CV
python experiments/simulation/backbone_generalization.py \
    --state-file data/your_trace.pkl \
    --memory-ratios 0.07,0.10

# Run throughput-sweep backbone extraction
python experiments/simulation/select_adaptive_resident_set.py \
    --state-file data/your_trace.pkl \
    --memory-ratios 0.07,0.10

# Export resident set for runtime
python experiments/simulation/export_resident_set.py \
    --state-file data/your_trace.pkl \
    --device-memory-ratio 0.10 \
    --output results/resident_set.json
```

No GPU needed for simulation experiments.

## Full Reproduction (with FineMoE Runtime)

Real-hardware evaluation requires the FineMoE runtime with our patches.

```bash
# 1. Install FineMoE and apply patches
# See patches/README.md for instructions

# 2. Run real-hardware evaluation
python experiments/runtime/runtime_eval.py \
    --model-path /path/to/Qwen1.5-MoE-A2.7B-Chat \
    --offload-path /path/to/offloads \
    --resident-expert-ids-file results/resident_set.json \
    --device-memory-ratio 0.10 \
    --prefetch-distance 0 \
    --output results/eval_result.json
```

## Project Structure

```
backbone_moe/          # Core library (no FineMoE dependency)
  simulator.py         # Event-driven MoE offloading simulator
  evaluation.py        # Backbone extraction and evaluation
  metrics.py           # Statistical utilities
  workload.py          # Trace loading and fold splitting

experiments/
  simulation/          # Simulator-based experiments (standalone)
  runtime/             # Real-hardware experiments (requires FineMoE)
  plotting/            # Visualization scripts

patches/               # FineMoE runtime patches for native backbone support
tests/                 # Unit tests
docs/                  # Documentation
```

## Method

1. **Expert ranking**: Rank experts by profiling-prefix frequency
2. **Knee detection**: Find the structural backbone core from the cumulative utility curve
3. **Throughput sweep**: Search from knee to cache capacity for the throughput-optimal resident size
4. **Pin and serve**: Pin the selected backbone on GPU, serve tail via demand-only loading

## Citation

```bibtex
@inproceedings{backbone-first-moe,
    title={Backbone-First MoE Serving},
    year={2026},
}
```

## License

Apache-2.0
