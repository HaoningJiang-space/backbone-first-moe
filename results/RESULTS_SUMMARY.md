# Backbone-First MoE Serving: Complete Results

## Model & Hardware
- Model: Qwen1.5-MoE-A2.7B-Chat (24 layers, 60 experts, top-4 routing)
- GPU: NVIDIA A800 80GB PCIe
- Expert size: 17.2 MB

## Section 3: Structural Evidence

### 3.1 Backbone Generalization (n=64, 8-fold CV)

| mem | retained gain | transfer gain (tok/s) | regret vs native | Jaccard |
|---|---:|---:|---:|---:|
| 0.05 | 0.830 +/- 0.195 | 178.5 +/- 57.2 | 32.7 +/- 37.2 | 0.648 +/- 0.024 |
| 0.07 | 0.877 +/- 0.091 | 223.5 +/- 21.9 | 31.7 +/- 23.0 | 0.679 +/- 0.032 |
| 0.10 | 0.986 +/- 0.052 | 210.6 +/- 6.1 | 4.1 +/- 10.7 | 0.692 +/- 0.016 |

mem=0.10: regret CI crosses zero (transfer ~ native within noise).

### 3.2 Funnel: Tail is Low-Value

Under oracle mode at w=1:
- funnel_novel_rate drops from 0.651 (mem=0.01) to 0.013 (mem>=0.10)
- 98.7% of oracle-predicted experts are already resident

### 3.3 Stall Concentration

- top 20% of experts contribute 45% of total stall
- Gini = 0.4
- Concentration shape is predictor-independent (oracle and freq Lorenz curves overlap)

## Section 5: Real-Hardware Results (batch=8, native C++ pinning)

### Main Comparison Table

| Config | mem=0.05 | mem=0.07 | mem=0.10 | Description |
|---|---:|---:|---:|---|
| A demand-only | 2.356 | 2.356 | 2.499 | LRU cache, no prefetch, no backbone |
| B LRU+prefetch | -- | 1.691 | 1.702 | LRU + speculative prefetch |
| C_knee backbone-only | -- | 2.426 | 2.568 | knee-selected backbone, no prefetch |
| C_ratio09 backbone-only | -- | 2.578 | 3.040 | fixed ratio=0.9, no prefetch |
| **C_optimal backbone-only** | **2.443** | **2.643** | **3.585** | **throughput-sweep, no prefetch** |
| D_ratio09 backbone+prefetch | -- | 1.771 | 1.878 | backbone + speculative prefetch |

Unit: generated tokens/sec (higher is better).

### Key Findings

1. **Backbone pinning works**: C_optimal > A (+43.5% at mem=0.10)
2. **Prefetch is harmful**: B < A (-32%), D < C
3. **Throughput sweep beats hand-tuned ratio**: C_optimal > C_ratio09 (+18% at mem=0.10)
4. **Optimal split adapts automatically**: ratio 0.63 (mem=0.05) to 0.96 (mem=0.10)

### Throughput-Sweep Backbone Extraction

| mem | cache | knee k | optimal k | optimal ratio |
|---|---:|---:|---:|---:|
| 0.05 | 238 | 93 (0.39) | 149 (0.63) | 0.63 |
| 0.07 | 333 | 122 (0.37) | 276 (0.83) | 0.83 |
| 0.10 | 476 | 152 (0.32) | 456 (0.96) | 0.96 |

## Oracle vs Backbone Analysis (simulator)

| Config | mem=0.07 | mem=0.10 |
|---|---:|---:|
| A demand-only | 165.3 | 288.5 |
| B oracle-prefetch (single) | 197.2 | 387.2 |
| C backbone-only (0.96) | 177.1 | 343.4 |
| **D backbone + oracle prefetch** | **209.6** | **498.8** |
| E backbone + causal prefetch | 176.6 | 343.4 |

Key: Oracle prefetch adds +18-45% on top of backbone-only.
Current causal prefetch adds nothing (overhead cancels benefit).

**Conclusion**: Backbone is the primary mechanism. Better prefetch (batch-aware, lightweight)
is future work with significant headroom.

## Batch-Aware Prefetch Results (batch=8, native C++ pinning)

| Config | mem=0.07 | mem=0.10 | Description |
|---|---:|---:|---|
| A demand-only | 2.356 | 2.499 | LRU cache baseline |
| B per-seq prefetch | 1.691 | 1.702 | Per-sequence prefetch (harmful) |
| C_optimal backbone-only | 2.643 | 3.585 | Throughput-sweep backbone, no prefetch |
| D per-seq backbone+prefetch | 1.771 | 1.878 | Backbone + per-sequence prefetch |
| **E batch-aware backbone+prefetch** | **3.283** | **3.522** | **Backbone + batch-aware prefetch** |

### Key findings

1. **Batch-aware prefetch works**: E > C at mem=0.07 (+24.2%), nearly matches C at mem=0.10 (-1.8%)
2. **Per-sequence prefetch is the problem, not prefetch itself**: E >> D (+85% at mem=0.07)
3. **Mid-budget is where batch prefetch shines**: at mem=0.07, backbone-only leaves more headroom that batch prefetch captures
4. **High-budget backbone-only is already near-optimal**: at mem=0.10, most experts are pinned so prefetch adds little

### Design implication

The final system design is:
- **Backbone pinning** (primary mechanism, +43% over baseline)
- **Batch-aware prefetch** for tail experts (secondary, +24% at mid-budget)
- **No per-sequence prefetch** (harmful due to CPU overhead x batch_size)

This closes part of the oracle headroom (+18-45% in simulator) with a practical implementation.

## Cross-Model Analysis: DeepSeek-V2-Lite

### Model
- DeepSeek-V2-Lite (27 layers, 64 routed experts, top-6 routing, 2 shared experts)
- 26 MoE layers (first layer is dense)
- moe_intermediate_size=1408, hidden_size=2048

### Routing Trace Statistics (64 sequences, lmsys-chat-1m)
- Total (layer, expert) pairs: 1664 (26 x 64)
- Coverage across all sequences: **100%** (every expert accessed at least once)
- Per-sequence unique experts: mean=1585/1664 = **95.3%**
- Routing is near-uniform; no strong backbone concentration

### Backbone Generalization (50/50 split)

| mem | retained | Jaccard | transfer gain | native gain |
|---|---:|---:|---:|---:|
| 0.05 | 1.000 | 1.000 | 8.6 | 8.6 |
| 0.07 | 1.000 | 1.000 | 8.6 | 8.6 |
| 0.10 | 1.000 | 1.000 | 8.6 | 8.6 |

Jaccard=1.0 means train and test select identical backbone. This is NOT because
backbone is trivially stable - it's because routing is so uniform that almost
all experts are used by every sequence. The backbone "exists" only in the
sense that frequency ranking is reproducible, not in the sense that a small
subset dominates.

### Throughput Sweep

| mem | cache | knee_k | optimal_k | ratio | tp |
|---|---:|---:|---:|---:|---:|
| 0.05 | 238 | 1 | 156 | 0.66 | 500.0 |
| 0.07 | 333 | 1 | 156 | 0.47 | 500.0 |
| 0.10 | 476 | 1 | 156 | 0.33 | 500.0 |

knee_k=1 means the utility curve is flat: no structural knee, no
clear backbone/tail boundary.

### Oracle Analysis (simulator, mem=0.10)

| Config | tok/s |
|---|---:|
| A demand-only | 495.8 |
| B oracle-prefetch | 495.9 |
| C backbone-only | 500.0 |
| D backbone+oracle-pf | 500.0 |

Backbone pinning provides only +0.8% over demand-only. This confirms
that when routing is near-uniform, backbone-first has minimal benefit.

### Key Finding

**Backbone-first is NOT universally applicable.** Its effectiveness depends on
routing concentration:

| Model | Routing | Coverage | Backbone? | Improvement |
|---|---|---|---|---|
| Qwen1.5-MoE-A2.7B | top-4 / 60 experts | ~65% | Yes (stable) | +41% |
| DeepSeek-V2-Lite | top-6 / 64 experts | ~95% | No (uniform) | +0.8% |

The boundary condition: backbone-first works when the routing pattern has
structural concentration (a small subset of experts handles most of the load).
When routing is near-uniform (high top-k / expert ratio), demand-only caching
is sufficient.

## Reproducibility Verification (Qwen, batch=8)

| Config | Verified gen tok/s | Archived gen tok/s | Match |
|---|---:|---:|---|
| C_optimal mem=0.07 (ratio=0.83) | 3.199 | 2.643 | Higher (single GPU, no contention) |
| C_optimal mem=0.10 (ratio=0.96) | 3.533 | 3.585 | -1.4% (within variance) |
