# Roadmap: Backbone-First MoE Serving

## 1. Project Thesis

### One-sentence thesis

MoE serving should not be framed as a global cache-heuristic problem.
It should be framed as a `backbone-first` systems problem:

1. identify the stable resident expert backbone
2. pin it on GPU
3. treat the remaining tail as a low-value residual problem

### Preferred paper framing

The project should now use the following framing:

> MoE serving has two structural facts:
> 1. there exists a stable resident backbone that nearly saturates the achievable gain
> 2. once that backbone is pinned, the remaining tail has very little system-useful novel opportunity
>
> Therefore, the correct serving abstraction is not a single global cache, but
> `resident backbone + demand fallback`.

This is stronger than the earlier negative framing:

```text
P(access) != usefulness
```

That observation is still true, but it now serves as a mechanism explanation, not the headline.

### Why this is a bigger project

This project is no longer about one better prefetch heuristic.
It is about a reusable systems abstraction for MoE serving:

- backbone extraction
- backbone placement
- residual-tail management
- future adaptation across workloads, models, and deployment settings

That is large enough to support:

- one strong systems paper now
- follow-up work on adaptation, transfer, and distributed serving later

---

## 2. What Has Changed

### Old roadmap

The old roadmap was still too close to:

- improving prefetch
- improving cache policy
- improving token-level prediction

### New roadmap

The new roadmap is:

1. prove that a stable resident backbone exists
2. prove that this backbone generalizes
3. prove that the tail is low-value even under strong prediction
4. build a system around this structure

This is a project-level shift, not a tuning shift.

---

## 2.1 Related Work and Positioning

The current landscape is no longer empty.
There are already multiple strong MoE offloading / serving papers, but they mostly use a different abstraction.

### Closest related papers

1. `MoE-Infinity` ([arXiv:2401.14361](https://arxiv.org/abs/2401.14361))
   - request-level tracing
   - dynamic expert caching
   - dynamic prefetching
   - abstraction: `global expert cache`

2. `Pre-gated MoE` ([arXiv:2308.12066](https://arxiv.org/abs/2308.12066))
   - algorithm-system co-design
   - reduces routing dynamism through pre-gating
   - abstraction: `make expert activation easier to serve`

3. `ExpertFlow` ([arXiv:2410.17954](https://arxiv.org/abs/2410.17954))
   - predictive routing path
   - token scheduling
   - CPU/GPU expert scheduling
   - abstraction: `predict and schedule better online`

4. `HOBBIT` ([arXiv:2411.01433](https://arxiv.org/abs/2411.01433))
   - mixed-precision expert offloading
   - token / layer / sequence level mechanisms
   - abstraction: `reduce miss cost and improve dynamic cache behavior`

5. `MoE-Lightning` ([arXiv:2411.11217](https://arxiv.org/abs/2411.11217))
   - CPU-GPU-I/O pipelining
   - hierarchical roofline performance model
   - abstraction: `high-throughput batch inference under memory constraints`

6. `DAOP` ([arXiv:2501.10375](https://arxiv.org/abs/2501.10375))
   - per-sequence dynamic expert placement
   - predictive pre-calculation
   - abstraction: `adaptive expert allocation`

7. `DuoServe-MoE` ([arXiv:2509.07379](https://arxiv.org/abs/2509.07379))
   - dual-phase scheduling for prefill vs decode
   - offline predictor + online prefetch
   - abstraction: `decode-stage predictive scheduling`

8. `Context-Aware MoE Inference on CXL-Enabled GPU-NDP Systems` ([arXiv:2512.04476](https://arxiv.org/abs/2512.04476))
   - prefill-stage activation statistics guide decode placement
   - dynamically pins hot experts in GPU HBM
   - maps cold experts to CXL-NDP
   - this is the most similar system-level direction

9. `OD-MoE` ([arXiv:2512.03927](https://arxiv.org/abs/2512.03927))
   - removes expert caches entirely
   - fully on-demand loading with a very strong predictor
   - useful as a contrast point rather than a direct baseline

### What most existing work assumes

Most existing systems still assume one of the following:

- the main problem is `better online prediction`
- the main problem is `better global expert caching`
- the main problem is `better pipelining / overlap`

### Our current positioning

This project should not claim:

- "we are the first to pin hot experts"
- "we have a better prefetch heuristic"
- "we improve cache hit rate"

The stronger and more defensible position is:

> Existing work mostly optimizes a global cache / prediction abstraction.
> Our evidence suggests the abstraction itself is incomplete:
> MoE serving naturally decomposes into a stable resident backbone and a low-value residual tail.

That is the paper's main system contribution.

### Why the system angle is strong

The system angle is already non-incremental enough because it changes:

1. the basic object
   - from `uniform cache item`
   - to `backbone expert` vs `tail expert`

2. the optimization target
   - from `hit rate / prediction accuracy`
   - to `captured stall value under memory budget`

3. the architecture
   - from `single global cache`
   - to `resident backbone + demand fallback`

### Why the algorithm angle is still weaker right now

The current algorithmic implementations:

- `profile_freq`
- `profile_miss_stall`
- `profile_depth_freq`
- `utility_freq`

are still too close to heuristic ranking functions.

If presented as the main novelty, they will look incremental.

So the algorithm must be reframed as an explicit optimization problem rather than another score.

---

## 3. Current Evidence

## 3.1 Generalization is now a core result

The main evidence is no longer the old `n=8` split experiment.
The project now has `n=64` routed traces with `8-fold CV`.

### `profile_freq`, `resident_ratio=0.9`, `resident_profile_ratio=0.2`

| mem | retained gain | transfer gain over single | regret vs native | Jaccard |
|---|---:|---:|---:|---:|
| 0.05 | 0.830 +/- 0.195 | 178.5 +/- 57.2 | 32.7 +/- 37.2 | 0.648 +/- 0.024 |
| 0.07 | 0.877 +/- 0.091 | 223.5 +/- 21.9 | 31.7 +/- 23.0 | 0.679 +/- 0.032 |
| 0.10 | 0.986 +/- 0.052 | 210.6 +/- 6.1 | 4.1 +/- 10.7 | 0.692 +/- 0.016 |

### What these numbers mean

1. `mem=0.10` is the strongest result.
   The regret confidence interval crosses zero, which means transfer backbone and native backbone are statistically indistinguishable within noise.

2. `mem=0.07` is also strong.
   The held-out transfer keeps most of the native gain with moderate variance.

3. `mem=0.05` is positive but unstable.
   It should be treated as a phase-boundary regime and moved to appendix in the final paper.

4. Jaccard improved from the earlier `~0.52` to `~0.69`.
   This suggests the profiled backbone is not arbitrary; it becomes more stable as data coverage increases.

### Consequence

`backbone generalization` is no longer defensive ablation-only evidence.
It is now a main-section result.

---

## 3.2 Funnel evidence explains why the tail is weak

From the existing fullscan JSONs, the key mechanism is already visible.

Under oracle mode at `w=1`:

- `funnel_novel_rate` drops from `~0.651` at `mem=0.01`
- to `~0.013` at `mem >= 0.10`

This means that in the stable-memory regime, even an oracle predictor mostly "predicts" experts that are already resident or already in transfer.

So the weak tail is not just:

```text
our predictor is not good enough
```

It is:

```text
the tail contains very little system-useful novel opportunity in the first place
```

This is the mechanism-level justification for the backbone-first abstraction.

---

## 3.3 What is already implemented

The current simulator already contains most of the needed instrumentation:

- funnel counters
- tail-only funnel counters
- per-layer residual stall
- per-expert critical stall accounting
- resident policies:
  - `oracle_freq`
  - `profile_freq`
  - `profile_depth_freq`
  - `profile_miss_stall`
- deadline-aware admission
- value-aware admission

So the remaining work is primarily:

- aggregation
- plotting
- a small targeted dump for per-expert visualization

Not major simulator engineering.

---

## 4. Final Paper Claim

## 3.4 Real-hardware evidence (batch=8, native C++ pinning)

### Main comparison table

All results on Qwen1.5-MoE-A2.7B-Chat, A800 80GB, batch=8, 16 prompts, 64 new tokens.
`C_optimal` uses throughput-sweep backbone extraction (no manual ratio tuning).

| Config | mem=0.05 | mem=0.07 | mem=0.10 | description |
|---|---:|---:|---:|---|
| A demand-only | 2.356 | 2.356 | 2.499 | LRU cache, no prefetch, no backbone |
| B LRU+prefetch | -- | 1.691 | 1.702 | LRU cache + speculative prefetch |
| C_knee backbone-only | -- | 2.426 | 2.568 | knee-selected backbone, no prefetch |
| C_ratio09 backbone-only | -- | 2.578 | 3.040 | fixed ratio=0.9 backbone, no prefetch |
| **C_optimal backbone-only** | **2.443** | **2.643** | **3.585** | **throughput-sweep backbone, no prefetch** |
| D_ratio09 backbone+prefetch | -- | 1.771 | 1.878 | backbone + speculative prefetch |

Unit: generated tokens/sec (higher is better).

### What the table proves

1. **Backbone pinning works**: `C_optimal > A` at all memory budgets (+3.7% to +43.5%)
2. **Prefetch is harmful**: `B < A` everywhere; `D < C` everywhere
3. **Throughput sweep beats hand-tuned ratio**: `C_optimal > C_ratio09` (+2.5% at mem=0.07, +18% at mem=0.10)
4. **Optimal split adapts to memory budget**: ratio automatically varies from 0.63 (mem=0.05) to 0.96 (mem=0.10)

### Throughput-sweep backbone extraction results

| mem | cache | knee k (structural lower bound) | optimal k (throughput-maximizing) | optimal ratio | simulator tok/s |
|---|---:|---:|---:|---:|---:|
| 0.05 | 238 | 93 (0.39) | 149 (0.63) | 0.63 | 104.2 |
| 0.07 | 333 | 122 (0.37) | 276 (0.83) | 0.83 | 188.5 |
| 0.10 | 476 | 152 (0.32) | 456 (0.96) | 0.96 | 324.3 |

The knee defines the structural backbone core (~1/3 of cache).
The throughput sweep finds that pinning well beyond the knee is optimal, because in the demand-only tail regime every additional pinned expert eliminates one H2D transfer.
The optimal ratio is NOT a fixed constant; it increases with budget because the minimum demand buffer is approximately constant (~30-50 slots).

### Why prefetch hurts in batch=8

The per-sequence prefetch control path scales linearly with batch size:
- 8 embed_prefetch calls per token step
- 8 x 24 = 192 traj_prefetch calls per token step
- each call does cosine similarity matching, priority sorting, and H2D enqueue
- all on CPU, in Python, sequentially

This creates three compounding costs:
1. CPU scheduling overhead x batch_size
2. H2D bandwidth contention between speculative and demand transfers
3. Cache pollution across sequences in the same batch

### Batch-aware prefetch results

| Config | mem=0.07 | mem=0.10 |
|---|---:|---:|
| C_optimal backbone-only | 2.643 | 3.585 |
| D per-seq backbone+prefetch | 1.771 | 1.878 |
| **E batch-aware backbone+prefetch** | **3.283** | **3.522** |

Batch-aware prefetch replaces 200 per-sequence Python calls per step with 24 batch-level calls (one per layer). This eliminates CPU scheduling overhead while preserving prefetch benefit.

At mem=0.07: E beats C by +24% (batch prefetch captures oracle headroom at mid-budget).
At mem=0.10: E nearly matches C (-1.8%, backbone already near-saturated).

### Design conclusion

The strongest runtime configuration is:

```text
resident backbone + batch-aware prefetch for tail
```

- **Backbone pinning** is the primary mechanism (+43% over demand-only baseline)
- **Batch-aware prefetch** adds secondary gain at mid-budget (+24% at mem=0.07)
- **Per-sequence prefetch** is harmful and should not be used

---

## 4. Final Paper Claim

## 4.1 Claim

The paper should claim:

> MoE serving has a stable resident backbone that generalizes across held-out sequences.
> Once this backbone is pinned, the tail should be served via batch-aware demand/prefetch, not per-sequence speculative prefetch.
> The backbone is extracted automatically via throughput-sweep selection with zero manual tuning.

Therefore:

> the right abstraction is `resident backbone + demand fallback`, not a single global cache.

## 4.2 What the paper is not

The paper is not:

- a better cache score
- a better predictor
- a better admission threshold
- a better prefetch window

Those may appear as supporting components, but they are not the contribution.

---

## 5. Method Roadmap

## 5.1 Paper-1 method

The first paper should focus on:

1. `Backbone extraction`
   - profile a short prefix
   - rank experts by systems value
   - select a resident set under a memory budget

2. `Backbone-first serving`
   - resident backbone is pinned
   - tail is handled by exact on-demand loading
   - speculative prefetch is not required for the main result

3. `Structural evidence`
   - backbone generalization
   - predictor funnel collapse
   - stall concentration
   - real-hardware evidence that resident-only beats prefetching variants

This is already enough for a strong systems paper.

## 5.2 Algorithm roadmap: make the method solid

The algorithm should no longer be described as:

- "pick top experts by frequency"
- "pick experts by a better score"
- "use a stronger heuristic than LRU/prefetch"

That framing will always look incremental.

Instead, the algorithm should be formalized as:

## 5.2.1 Stable Backbone Extraction (SBE)

Given:

- a short profiling prefix `P`
- a resident memory budget `B`
- expert universe `E`

select a resident set `S subset E` that maximizes expected held-out stall reduction:

```text
maximize    U(S)
subject to  size(S) <= B
```

where `U(S)` is not hit rate, but systems value:

```text
U(S) = expected demand-stall reduction on future tokens / sequences
```

This is the right formal statement of the problem.

## 5.2.2 Immediate algorithm upgrade path

To make the algorithm more solid without over-expanding the scope:

1. `Replace score heuristics with counterfactual utility estimation`

   For each expert `e`, estimate:

   ```text
   u_e = stall_without_pinning(e) - stall_with_pinning(e)
   ```

   under the profiling prefix or profiling subset.

   This is stronger than `frequency` because it directly targets the systems objective.

2. `Promote resident selection to budgeted critical-set extraction`

   Instead of "rank by freq and take top-k", describe the solver as:

   - estimate per-expert criticality
   - solve a budgeted selection problem
   - evaluate transfer to held-out sequences

   If expert sizes are equal, top-k by estimated `u_e` is the budget solver.
   That is much cleaner than pretending the main contribution is a heuristic.

3. `Add a stability criterion`

   A backbone is not valuable just because it has high utility on the prefix.
   It should also be stable across folds / prefix slices.

   So the algorithmic target should become:

   ```text
   maximize utility + stability
   ```

   or at minimum:

   - utility on prefix
   - transfer retained gain on held-out sequences
   - Jaccard stability across folds

   This moves the method from "best score on one trace" to "extract a transferable structure."

4. `Treat online speculative logic as optional ablation`

   The main algorithm is the extraction of `S`.
   Any online predictor or prefetch path should be explicitly positioned as:

   - optional
   - secondary
   - removable without weakening the core backbone result

## 5.2.3 Stronger algorithm names

If the paper needs an algorithmic name, use one of:

- `Stable Backbone Extraction`
- `Budgeted Backbone Extraction`
- `Critical Resident Set Selection`

These names are stronger than `profile_freq`.

## 5.2.4 What would make the algorithm genuinely non-incremental

The algorithm becomes genuinely non-incremental if it has all three:

1. `explicit objective`
   - maximize stall reduction, not hit rate

2. `budgeted selection formulation`
   - choose a resident set under memory budget

3. `transfer / stability criterion`
   - the selected set is evaluated by held-out retained gain, not just in-prefix fit

Once written this way, even a simple solver is acceptable.
The novelty is then in the formulation and the transferable structure, not in a complicated ranker.

## 5.3 Runtime solidification roadmap

The current implementation is already a valid research prototype:

- structural findings are established in Python
- the simulator is usable
- the runtime path already supports resident pinning on real hardware

But it is not yet a fully solid open-source system.

The missing step is not "write random C++ because systems papers use C++".
The missing step is:

> turn backbone-first from a Python policy layered on top of FineMoE
> into a runtime-native memory hierarchy feature.

### 5.3.1 What is acceptable to keep in Python

These components should remain in Python for now:

1. `profiling and backbone extraction`
   - routed trace loading
   - fold split / held-out evaluation
   - resident-set selection logic

2. `paper-facing evaluation`
   - CV aggregation
   - plotting
   - oracle-native augmentation
   - section-3 / section-5 result summarization

3. `high-level policy orchestration`
   - choosing memory budgets
   - switching between resident policies
   - running ablations

These parts are not runtime critical-path bottlenecks.
Keeping them in Python is normal and does not weaken the systems contribution.

### 5.3.2 What should move into runtime-native support

The first solidification target is not a new executor.
It is explicit runtime support for the backbone abstraction.

The minimum set of runtime-native features should be:

1. `resident set as a first-class runtime object`
   - not just "remove tensors from offload_set"
   - resident tensors should have explicit runtime identity
   - runtime should expose resident count / bytes / pressure directly

2. `resident-aware memory accounting`
   - resident capacity reserved explicitly
   - speculative and demand capacity tracked separately
   - avoid relying on accidental capacity interactions

3. `resident-aware fetch / evict path`
   - resident tensors never enter normal eviction
   - speculative fetch should skip resident tensors natively
   - demand path should know whether a miss is in the tail or not

This is the most valuable place to add C++ / runtime work.

### 5.3.3 What can stay as a later phase

These are useful, but should come after the core backbone abstraction is made runtime-native:

1. `async stream-aware overlap`
   - multiple H2D streams
   - transfer/compute overlap
   - deeper executor scheduling

2. `grouped / chunked expert transfer`
   - sub-expert loading
   - chunk-level materialization

3. `fully custom serving runtime`
   - deeper executor replacement
   - large-scale productionization

These are good future system-building directions, but they are not the first thing needed for a solid backbone-first release.

### 5.3.4 Three-phase solidification plan

#### Phase A: paper prototype -- DONE

- structural findings established in Python simulator
- real hardware smoke tests validated backbone pinning
- throughput sweep selector implemented and validated

#### Phase B: runtime-native backbone support -- DONE

Implemented in commit `ebfceae`:

- `Node::is_resident` flag in `core/model/model_topology.h`
- `RemoveCachedSparseNode` skips `is_resident` nodes in `core/prefetch/task_scheduler.cpp`
- `PinResidentNodes()` in `core/prefetch/archer_prefetch_handle.cpp`: loads to GPU + marks resident
- `pin_resident_nodes()` exposed via Python bindings in `core/python/py_archer_prefetch.cpp`
- Python `pin_resident_experts()` calls native API, then `begin()` to materialize tensor pointers

All Section 5 real-hardware results (C_optimal = 3.585 tok/s) use this native path.

#### Phase C: high-performance runtime refinement

Goal:

- make the system more production-like

Possible work:

- async transfer overlap
- stream scheduling
- chunked loading
- stronger memory-tier manager

This phase is about performance refinement, not about establishing the core abstraction.

### 5.3.5 Open-source release standard

Before open-source release, the backbone-first stack should satisfy:

1. `clear module boundaries`
   - library code in `finemoe/backbone/*`
   - runtime entrypoints separate
   - paper analysis separate

2. `runtime-native resident abstraction`
   - no hidden dependence on ad hoc script patches

3. `reproducible entrypoints`
   - one command for profiling
   - one command for CV/generalization
   - one command for section-5 runtime evaluation

4. `basic automated tests`
   - workload split
   - metric aggregation
   - runtime sweep config generation

5. `clean config story`
   - no host-specific path edits in shared defaults

This is what "solid" should mean for this project.

## 5.4 Bigger-project extensions

After Paper-1, the project can naturally expand into:

1. `Adaptive backbone`
   - online backbone updates
   - sliding-window re-profiling
   - workload-shift tracking

2. `Cross-workload transfer`
   - chat -> code
   - code -> chat
   - safety / jailbreak / structured tasks

3. `Cross-model transfer`
   - Qwen-MoE
   - DeepSeek-MoE
   - future MoE LLMs

4. `Backbone-aware distributed serving`
   - multi-GPU placement
   - disaggregated GPU/CPU memory tiers
   - cluster-aware backbone replication

5. `Backbone-aware compression`
   - quantize tail more aggressively
   - compress tail routes / storage
   - preserve backbone quality first

This is why the current roadmap now supports a larger project rather than a one-off paper.

---

## 6. Immediate Priorities

The priority order has changed.

## Priority 1: Make `n=64 CV` the central result

Deliverables:

1. `retained_gain_fraction` bar plot with `95% CI`
2. `native vs transfer` scatter with `y=x`
3. `regret vs native` plot

Interpretation:

- `mem=0.10` is main text
- `mem=0.07` is supporting main-text evidence
- `mem=0.05` goes to appendix

## Priority 2: Aggregate the cross-predictor funnel figure

Use existing fullscan JSONs for:

- oracle
- `history_freq`
- `pl_ctr`

Deliverables:

1. funnel-stage comparison
2. `novel_rate vs memory_ratio`
3. possibly `useful_rate vs memory_ratio`

Interpretation:

This becomes the main mechanism figure for why tail prediction is weak.

## Priority 3: Targeted per-expert stall dump

Run one targeted configuration, dump:

- `per_expert_critical_stall_ms`

Deliverables:

1. Lorenz-style concentration curve
2. `(layer, expert)` heatmap
3. top-5 / top-10 / top-20 contribution table

Interpretation:

This supports the claim that stall value is structurally asymmetric across experts.

## Priority 4: Tighten useful-stall accounting

If time allows:

- replace proxy `transfer_time` credit with stored `saved_stall`

This is good to have, but no longer blocks the main storyline.

## Priority 5: Second routed trace / cross-workload transfer

This is now a plus, not a rescue path.

It should be done after the main figures above, unless a second routed trace is already available at low cost.

---

## 7. Revised Section Structure

The paper should now follow the stronger structural framing.

## Section 1: Introduction

Lead with:

- stable resident backbone
- low-value tail
- backbone-first serving abstraction

Not with heuristic tuning.

## Section 2: Background

Explain:

- MoE routing and expert residency
- why global cache intuitions fail
- why systems value differs from access count

## Section 3: Structural Findings

### 3.1 Stable resident backbone

Use the `n=64 + 8-fold CV` result.

### 3.2 Tail opportunity collapses

Use the funnel figure.

### 3.3 Stall is structurally asymmetric

Use the per-expert concentration figure.

## Section 4: System Design

Present:

- resident backbone
- fallback path
- optional speculative logic only as an ablation, not as the main design

### Important correction: fixed split is not the final method

The current prototype uses a fixed:

```text
resident_ratio = 0.9
```

This should **not** be presented as the final backbone-first method.

The runtime evidence already shows why:

- `mem=0.05`: total `238`, resident `214`, speculative slack `24`
- `mem=0.07`: total `333`, resident `300`, speculative slack `33`
- `mem=0.10`: total `476`, resident `428`, speculative slack `48`

This means the current prototype can over-allocate memory to the resident pool in low-budget regimes when paired with speculative prefetch.
So the real problem is not just `which experts to pin`, but also:

```text
how much budget should be allocated to the resident backbone
vs. how much should remain elastic for the residual tail
```

This is a more fundamental formulation than tuning `resident_ratio`.

### Non-incremental formulation (now implemented)

The backbone extraction method has two stages, both fully data-driven:

```text
Stage 1: RANK experts by profiling-prefix frequency (or stall contribution)
Stage 2: SWEEP resident capacity k from the utility-curve knee to cache capacity,
         evaluate simulated throughput at each point, select the k that maximizes throughput
```

This is equivalent to solving:

```text
maximize    throughput(k)
subject to  knee <= k <= cache_capacity
```

where `throughput(k)` is evaluated by the simulator with the top-k experts pinned and the remaining `cache_capacity - k` slots used for demand cache.

The sweep takes ~20 points and runs in seconds (simulator only, no GPU needed).

### Why this works

In the demand-only tail regime (no prefetch):
- each pinned expert eliminates one potential H2D transfer
- the only cost of pinning more is shrinking the demand cache
- the demand cache only needs enough slots for the per-step tail working set (~30-50 experts)

So the optimal split is where marginal pinning benefit equals marginal demand-cache cost.
At high memory budgets, this pushes the ratio near 1.0 (pin almost everything).
At low memory budgets, more demand buffer is needed, pulling the ratio down.

### Empirical results (already validated)

| mem | knee k (structural core) | optimal k (throughput sweep) | optimal ratio |
|---|---:|---:|---:|
| 0.05 | 93 (0.39) | 149 (0.63) | 0.63 |
| 0.07 | 122 (0.37) | 276 (0.83) | 0.83 |
| 0.10 | 152 (0.32) | 456 (0.96) | 0.96 |

The knee defines "which experts matter" (structural boundary).
The throughput sweep decides "how many to pin" (budget allocation).
Both are workload-adaptive with zero manual tuning.

### Relationship to fixed ratio=0.9

The old `resident_ratio=0.9` was a hand-tuned heuristic.
The throughput sweep produces a ratio that:
- matches 0.9 approximately at high memory (0.96 at mem=0.10)
- automatically lowers at low memory (0.63 at mem=0.05)
- beats the fixed 0.9 on real hardware (+18% at mem=0.10 because it pins slightly more)

So `ratio=0.9` is now superseded. The paper should present the throughput sweep as the method.

This is where BDMS or the final system name belongs.

## Section 5: Evaluation

Main comparisons:

- single cache baseline
- backbone-only system
- prefetching baselines as negative evidence
- oracle upper-bound references where appropriate

## Section 6: Discussion

Discuss:

- why `mem=0.05` behaves differently
- why batch size changes the working-set regime
- why speculative prefetch is harmful in the current runtime
- what remains for cross-workload / cross-model transfer
- what this implies for future MoE systems

---

## 8. Success Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `mem=0.10` backbone generalization: transfer ~ native | PASS (retained 0.986 +/- 0.052, regret CI crosses 0) |
| 2 | `mem=0.07` backbone generalization: strongly positive | PASS (retained 0.877 +/- 0.091) |
| 3 | Oracle funnel: low novel-rate in stable regimes | PASS (novel_rate = 0.013 at mem >= 0.10) |
| 4 | Per-expert stall concentration: non-uniform | PASS (top 20% = 45%, Gini = 0.4, predictor-independent) |
| 5 | Real hardware: `C_optimal > A` in batched regime | PASS (3.533 > 2.499 at mem=0.10, verified) |
| 6 | Throughput sweep: automatic, no manual tuning | PASS (ratio auto-adapts: 0.63-0.96) |
| 7 | Runtime-native C++ pinning: no shape mismatch | PASS (is_resident flag, all mem points work) |
| 8 | Second model tested | PASS (DeepSeek-V2-Lite: boundary condition identified) |
| 9 | Paper reads as structural systems result, not heuristic | ON TRACK |

---

## 9. Current Bottom Line

The project has completed its core experimental validation:

- **Structural evidence** (Section 3): backbone exists and generalizes on Qwen (retained 0.986)
- **Cross-model evidence** (Section 3): DeepSeek-V2-Lite shows no backbone (near-uniform routing), confirming backbone is routing-concentration-dependent
- **System design** (Section 4): backbone pinning + demand-only tail
- **Algorithm** (Section 4): two-stage throughput-sweep extraction, zero manual tuning
- **Runtime** (Phase B): C++ native `is_resident` flag, eviction exemption
- **Real hardware** (Section 5): C_optimal = 3.533 tok/s (+41% over baseline), reproducibly verified

### Cross-Model Summary

| Model | Routing | Expert Coverage | Backbone? | Backbone Improvement |
|---|---|---|---|---|
| Qwen1.5-MoE-A2.7B | top-4 / 60 experts | ~65% per sequence | **Yes** (stable) | **+41%** |
| DeepSeek-V2-Lite | top-6 / 64 experts | ~95% per sequence | No (uniform) | +0.8% |

**Backbone-first is effective when routing has structural concentration. When routing is near-uniform, demand-only caching is sufficient.**

The main remaining work is:

1. **Write the paper** -- all data exists, just needs to be written up
2. **Section 3 formal plots** -- n=64 CV bar plots, funnel figure, Lorenz curve
3. **Batch=1 verification** -- confirm backbone wins at batch=1 (Qwen verify still running)

The project thesis is:

```text
MoE serving with concentrated routing decomposes into a stable resident
backbone and a demand-only tail. The backbone is extracted automatically
via throughput-sweep selection. The method's applicability depends on
routing concentration: it provides +41% throughput on Qwen (top-4/60)
but is not needed for DeepSeek-V2-Lite (top-6/64, near-uniform routing).
```

This is a structural systems contribution with clearly identified boundary conditions.
