# Current Plan

## Ground Truth

- `fresh clone` on `172` is now end-to-end runnable for `Qwen`, `OLMoE`, and `DeepSeek` smoke.
- Fair-budget runtime now uses `free_device_memory_ratio` instead of total-memory ratio.
- Selector now supports `--sparse-budget-bytes` and can be calibrated from the runtime `A` run.
- Zero-resident cases now explicitly degenerate to baseline outputs instead of pretending to run a meaningful `C`.

## Current Fair Results

- `Qwen @ 0.10`
  - old fair unified runtime:
    - `A = 14.0118 tok/s`
    - `C = 14.3550 tok/s`
    - `+2.45%`
  - current modulelist two-lane runtime:
    - long-workflow gains observed at `+6.25%`, `+4.69%`, `+3.46%`
    - current mean is `~+4.8%`
    - this is a stable positive result, but still only medium-strength throughput evidence
  - `resident_admitted_count = 165`
  - `resident_registry.budget_bytes = sparse_budget_bytes = 4735686246`
- `Qwen @ 0.07`
  - runtime-calibrated selector returns `resident = 0`
  - `C` degenerates to baseline
- `OLMoE @ 0.045 / 0.05`
  - runtime-calibrated selector returns `resident = 0`
  - `C` degenerates to baseline
- `DeepSeek`
  - `0.07 / 0.10` fair-budget reruns both return `resident = 0`
  - `C` degenerates to baseline

## Main Diagnosis

- The old gains were partly inflated by unfair or weakly calibrated budget accounting.
- The new fair pipeline is more credible but much more conservative.
- The latest `Qwen 0.10` result shows that runtime bifurcation matters:
  - structural backbone signal is real
  - unified generic runtime was hiding much of that signal
  - modulelist `resident lane + demand lane` raises fair long-workflow gains from `+2.45%` to roughly `+5%`
- We still do not know whether MoE serving is fundamentally memory-bound enough to justify another round of memory-hierarchy design.
- The main unanswered question is now:
  - how much of the end-to-end time is actually load stall
  - what the realistic zero-loading / oracle ceiling looks like
  - whether current weak gains come from a bad method or a low memory-optimization ceiling

## Three-Layer Diagnosis

Use the following order when judging weak throughput results:

### 1. Pin What

Question:
- is the resident set itself valuable?

Primary evidence:
- `retained_gain_fraction`
- `regret_vs_native`
- `Jaccard` stability across folds
- top-k stall coverage
- per-expert `saved_stall_per_byte`

Current judgment:
- not the main problem on `Qwen`
- backbone signal is stable and transferable enough that selector quality is not the first bottleneck

Relevant code paths:
- `experiments/simulation/select_adaptive_resident_set.py`
- `backbone_moe/evaluation.py`

### 2. Pin How Much

Question:
- is the feasibility rule too conservative, leaving too much budget unused?

Primary evidence:
- `sparse_budget_bytes`
- `selection_budget_bytes`
- `resident_registry.admitted_bytes`
- `resident_registry.admitted_count`
- `resident_registry.clipped`
- `resident_bytes / budget_bytes`
- zero-resident degeneration frequency

Current judgment:
- likely a real secondary problem
- current service envelope is still conservative enough that some fair points collapse to `resident = 0`

Relevant code paths:
- `experiments/simulation/select_adaptive_resident_set.py`
- `finemoe/runtime/model_offload.py`

### 3. Pin Then Serve

Question:
- are resident hits and tail misses being served cheaply enough?

Primary evidence for modulelist:
- `module_begin_wall_time_sec`
- `module_end_wall_time_sec`
- `manual_subtree_begin_wall_time_sec`
- `manual_subtree_end_wall_time_sec`
- `modulelist_demand_compute_wall_time_sec`
- `modulelist_resident_compute_wall_time_sec`
- `resident_fastpath_module_skips`

Primary evidence for packed:
- `packed_dispatch_wait_wall_time_sec`
- `packed_dispatch_wait_calls`
- `packed_demand_expert_blocks`
- `packed_demand_token_assignments`

Current judgment:
- this is the main problem today
- `Qwen` improved when only the runtime path changed
- `DeepSeek` is still dominated by packed dispatch/wait cost

Relevant code paths:
- `finemoe/models/modulelist_runtime.py`
- `finemoe/runtime/model_offload.py`
- `finemoe/models/packed_runtime.py`
- `core/parallel/expert_dispatcher.cpp`

## Simulator Interpretation

- The simulator is still useful, but only in a limited role:
  - it is an `idealized structure oracle`
  - it is an `upper-bound generator`
- It is **not** a faithful quantitative runtime predictor.
- In particular, the `Qwen` observation results:
  - `oracle_two_pool_speedup ~16x`
  - `zero_loading_speedup ~18x`
  - `loading_share ~0.94`
  should be read as evidence that:
  - the structural headroom is not obviously exhausted
  - current runtime gains are far below the idealized ceiling
- Those numbers should **not** be written as realistic expected deployment gains.
- The correct way to use the simulator now is:
  - use observation to establish structural headroom and failure of single-cache abstractions
  - use real-machine attribution to explain how much of that headroom is currently lost to runtime realization costs

## Priority

### 1. Observation-First Bottleneck Attribution

Goal:
- Measure whether MoE serving is sufficiently memory-bound to justify another serving abstraction.

Concrete work:
- Run simulator-backed observation experiments on the calibrated fair traces.
- Produce three outputs per model:
  - bottleneck breakdown: compute time vs residual stall
  - upper bounds:
    - compute-only / zero-loading theoretical ceiling
    - oracle single-cache ceiling
    - oracle two-pool ceiling
  - trace structure:
    - reuse-distance distribution
    - working-set statistics
- Use these observations to decide whether the next step should be:
  - improved memory hierarchy
  - tail service/runtime work
  - or a more fundamental pivot away from cache-style thinking
- Keep the interpretation disciplined:
  - simulator outputs are for structural comparison and idealized ceilings
  - real-machine attribution remains the authority for quantitative runtime claims

Why:
- This is the right EuroSys move now.
- It prevents us from over-optimizing a method whose ceiling may already be low.

### 2. Runtime-Calibrated Service Envelope

Goal:
- Replace the current overly hard frontier interpretation with a calibrated tail service model.

Concrete work:
- Keep the same feasibility formulation:
  - `bytes(R_k) + S_tail(k) <= B`
- Redefine `S_tail(k)` as a runtime-calibrated service envelope instead of a raw hard-max burst.
- Use runtime-observed budgets and runtime-observed feasibility boundaries to calibrate the envelope.
- Treat percentile-like knobs only as calibration internals, not as the public paper method.

Why:
- This keeps the method principled.
- It avoids falling back to ratio tuning or heuristic sweeping.
- It is a better EuroSys story than "we changed max to P95".

### 3. Utility-Per-Byte Resident Ranking

Goal:
- Improve resident quality without introducing hand-tuned per-layer heuristics.

Concrete work:
- Replace plain frequency-only ranking with a `stall_reduction_per_byte` style score.
- Keep bytes in the ranking objective so packed and modulelist models share one metric.
- Do not introduce manual layer weights such as "early layers = 1.5".

Why:
- This stays aligned with the budgeted systems formulation.
- It is easier to defend than weighted-frequency heuristics.

### 4. Batch-Aware Tail Coalescing

Goal:
- Make `demand-only tail fallback` cheaper under batch traffic.

Concrete work:
- Coalesce per-step tail demand into a batch-level union.
- Reuse grouped demand metadata on repeated decode steps when possible.
- Keep this strictly as coalesced demand service, not speculative prefetch.

Why:
- This is the most likely runtime change to recover real throughput.
- It stays fully within the current paper story.

Immediate implementation note:
- for `Qwen/modulelist`, the next concrete step is to replace per-expert subtree demand service with grouped demand-lane service so one lane activation can cover multiple demand experts in the same layer-step
- for `DeepSeek/packed`, the next concrete step is to reduce grouped tail wait/sync boundaries and reuse dispatch metadata more aggressively

### 5. Resident Fast Path

Goal:
- Make resident hits materially cheaper than ordinary offloaded expert hits.

Concrete work:
- Reduce generic hook bookkeeping on resident hits.
- Make resident metadata queryable without repeated Python-side walks.
- Extend the true fast path to packed experts once budget/accounting are stable.

Why:
- The backbone story only pays off if resident hits are cheap enough.

### 6. Budget Accounting in Core Runtime

Goal:
- Keep budget semantics stable and make them easier to audit.

Concrete work:
- Push more of requested/admitted/clipped accounting into the core runtime.
- Preserve stable JSON payloads:
  - `sparse_budget_bytes`
  - `sparse_budget_source`
  - `resident_registry.*`
- Ensure selector-side and runtime-side budgets cannot drift again.

Why:
- This is necessary for a defensible fairness claim.

## Experiment Queue

### Running Now

- `Qwen fair64` observation sweep on `172`
  - bottleneck attribution
  - compute-only ceiling
  - oracle single/two-pool ceilings
  - reuse / working-set statistics

### Next Immediate Runs

- Run the same observation pipeline on:
  - `DeepSeek fair64`
  - `OLMoE fair64`
  - `Mixtral fair64`
- Use those results to decide whether the next engineering step is:
  - service-envelope recalibration
  - runtime tail coalescing
  - or a larger method pivot

## Not Planned

- No ratio sweeps as the main method.
- No "P95 frontier" story as a standalone paper method.
- No manual layer-weight heuristics.
- No speculative prefetch as the main serving path.
- No controller-heavy mode switching as the paper centerpiece.
- No claiming gains from points that degenerate to `resident = 0`.

## Method Statement

The method should now be read as a first-principles asymmetric service decomposition problem, not as a better cache heuristic.

Problem:

- under GPU memory budget `B`, split MoE traffic into:
  - a `stable resident set` that amortizes future stall
  - a `residual tail` that must remain dynamically serviceable

Method:

- choose resident set `R` to maximize expected future saved stall
- subject to:
  - `bytes(R) + S_tail(R) <= B`
- where `S_tail(R)` is a runtime-calibrated residual tail service envelope, not a raw cache frontier heuristic

Runtime:

- serve `R` through a low-overhead `resident lane`
- serve `E \\ R` through `grouped exact tail service`

Implication:

- the headline novelty is no longer "a better pinning heuristic"
- the headline novelty is:
  - asymmetric resource allocation by `saved stall`
  - serviceability-constrained resident capacity
  - bifurcated runtime realization

Immediate next implementation step:

- finish moving `Qwen/modulelist` from per-expert subtree demand service toward grouped exact tail service
- then attack `DeepSeek/packed` dispatch/wait with the same bifurcated-runtime lens
