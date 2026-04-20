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
  - mixed / semi-cold modulelist two-lane runtime:
    - long-workflow gains observed at `+6.25%`, `+4.69%`, `+3.46%`
    - current mean is `~+4.8%`
    - this is now best treated as regime sensitivity, not as the primary runtime claim
  - warm steady-state modulelist two-lane runtime:
    - `n=5` alternating-pair gains observed at `+2.50%`, `+2.49%`, `+2.30%`, `+2.42%`, `+2.22%`
    - current mean is `~+2.38%`
    - this is the most defensible primary throughput claim today
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
  - modulelist `resident lane + demand lane` raises the old unified fair runtime from `+2.45%` to `~+2.4%` under warm steady-state and to `~+4.8%` under mixed/semi-cold runs
- We still do not know whether MoE serving is fundamentally memory-bound enough to justify another round of memory-hierarchy design.
- The main unanswered question is now:
  - how much of the end-to-end time is actually load stall
  - what the realistic zero-loading / oracle ceiling looks like
  - whether current weak gains come from a bad method or a low memory-optimization ceiling

## Strategic Read

- `Qwen` still supports a real structural backbone story, but the strongest current evidence is no longer a pure memory story.
- Warm steady-state throughput gains remain small enough that another round of cache-policy tuning is unlikely to become the headline result.
- The strongest new evidence is:
  - backbone concentrates routed compute mass
  - backbone materially sparsifies the residual tail
  - exact assignment-shape reuse is weak
  - coarse backbone active-set reuse is very strong on `Qwen fair`
- Therefore backbone should now be interpreted primarily as:
  - a `compute-mass concentrator`
  - a `tail sparsifier`
  - not as a strong `exact plan-cache key`

Implication:
- same-resource throughput can still improve
- but the improvement must come from `execution efficiency`, not from reduced mathematical compute
- without quantization, reduced top-k, pruning, or other semantic changes, `huge` gains are unlikely
- the realistic systems target is therefore:
  - medium gains from more regular grouped execution
  - not giant gains from cache heuristics alone

## Backbone-Centric Reading

The core contribution must stay centered on `backbone`, not on generic runtime hygiene.

What counts as the main story:

- `backbone discovery`
  - stable routed backbone exists on real workloads
  - the selector is runtime-calibrated rather than static / hand-tuned
- `two-lane architecture`
  - backbone lane and tail lane should not share one unified generic service path
- `backbone-first > unified`
  - under the same fair budget and the same resident set, separating backbone from tail is better than treating every expert access uniformly

What does **not** count as the main story:

- generic `begin_group` speedups
- generic `demand compute` micro-optimizations
- generic metadata caching that helps both `A` and `C` roughly equally

These optimizations are still useful, but only as:

- supporting attribution
- implementation hygiene
- evidence that the current runtime can realize the backbone signal more cheaply

They should not be promoted into the paper thesis by themselves.

Practical rule:

- if a change speeds up both `A` and `C` similarly, it is `runtime hygiene`
- if a change disproportionately improves `C` relative to `A`, it is evidence for `backbone-first realization`
- if a change only makes sense after splitting `backbone` from `tail`, it is a candidate `backbone-specific systems contribution`

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
  - but the simulator is not a quantitative predictor of warm steady-state throughput
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

### 1A. Backbone Compute Regularity Observation

Goal:
- Test whether the extracted backbone is valuable as a `compute regularity` object, not only as a memory object.

Phase 1:
- analysis only; do not change kernels or model math yet
- compute the following from routed traces plus a fixed resident plan:
  - `backbone_access_coverage`
  - `backbone_token_coverage`
  - `backbone_flop_coverage`
  - `backbone_expert_block_coverage`
  - `active_expert_count_before_after_backbone`
  - `group_size_before_after_backbone`
  - `assignment_shape_reuse_rate`
- keep the interpretation disciplined:
  - for homogeneous-expert models such as current `Qwen`, `backbone_flop_coverage` is an approximate proxy derived from routed assignments rather than a direct hardware FLOP counter
  - dispatch-overhead breakdown remains a separate runtime-profiling question, not a trace-only claim

Success criteria for moving forward:
- at least two of the following should hold on the analyzed fair workload:
  - `backbone_flop_coverage > 0.40`
  - mean `active_expert_count` reduction after removing backbone assignments `> 0.30`
  - `assignment_shape_reuse_rate > 0.50`

Why:
- if these signals are weak, backbone remains primarily a memory story
- if these signals are strong, backbone becomes a plausible `compute regularity` object and justifies execution-plan work

Phase 2:
- execution-plan reuse, still without quantization or model changes
- target items:
  - prebuilt dispatch metadata for backbone lane
  - backbone-specific reusable buffers
  - static stream / workspace binding
  - grouped plan cache for `modulelist` and `packed`
- first `Qwen @ 0.10` observation result:
  - `backbone_flop_coverage ~= 0.836` under the homogeneous-expert proxy
  - mean active-expert reduction after removing backbone assignments `~= 0.443`
  - weighted `assignment_shape_reuse_rate ~= 0.011`
- interpretation:
  - backbone clearly concentrates compute mass and sparsifies the residual tail
  - exact assignment-shape reuse is weak
- therefore Phase 2 should prioritize `coarse grouped reuse + reusable buffers + static workspace binding`
- do **not** treat exact plan-cache reuse as the default implementation path
- updated `Qwen fair` read with coarse grouped reuse:
  - weighted exact assignment-shape reuse `~= 0.011`
  - weighted backbone active-set reuse `~= 0.966`
  - weighted backbone active-count reuse `~= 0.966`
  - weighted tail active-set reuse `~= 0.329`
- interpretation:
  - the backbone is not a good `exact plan-cache key`
  - it **is** a strong key for `coarse grouped metadata reuse`, `reusable buffer sizing`, and `static workspace binding`

Phase 3:
- persistent backbone lane
- only after Phase 1 and Phase 2 justify it
- target items:
  - backbone-specialized dispatch / merge path
  - persistent backbone execution lane

Immediate next step:
- run a trace-driven observation pass first on `Qwen @ 0.10` with the current fair resident plan
- only promote this into a systems implementation track if the observation numbers are strong enough
- before writing the first Phase 2 prototype, add one runtime-side observation pass on `Qwen/modulelist` to separate:
  - dispatch metadata build
  - buffer / workspace preparation
  - gather / scatter / merge
  from pure expert compute
- this is the missing bridge between `trace regularity` and `execution-plan reuse`

Current observation read:
- `Qwen fair` clears the go/no-go bar because:
  - assignment mass on backbone is high
  - residual active-expert count drops materially
  - backbone groups are much larger than tail groups
  - backbone active-set and active-count reuse are both very high
- `Mixtral adaptive` shows a weaker but still non-trivial version of the same pattern
  - backbone active-set reuse is also high there
  - but tail active-set reuse stays similarly high
  - so it does **not** support as clean a backbone-vs-tail split as `Qwen fair`
- `DeepSeek zero-resident` behaves as the expected negative control
- this is enough to justify a compute-regularity implementation track
- it is **not** enough to justify exact plan-cache work as the first implementation step

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

### 4A. Backbone-Guided Compute Regularization

Goal:
- convert backbone structure into faster same-resource execution without changing model semantics

Concrete work:
- keep the mathematical compute unchanged
- improve how backbone compute is executed:
  - coarse grouped metadata reuse for backbone lane
  - reusable backbone-specific buffers
  - static workspace / stream binding
  - grouped backbone lane separated from sparse dynamic tail service
- explicitly avoid overfitting to exact assignment patterns

Why:
- this is the highest-confidence path supported by current observations
- it uses the part of backbone regularity that is actually strong:
  - mass concentration
  - larger grouped compute
  - residual tail sparsification
- it does not rely on a reuse signal that is currently weak

Contribution boundary:

- this track is valid only if it remains clearly `backbone-guided`
- generic speedups that are equally available to unified serving do not strengthen the core claim
- therefore every implementation step on this track should be evaluated with an explicit `A/B/C(/D)` ladder rather than only as a raw throughput increase

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

### Backbone Value Ladder

The paper-facing ablation stack should be read as:

- `A`
  - unified baseline, no resident backbone
- `B`
  - unified / generic runtime with the same discovered backbone resident set
  - this isolates the value of `backbone discovery + resident materialization`
- `C`
  - backbone-first two-lane runtime with the same resident set
  - this isolates the value of `backbone-aware realization`
- `D` (optional)
  - backbone-guided compute regularization on top of `C`
  - this isolates the value of treating backbone as a compute-regularity object

Interpretation discipline:

- `A -> B` answers whether backbone itself has usable systems value
- `B -> C` answers whether two-lane architecture realizes that value better than a unified runtime
- `C -> D` answers whether backbone also supports same-resource compute-path gains

Current evidence status:

- `A -> C` is positive on `Qwen`
- `A -> C` alone is not enough to cleanly attribute where the gain comes from
- therefore future evaluation should explicitly move toward `A/B/C(/D)` rather than treating all runtime work as one monolithic delta

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
- No betting the main implementation path on exact assignment-shape plan caches.
- No promising giant throughput gains under exact semantics without evidence of real compute reduction.

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
  - backbone-guided compute regularization under the same resource budget

Limits:

- if semantics are held fixed:
  - same model
  - same top-k
  - no quantization
  - no pruning / skipping
- then backbone can make execution more efficient, but cannot create giant gains by itself
- giant gains would require one of:
  - real compute reduction
  - more aggressive router-aware batching / scheduling
  - a changed serving objective that trades latency fairness for throughput
- those are possible future directions, but are not the default claim of the current plan

Immediate next implementation step:

- for `Qwen/modulelist`, start with a minimal Phase 2 prototype:
  - reusable backbone buffers
  - coarse grouped metadata reuse
  - static workspace binding
- do not start with exact plan-cache reuse
- treat any generic `A` and `C` co-speedup as supporting evidence only
- only promote work into the main story if it:
  - enlarges the `B -> C` gap
  - or enables a credible `C -> D` backbone-compute result
- after that, revisit `DeepSeek/packed` only if the same compute-regularity lens produces a plausible grouped service prototype
