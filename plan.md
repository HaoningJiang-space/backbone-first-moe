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

## Current A/B/C Decomposition

The most important missing experiment was a clean `B`:

- `A`: unified baseline, no resident
- `B`: same fair resident set and same budget as `C`, but with `disable_backbone_lane_split=true`
- `C`: backbone-first two-lane runtime with the same resident set and budget

The first completed `A/B/C` pair on `Qwen @ 0.10` is:

- `A = 13.0264 tok/s`
- `B = 13.2989 tok/s`
- `C = 13.3310 tok/s`

Current pair-1 decomposition:

- `A -> B = +2.09%`
- `B -> C = +0.24%`
- `A -> C = +2.34%`

Interpretation:

- the clean `B` mode is now implemented and runnable
- the first pair suggests that the current value comes primarily from `backbone resident/materialization`
- the incremental value of the current two-lane realization is small in this first pair
- therefore the paper should not currently claim a strong `two-lane` win by default

Important scope limit:

- this is still a `preliminary decomposition`, not the final `n`-pair result
- do not overwrite the primary warm steady-state claim with this single pair
- use it to guide implementation and attribution priorities:
  - `A -> B` currently looks like the stronger effect
  - `B -> C` still needs the full repeated run before it can be treated as stable

Held-out short `A/B/C` is now complete under the runtime-feasible `top165` regime on shard-`1` prompts:

- pair means:
  - `A = 5.2144 tok/s`
  - `B = 5.3792 tok/s`
  - `C = 5.3577 tok/s`
- mean gains:
  - `A -> B = +3.16%`
  - `B -> C = -0.40%`
  - `A -> C = +2.75%`

Interpretation:

- this is currently the cleanest repeated held-out runtime decomposition
- `backbone resident/materialization` is now a stable demonstrated gain
- current `two-lane` realization does **not** add a stable extra win on top of `B`
- therefore the paper should presently treat:
  - `A -> B` as the main quantitative systems result
  - `B -> C` as the current realization-gap evidence

Supporting `B/C` breakdown on a matched short observation run:

- `B = 5.7355 tok/s`
- `C = 5.6999 tok/s`
- `B -> C = -0.62%`

Field-level comparison:

- `tail_group_begin_wall_time_sec`
  - `B = 26.763s`
  - `C = 26.732s`
  - almost unchanged
- `modulelist_demand_compute_wall_time_sec`
  - `B = 28.400s`
  - `C = 28.164s`
  - only `-0.236s`
- `C` adds resident-lane cost:
  - `resident_compute = 0.277s`
  - `resident_gather = 0.024s`
  - `resident_merge = 0.051s`

Current best read:

- the present `B -> C` gap is not a large hidden tail-side win waiting to be unlocked
- under the current implementation, `C` only saves a small amount on the tail side relative to `B`
- and that small saving can be canceled by the resident-lane realization cost itself
- therefore `two-lane specialization` should currently be treated as:
  - an implementation hypothesis that still needs work
  - not as the already-proven main source of gain

## Backbone-Specific Filter

Use the following filter before promoting any optimization into the main story.

An optimization should count as `backbone-specific` only if it satisfies most of these:

1. it uses backbone's distinct structural signals:
   - high coarse active-set reuse
   - large backbone group size
   - strong compute-mass concentration
2. it depends on backbone discovery:
   - without the discovered backbone set, the optimization is not well-defined
3. it treats backbone and tail differently:
   - stable backbone path
   - dynamic tail path
4. it exploits `stability` rather than only `frequency`:
   - reuse / repeated active sets / reusable grouped execution state
   - not merely "these experts are often accessed"

Practical consequence:

- `generic runtime hygiene` is not backbone-specific even if it helps `C`
- `backbone-guided heuristics` are weaker than true backbone-specific mechanisms
- only optimizations that genuinely rely on backbone stability / group structure should be promoted into the main thesis

### What Currently Qualifies

Strong candidates:

- `backbone resident/materialization`
  - requires the discovered resident backbone set
- `backbone-aware grouped realization`
  - grouped backbone-serving path
  - explicit separation from dynamic tail service
- `grouped backbone dispatch / grouped backbone GEMM`
  - if it actually relies on backbone group-size and active-set stability
- `backbone-specialized fused or static execution`
  - only if it is keyed by repeated backbone structure and evaluated as `C -> D`

### What Does Not Currently Qualify As Main Story

Do not promote these as the next headline result:

- generic metadata caching
- generic output-buffer reuse
- generic `begin_group` or `demand compute` micro-optimizations
- quantization by itself
- pruning by itself

Reason:

- these can be useful engineering tools
- but by themselves they do not yet demonstrate a backbone-specific systems insight
- and several of them have already shown only weak gains in our current runtime

### Interpretation Rule

Use this ladder when evaluating any new implementation:

- if it helps `A`, `B`, and `C` similarly:
  - runtime hygiene
- if it improves `B -> C` under the same resident set:
  - backbone-aware realization evidence
- if it additionally enables a clean `C -> D` gain:
  - backbone-specific compute-regularity evidence

## Fresh Real-Machine Backbone Validation

This question is now answered with **fresh real-machine evidence**, not only with previously collected traces.

Protocol:

- regenerate `4` non-overlapping routed trace shards on the real machine
- each shard uses fresh prompts and real model execution
- each shard selects its own resident backbone independently
- then test both:
  - cross-shard trace-level transfer
  - held-out runtime transfer

### Real-Machine Shard-Level Existence

On `Qwen`, with `4 x 16` fresh-prompt shards at `mem=0.10`:

- per-shard resident counts:
  - `405, 413, 447, 425`
- native assignment-fraction-per-token means:
  - `0.4219, 0.3998, 0.4141, 0.3916`
- native active-reduction means:
  - `0.3533, 0.3506, 0.3710, 0.3470`

Interpretation:

- each fresh shard yields a large, non-degenerate resident backbone
- each shard shows substantial native routed-assignment coverage
- therefore `backbone exists` is now supported by fresh real-machine routed traces

### Real-Machine Cross-Shard Stability

Across all train-shard to test-shard transfers:

- resident-set Jaccard:
  - mean `0.3054`
  - min `0.2547`
  - max `0.3432`
- held-out retained assignment fraction:
  - mean `0.9220`
  - min `0.8740`
  - max `0.9916`
- held-out retained active-reduction fraction:
  - mean `0.9513`
  - min `0.8887`
  - max `1.0028`

Interpretation:

- exact resident identity overlap is only moderate
- but the **functional backbone** transfers strongly across fresh shards
- what is stable is not a fixed exact expert-id set
- what is stable is a backbone that preserves most of the useful routed coverage and tail sparsification on held-out prompts

### Held-Out Real-Machine Runtime Transfer

Trace-level transfer is now also backed by a real runtime check.

Held-out runtime protocol:

- prompts: shard `1`
- `A`: no resident backbone
- `B_native`: resident set selected from shard `1`
- `B_transfer`: resident set selected from shard `0`
- both resident files are trimmed to the same runtime-feasible `top165` regime

Measured throughput:

- `A = 5.1729 tok/s`
- `B_native = 5.3580 tok/s`
- `B_transfer = 5.3392 tok/s`

Transfer retention:

- `A -> B_native = +3.58%`
- `A -> B_transfer = +3.22%`
- retained gain fraction = `0.8984`

Interpretation:

- a backbone learned on shard `0` preserves about `90%` of the native runtime gain on held-out shard `1`
- therefore `backbone is stable` is no longer just a trace-side claim
- it now has **held-out real-machine runtime transfer** support

Most important conclusion:

- `backbone exists and is stable` should now be treated as a proven statement for `Qwen`
- this conclusion is stronger than the current `two-lane` realization claim
- the current uncertainty is no longer whether backbone is real
- the current uncertainty is how to realize backbone cheaply enough at runtime

## Conservative Backbone Roadmap

This is the evidence-aligned roadmap for the next phase.

The main thesis stays:

- `backbone` is the core insight
- `backbone` is not only a memory object
- `backbone` is also a `compute regularity` object

But the roadmap must distinguish clearly between:

- what is already proven
- what has been tried and found weak
- what is still only a plausible next step

### What Is Already Proven

#### Level 1: Backbone Discovery

What is proven:

- stable backbone exists on real routed workloads
- stable backbone also transfers across fresh real-machine prompt shards
- held-out runtime gain transfer is strong in the runtime-feasible resident regime
- backbone concentrates routed compute mass
- backbone materially sparsifies the residual tail

Current strongest evidence:

- `assignment_fraction_per_token_mean = 0.8359`
- `active_reduction_mean = 0.4426`
- `backbone_group_mean = 31.96`
- `tail_group_mean = 3.65`
- `weighted_backbone_active_set_reuse_rate = 0.9664`
- `shape_reuse_weighted_mean = 0.0108`

Interpretation:

- backbone is a strong `compute-mass concentrator`
- backbone is a strong `tail sparsifier`
- backbone is stable in a functional sense even when exact resident-set overlap is only moderate
- backbone is **not** currently a strong `exact assignment-shape cache key`

#### Level 2: Backbone Resident / Materialization

What is proven:

- under the same fair resident set and budget, resident backbone materialization already gives measurable value

Current cleanest evidence:

- first `A/B/C` pair:
  - `A -> B = +2.09%`
  - `B -> C = +0.24%`
  - `A -> C = +2.34%`

Interpretation:

- the strongest currently demonstrated gain comes from `backbone resident/materialization`
- the paper can safely claim that backbone is valuable as a resident serving object

### What Has Already Been Tried And Looks Weak

The following should **not** be treated as the next headline optimization path:

- exact assignment-shape plan cache
- metadata-cache-first story
- output-buffer-cache-first story

Why:

- exact shape reuse is weak:
  - `shape_reuse_weighted_mean = 0.0108`
- metadata reuse was implemented and helped only slightly
- output-buffer reuse was implemented and helped only slightly
- both are useful as hygiene, but neither currently looks like the main missing realization win

Practical implication:

- do not frame the next phase as:
  - `cache more metadata`
  - `cache more buffers`
  - `exact plan reuse`

These are secondary optimizations, not the current strategic lever.

### What The Current B/C Evidence Says

Matched short-run `B/C` breakdown:

- `B = 5.7355 tok/s`
- `C = 5.6999 tok/s`
- `B -> C = -0.62%`

Key fields:

- `tail_group_begin_wall_time_sec`
  - `B = 26.763s`
  - `C = 26.732s`
  - almost unchanged
- `modulelist_demand_compute_wall_time_sec`
  - `B = 28.400s`
  - `C = 28.164s`
  - only `-0.236s`
- `C` adds resident-lane cost:
  - `resident_compute = 0.277s`
  - `resident_gather = 0.024s`
  - `resident_merge = 0.051s`

Interpretation:

- the current `B -> C` gap is **not** a large hidden win that is already visible in tail-side counters
- current `C` saves only a small amount relative to `B`
- that small saving can be canceled by the realization cost of the resident lane itself

This means:

- `two-lane specialization` is still a plausible systems direction
- but it is **not yet** a strong demonstrated gain
- therefore it should be treated as a redesign target, not as an already-proven headline result

### The Next Real Engineering Target

The next target should be:

- `backbone-aware grouped realization redesign`

Not:

- more generic runtime hygiene
- more metadata caching
- more exact plan caching

The concrete goal is:

- keep the proven value of `backbone resident/materialization`
- reduce the extra realization cost introduced by the current backbone lane
- do so without requiring aggressive semantic changes

The current systems problem is therefore:

> how to realize a resident backbone through a grouped, low-overhead path
> that is meaningfully cheaper than the current unified resident-serving path

### Progressive, Evidence-Aligned Levels

#### Level 3: Backbone-Aware Grouped Realization

This is the next recommended implementation target.

Desired properties:

- grouped backbone-serving path
- reduced resident gather / merge / coordination overhead
- reusable backbone workspace and stable execution state
- no dependence on exact assignment-shape reuse
- explicit use of backbone stability / group structure, not just frequent-expert caching

This level should be evaluated by:

- clean `B -> C` repeated runs
- `B/C` matched short-run breakdowns
- explicit resident-lane cost attribution

Success criterion:

- `B -> C` becomes a stable positive gain, not just a single-pair fluctuation

#### Level 4: Optional Backbone-Specialized Fused / Static Execution

This remains a possible heavy endpoint, not the current default path.

It becomes worth pursuing only if:

- Level 3 succeeds
- the remaining resident-lane overhead is still significant
- grouped backbone execution clearly dominates the remaining realization gap

This level includes ideas such as:

- more aggressive grouped backbone dispatch
- grouped backbone GEMM if it is actually keyed by backbone grouping
- static workspace / stream specialization
- backbone-specialized fused execution

Important scope boundary:

- grouped GEMM / fusion only belong in the main paper path if they are implemented as `backbone-specific`
- they do **not** qualify merely because they speed up MoE generically
- do not attach speculative gain numbers to them before measuring `C -> D`

Important rule:

- do not claim fused-kernel gains before they are implemented and measured
- treat them as `future heavy endpoints` until real data exists

### What The Paper Should Say Today

The strongest current version is:

1. `backbone discovery` is real
2. `backbone resident/materialization` is already valuable
3. `backbone` also exhibits strong compute-regularity structure
4. current `two-lane` realization is not yet strong enough to be the main demonstrated source of gain
5. the next systems challenge is to build a cheaper `backbone-aware grouped realization`

In other words:

- the current main result is not:
  - `two-lane already wins big`
- the current main result is:
  - `backbone exists`
  - `backbone resident already helps`
  - `backbone exposes a new realization problem that current runtimes do not yet solve well`

### Immediate Next Step

Do next:

- keep running clean `A/B/C` repeated experiments
- treat `A -> B` as the current strongest proven gain
- use `B/C` breakdown to guide a resident-lane redesign
- only after that, decide whether a heavier fused/static backbone path is justified

Do not do next:

- present metadata reuse as the next major win
- present buffer reuse as the next major win
- present fused kernels as already-validated backbone value

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

Current implementation read:

- `metadata reuse` and `output-buffer reuse` were both implemented and measured
- both produced only small gains
- therefore they should be treated as:
  - useful subcomponents
  - not the next headline optimization track
- the next serious attempt on this line should be:
  - a resident-aware grouped backbone realization
  - or a true `C -> D` grouped-backbone compute prototype

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

- for `Qwen/modulelist`, do **not** keep iterating on metadata-cache-first or buffer-cache-first ideas as the main path
- instead, the next implementation must satisfy the `backbone-specific filter` above
- preferred order:
  - first, keep the clean repeated `A/B/C` authority current
  - second, attempt a resident-aware grouped backbone realization that is measured as `B -> C`
  - third, only if that succeeds, attempt a true `C -> D` grouped-backbone compute prototype
- do not start with exact plan-cache reuse
- treat any generic `A` and `C` co-speedup as supporting evidence only
- only promote work into the main story if it:
  - enlarges the `B -> C` gap
  - or enables a credible `C -> D` backbone-compute result
- after that, revisit `DeepSeek/packed` only if the same compute-regularity lens produces a plausible grouped service prototype
