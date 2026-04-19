# Current Plan

## Ground Truth

- `fresh clone` on `172` is now end-to-end runnable for `Qwen`, `OLMoE`, and `DeepSeek` smoke.
- Fair-budget runtime now uses `free_device_memory_ratio` instead of total-memory ratio.
- Selector now supports `--sparse-budget-bytes` and can be calibrated from the runtime `A` run.
- Zero-resident cases now explicitly degenerate to baseline outputs instead of pretending to run a meaningful `C`.

## Current Fair Results

- `Qwen @ 0.10`
  - `A = 14.0118 tok/s`
  - `C = 14.3550 tok/s`
  - `+2.45%`
  - `resident_admitted_count = 165`
  - `resident_registry.budget_bytes = sparse_budget_bytes = 4735686246`
- `Qwen @ 0.07`
  - runtime-calibrated selector returns `resident = 0`
  - `C` degenerates to baseline
- `OLMoE @ 0.045 / 0.05`
  - runtime-calibrated selector returns `resident = 0`
  - `C` degenerates to baseline
- `DeepSeek`
  - fair-budget rerun is in progress

## Main Diagnosis

- The old gains were partly inflated by unfair or weakly calibrated budget accounting.
- The new fair pipeline is more credible but much more conservative.
- The main problem is no longer "can we pin experts?".
- The main problem is:
  - resident bytes are underutilized in some feasible regimes
  - the tail service model is too conservative
  - resident hits still do not realize enough runtime advantage

## Priority

### 1. Finish Fair Reruns

Goal:
- Close the current experiment queue on the fair workflow before changing the method again.

Concrete work:
- Finish `DeepSeek` fresh-clone fair rerun.
- Keep the `Qwen @ 0.10` fair result as the current reference point.
- Treat `OLMoE @ 0.045 / 0.05` as threshold-finding evidence, not final positive points.

Why:
- We need one clean baseline before claiming any further runtime gain.

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

- `DeepSeek` fresh-clone fair rerun on `172`

### Next Immediate Runs

- If `DeepSeek` also degenerates to zero resident, keep the result as-is and do not fake a positive `C`.
- After the current reruns finish, do threshold-finding runs only where the calibrated selector starts producing nonzero resident prefixes.
- For `OLMoE`, raise the budget gradually above `0.05` to identify the first nonzero fair regime.

## Not Planned

- No ratio sweeps as the main method.
- No "P95 frontier" story as a standalone paper method.
- No manual layer-weight heuristics.
- No speculative prefetch as the main serving path.
- No controller-heavy mode switching as the paper centerpiece.
- No claiming gains from points that degenerate to `resident = 0`.

## Method Statement

The method remains:

- runtime-calibrated budget
- utility-aware resident backbone
- service-envelope-constrained feasible prefix
- resident backbone + coalesced demand-only tail fallback
