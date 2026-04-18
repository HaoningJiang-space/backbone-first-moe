# Current Plan

## Done

- Added a runtime-visible resident registry in [finemoe/runtime/model_offload.py](/home/abc/Placement/Efficient_AI/backbone-first-moe_git/finemoe/runtime/model_offload.py).
- Resident loading now preserves prefix order from the resident JSON instead of collapsing it into a sorted set.
- Runtime now records resident admission metadata:
  - `requested_count`
  - `admitted_count`
  - `requested_tensor_count`
  - `admitted_tensor_count`
  - `clipped`
  - `selection_rule`
  - `source_file`
- Runtime evaluation output now exposes this state in [finemoe/backbone/runtime_eval.py](/home/abc/Placement/Efficient_AI/backbone-first-moe_git/finemoe/backbone/runtime_eval.py).
- Added resident-registry tests in [tests/test_resident_registry.py](/home/abc/Placement/Efficient_AI/backbone-first-moe_git/tests/test_resident_registry.py).
- Added a modulelist resident fast path:
  - resident expert subtrees are marked once after pinning
  - generic pre/post hook bookkeeping now bypasses those marked resident subtrees
- Added grouped modulelist expert dispatch:
  - `Qwen` and `OLMoE` now dispatch active experts by sorted expert blocks
  - removed the hot-path `one_hot + torch.where` loop from modulelist MoE execution
- Added grouped packed expert dispatch:
  - `Mixtral` / `DeepSeek` now build active packed expert assignments once per step
  - removed the hot-path per-expert `torch.where(router_mask[:, expert_idx])` lookup
- Aligned packed model `device` semantics with runtime execution device:
  - packed runtime models now expose the configured runtime device to `generate()`
  - removed the `input_ids on cuda / model on cpu` warning on packed tiny probes
- Added an explicit packed resident fast-path contract:
  - runtime now assigns `resident_fastpath_local_expert_ids` per layer
  - packed execution consumes that explicit set instead of guessing from container type
  - `ModuleList`-backed packed paths now safely fall back to demand execution

## Next

### 1. Resident Fast Path

Goal:
- Make resident experts a true runtime fast path instead of letting them share as much bookkeeping as ordinary offloaded experts.

Concrete work:
- Separate resident-hit handling from generic `begin()/end()` hook logic.
- Make resident metadata queryable without walking Python module state.
- Add a true resident-hit fast path for packed experts, not just grouped demand dispatch.

Why:
- The story stays the same: resident backbone is the main source of throughput gain.
- This is a runtime realization improvement, not a new selector heuristic.

### 2. Demand Tail Coalescing

Goal:
- Keep `demand-only tail fallback`, but make it cheaper under batch traffic.

Concrete work:
- Reuse grouped packed/modulelist demand metadata across repeated warm-path decode steps when possible.
- Reuse warm-path artifacts to reduce repeated setup cost.

Why:
- This improves throughput without bringing speculative prefetch back into the critical path.
- It matches the current paper story and EuroSys positioning.

### 3. Resident Admission in Core Runtime

Goal:
- Move requested/admitted/clipped/slack semantics closer to the core runtime and eventually expose them from C++.

Concrete work:
- Surface resident admission stats from the native engine if possible.
- Replace Python-side approximation with runtime-native accounting.
- Keep result JSON stable while changing the source of truth.

Why:
- This is the right C++ work for the current stage.
- It strengthens the system abstraction instead of adding controller logic.

### 4. Packed Runtime Cleanup

Goal:
- Reduce overhead on packed MoE paths (`Mixtral`, `DeepSeek`) without changing the serving abstraction.

Concrete work:
- Clean synthetic-slice lookup and registration paths.
- Reduce packed hook overhead.
- Finish full-model runtime validation when assets are available.

Why:
- Packed models already show positive `A -> C` direction.
- The current gap is runtime efficiency and asset completeness, not selector design.

## Not Planned

- No new ratio sweeps as the main method.
- No speculative prefetch as the main serving path.
- No mode controller as the paper centerpiece.

The method stays:
- utility-ranked backbone
- burst-aware feasible resident prefix
- resident backbone + demand-only tail fallback
