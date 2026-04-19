# Qwen Ceiling vs Realization

This note summarizes the current paper-facing interpretation for `Qwen1.5-MoE-A2.7B-Chat`.

Machine-readable companion:

- [qwen_realization_gap_summary.json](qwen_realization_gap_summary.json)
- Repo-contained artifacts:
  - [artifacts/observation/qwen_fair64_observation.json](artifacts/observation/qwen_fair64_observation.json)
  - [artifacts/main_run/qwen_A_mem0p10_lane_long.json](artifacts/main_run/qwen_A_mem0p10_lane_long.json)
  - [artifacts/main_run/qwen_C_mem0p10_lane_long.json](artifacts/main_run/qwen_C_mem0p10_lane_long.json)
  - [artifacts/repeat_1/qwen_A_mem0p10_lane_long.json](artifacts/repeat_1/qwen_A_mem0p10_lane_long.json)
  - [artifacts/repeat_1/qwen_C_mem0p10_lane_long.json](artifacts/repeat_1/qwen_C_mem0p10_lane_long.json)
  - [artifacts/repeat_2/qwen_A_mem0p10_lane_long.json](artifacts/repeat_2/qwen_A_mem0p10_lane_long.json)
  - [artifacts/repeat_2/qwen_C_mem0p10_lane_long.json](artifacts/repeat_2/qwen_C_mem0p10_lane_long.json)

It separates:

- `structure-level headroom` from trace-driven idealized observation
- `implementation-level realization` from fair real-machine runs

The goal is to make one point explicit:

> `Qwen` does not currently look like a low-ceiling case.
> But the most defensible warm steady-state throughput gain is only `~+2.4%`.
> The higher `~+4.8%` mixed/semi-cold result should be read as sensitivity to colder offload regimes, not as the primary paper-facing runtime claim.

## 1. Structure-Level Headroom

Trace-driven observation on `Qwen fair64 @ mem=0.10`:

| Metric | Value |
|---|---:|
| Baseline demand-only | `27.78 tok/s` |
| Oracle single-cache best | `27.78 tok/s` |
| Oracle two-pool best | `445.01 tok/s` |
| Oracle two-pool speedup | `16.02x` |
| Zero-loading speedup upper bound | `18.00x` |
| Loading share | `0.944` |

Interpretation:

- plain single-cache behavior does not help
- idealized two-pool service still shows very large headroom
- most of the simulated end-to-end time is still attributed to loading/stall

Important scope limit:

- these numbers are `idealized upper bounds`
- they should be used as `structural evidence`, not as realistic throughput predictions

## 2. Implementation-Level Realization: Mixed / Semi-Cold Long Workflow

Current fair long-workflow `Qwen 0.10` results on the two-lane modulelist runtime:

| Run | A gen tok/s | C gen tok/s | Gain |
|---|---:|---:|---:|
| Main run | `12.2999` | `13.0684` | `+6.25%` |
| Repeat 1 | `12.4742` | `13.0596` | `+4.69%` |
| Repeat 2 | `12.6322` | `13.0687` | `+3.46%` |

Current range:

- `+3.46%` to `+6.25%`

Current mean:

- `~+4.8%`

All three runs used the same long workflow:

- `total_generated_tokens = 1024`
- `total_prompt_tokens = 9455`
- `batch_size = 8`
- `num_prompts = 64`
- `max_new_tokens = 16`
- `A/C sparse_budget_bytes = 4735686246`
- `resident_capacity = admitted_count = 165`
- `admitted_bytes = 2854748160`
- `clipped = false`

Interpretation:

- this is a `stable positive result`
- it is stronger than the older fair unified-runtime result (`+2.45%`)
- it is still only a `medium-strength throughput result`
- it is best read as `mixed / semi-cold sensitivity`, not as the primary steady-state claim

## 3. Primary Warm Steady-State Result

Warm steady-state protocol:

- fix one `sparse_budget_bytes` from a planning `A` run
- fix one resident plan from that budget
- warm up `A` once and `C` once
- measure `n=5` alternating `A/C` pairs under the same fixed budget and resident plan

Warm steady-state `Qwen 0.10` results:

| Pair | A gen tok/s | C gen tok/s | Gain |
|---|---:|---:|---:|
| Pair 1 | `12.9977` | `13.3220` | `+2.50%` |
| Pair 2 | `12.9939` | `13.3172` | `+2.49%` |
| Pair 3 | `13.0198` | `13.3194` | `+2.30%` |
| Pair 4 | `12.9936` | `13.3078` | `+2.42%` |
| Pair 5 | `13.0089` | `13.2974` | `+2.22%` |

Warm steady-state summary:

- mean gain: `+2.38%`
- median gain: `+2.42%`
- range: `+2.22%` to `+2.50%`

Interpretation:

- this is the most defensible primary quantitative claim for `Qwen`
- it is a `stable positive result`, but clearly weaker than the mixed/semi-cold runs
- this suggests a meaningful fraction of the larger mixed-state gain came from colder offload behavior, not just runtime control-path savings

## 4. What Actually Improved

Representative main run breakdown:

| Metric | A | C | Delta |
|---|---:|---:|---:|
| Total elapsed | `83.25s` | `78.36s` | `-4.90s` |
| `manual_subtree_begin_wall_time_sec` | `48.30s` | `44.09s` | `-4.21s` |
| `modulelist_demand_compute_wall_time_sec` | `56.60s` | `51.11s` | `-5.48s` |
| `resident_fastpath_module_skips` | `0` | `41850` | positive |

Interpretation:

- `resident lane` is real
- `demand lane` is also reducing runtime cost
- the current gain comes primarily from reducing subtree begin/service overhead and tail execution overhead

This supports the current systems interpretation:

> the main problem is not absence of backbone signal;
> the main problem is that unified generic runtimes fail to realize that signal cheaply.

## 5. Paper-Facing Reading

The correct paper-facing claim is:

- `structural claim`
  - `Qwen` still shows substantial idealized headroom under a backbone/two-pool structure
- `systems claim`
  - current real runtimes only realize a small part of that headroom
  - splitting modulelist into `resident lane + demand lane` materially improves realization under fair budget
  - the most defensible steady-state throughput gain is only `~+2.4%`, while mixed/semi-cold sensitivity is higher

What should **not** be claimed:

- that the simulator is a faithful throughput predictor
- that `Qwen` already demonstrates a strong throughput story by itself
- that mixed/semi-cold gains and warm steady-state gains should be averaged into one runtime number

## 6. Implication

The next engineering priority is not more selector tuning.

The next engineering priority is:

1. reduce `Qwen` demand-lane begin/end overhead further
2. continue reducing `DeepSeek` packed `dispatch_wait`
3. use simulator ceilings only as idealized headroom references, always paired with real-machine attribution
