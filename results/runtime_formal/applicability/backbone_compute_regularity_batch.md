# Backbone Compute Regularity Batch Observation

This page summarizes several trace-only backbone regularity probes.

Only `qwen_fair` should be treated as the primary result.
The other rows are auxiliary existence checks or negative controls and use older or non-fair resident plans.

## Primary Case

`qwen_fair`

- resident plan type: `fair_runtime_aligned`
- resident set size: `165`
- assignment fraction per token mean `= 0.8359`
- chunk assignment fraction min `= 0.7934`
- active-expert reduction mean `= 0.4426`
- chunk active-reduction min `= 0.6192`
- `backbone_group_mean = 31.96`
- `tail_group_mean = 3.65`
- `shape_reuse_weighted_mean = 0.0108`
- `backbone_active_set_reuse_weighted_mean = 0.9664`
- `backbone_active_count_reuse_weighted_mean = 0.9664`
- `tail_active_set_reuse_weighted_mean = 0.3293`
- `tail_active_count_reuse_weighted_mean = 0.6586`

Reading:

- strong compute-mass concentration
- strong tail sparsification
- weak exact shape reuse
- very strong coarse backbone reuse
- materially weaker tail active-set reuse than backbone reuse
- this now directly supports:
  - coarse grouped metadata reuse
  - reusable buffer sizing keyed by backbone activity
  - static workspace / stream binding for backbone lane

## Auxiliary Cases

`qwen_current`

- historical non-fair resident plan
- low coverage and weak tail sparsification on the fair trace
- chunk minima stay low as well
- this indicates resident quality matters; backbone regularity is not automatic
- coarse reuse remains non-zero, but the backbone/tail separation is much weaker than `qwen_fair`

`mixtral_adaptive`

- adaptive non-fair resident plan
- moderate assignment coverage and moderate active-expert reduction
- still no evidence for exact shape reuse
- both backbone and tail active-set reuse are high, so the split is less clean than `qwen_fair`
- this suggests `Mixtral` may support reusable grouped plans, but not necessarily a strong backbone-vs-tail specialization story

`olmoe_adaptive`

- adaptive non-fair large resident plan
- almost all assignments become resident
- useful mainly as an upper-end saturation reference, not as a fair comparison point

`deepseek_zero`

- fair zero-resident negative control
- all backbone regularity metrics collapse to zero, as expected

## Decision

The batch observation reinforces the same engineering conclusion:

- keep `qwen_fair` as the main evidence
- treat backbone as a compute-mass concentrator and tail sparsifier
- do not prioritize exact assignment-shape plan caching
- prioritize coarse grouped reuse, reusable buffers, and static workspace binding
- keep `Qwen/modulelist` as the Phase 2 target because it shows the cleanest `high-backbone-reuse + lower-tail-reuse` split
