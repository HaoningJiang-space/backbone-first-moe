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

Reading:

- strong compute-mass concentration
- strong tail sparsification
- weak exact shape reuse

## Auxiliary Cases

`qwen_current`

- historical non-fair resident plan
- low coverage and weak tail sparsification on the fair trace
- chunk minima stay low as well
- this indicates resident quality matters; backbone regularity is not automatic

`mixtral_adaptive`

- adaptive non-fair resident plan
- moderate assignment coverage and moderate active-expert reduction
- still no evidence for exact shape reuse

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
