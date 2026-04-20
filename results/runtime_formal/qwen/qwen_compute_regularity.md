# Qwen Backbone Compute Regularity

- trace: `Qwen1.5-MoE-A2.7B-Chat~lmsys-chat-1m~64.pkl`
- resident plan: `qwen_lane_mem0p10.json`
- resident set size: `165`
- protocol: trace-only observation; no kernel or runtime-path changes

## Main Numbers

- `backbone_access_coverage = 0.8359`
- `backbone_any_hit_token_coverage = 1.0000`
- `backbone_any_hit_layer_token_coverage = 0.9958`
- `backbone_expert_block_coverage = 0.3874`
- `backbone_flop_coverage = 0.8359`
  - homogeneous-expert proxy; not a hardware FLOP counter
- backbone assignment fraction:
  - per token mean `= 0.8359`
  - per token p50 `= 0.8542`
  - per token p95 `= 0.9271`
- chunk stability with `8`-sequence chunks:
  - assignment fraction per token mean: overall chunk mean `= 0.8359`, chunk min `= 0.7934`
  - active-reduction mean: overall chunk mean `= 0.6720`, chunk min `= 0.6192`
- mean `active_expert_count` reduction after removing backbone assignments `= 0.4426`
- per-step mean expert group size:
  - `all = 16.97`
  - `tail = 3.65`
  - `backbone = 31.96`
- weighted `assignment_shape_reuse_rate = 0.0108`

## Reading

This supports a `compute regularity` story, but not an `exact plan cache` story.

- The backbone covers most routed compute mass.
- Almost every token iteration touches backbone at least once, but that field should be read as an `any-hit` indicator rather than a mass-coverage metric.
- The stronger metric is assignment fraction per token, which is also high.
- The same concentration pattern remains visible across `8`-sequence workload chunks, so this is not only a full-batch aggregate artifact.
- Removing backbone assignments makes the residual tail materially sparser.
- Backbone groups are much larger than tail groups.
- Exact assignment-shape reuse is weak.

Layer highlights:

- strongest concentration:
  - layer `14`: assignment coverage `0.9878`, active-reduction mean `0.8337`
  - layer `15`: assignment coverage `0.9592`, active-reduction mean `0.6408`
  - layer `9`: assignment coverage `0.9459`, active-reduction mean `0.6246`
- weakest concentration:
  - layer `1`: assignment coverage `0.5994`, active-reduction mean `0.2015`
  - layer `0`: assignment coverage `0.6343`, active-reduction mean `0.2611`
  - layer `17`: assignment coverage `0.6678`, active-reduction mean `0.3458`

## Decision

Phase 1 clears the go/no-go bar on `2 / 3` criteria:

- `backbone_flop_coverage > 0.40`: pass
- active-expert reduction `> 0.30`: pass
- `assignment_shape_reuse_rate > 0.50`: fail

Therefore the next implementation step should prioritize:

- coarse grouped reuse
- reusable backbone buffers
- static workspace / stream binding

It should **not** prioritize exact assignment-shape plan caching as the default path.
