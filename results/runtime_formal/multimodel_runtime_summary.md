# Multi-Model Runtime Summary

Current paper-facing split:
- `Qwen` / `OLMoE`: strong positive cases
- `DeepSeek-V2-Lite`: weak positive / boundary case
- `Mixtral`: applicability / boundary case; tiny packed-runtime probe is positive, but full-model runtime is still missing

## Runtime Table

| Model | Role | mem | A gen tok/s | C gen tok/s | gain | C resident |
|---|---|---:|---:|---:|---:|---:|
| Qwen1.5-MoE-A2.7B-Chat | strong_positive | 0.07 | 2.9266 | 3.2331 | +10.5% | 275 |
| Qwen1.5-MoE-A2.7B-Chat | strong_positive | 0.10 | 3.0946 | 3.6162 | +16.9% | 464 |
| OLMoE-1B-7B-0924 | strong_positive | 0.07 | 0.3067 | 2.5799 | +741.2% | 1023 |
| OLMoE-1B-7B-0924 | strong_positive | 0.10 | 0.3157 | 1.7778 | +463.2% | 1023 |
| DeepSeek-V2-Lite | weak_positive_boundary | 0.07 | 0.1597 | 0.1782 | +11.6% | 186 |
| DeepSeek-V2-Lite | weak_positive_boundary | 0.10 | 0.1583 | 0.1776 | +12.2% | 174 |

## Applicability Table

| Model | mem | top20 access cov. | top20 stall cov. | knee ratio | frontier ratio | horizon | slack util. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen1.5-MoE-A2.7B-Chat | 0.07 | 0.771 | 0.384 | 0.369 | 0.910 | 13 | 1.000 |
| Qwen1.5-MoE-A2.7B-Chat | 0.10 | 0.771 | 0.320 | 0.279 | 0.832 | 13 | 0.000 |
| DeepSeek-V2-Lite | 0.05 | 0.365 | 0.327 | 0.353 | 0.000 | 14 | 2.723 |
| DeepSeek-V2-Lite | 0.07 | 0.365 | 0.324 | 0.378 | 0.000 | 14 | 1.946 |
| DeepSeek-V2-Lite | 0.10 | 0.365 | 0.314 | 0.366 | 0.000 | 14 | 1.361 |

## Mixtral Boundary Summary

| mem | retained | Jaccard | adaptive ratio | adaptive k | profile tput |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 3.986 | 0.315 | 0.963 | 78 | 7.7 |
| 0.07 | 0.962 | 0.355 | 0.518 | 59 | 10.3 |
| 0.10 | 0.974 | 0.531 | 0.534 | 87 | 15.8 |

Tiny packed-runtime probe:
- `A`: gen tok/s=7.4902, resident_count=0
- `C`: gen tok/s=8.5867, resident_count=1
- tiny probe gain: +14.6%

## Interpretation

- `Qwen` remains the clearest compact-backbone positive case.
- `OLMoE` is also strongly positive, but mainly because resident pinning dominates under the current small-expert regime.
- `DeepSeek-V2-Lite` is not a negative case: `C > A` on real hardware, but the gains stay modest, so it should be framed as a weak positive / boundary case.
- `Mixtral` has non-trivial backbone structure in simulation (`retained≈0.96-0.97` at `mem=0.07/0.10`) and a positive tiny packed-runtime probe, but without a full-model runtime asset it should still stay as an applicability / boundary model rather than a formal positive runtime case.

