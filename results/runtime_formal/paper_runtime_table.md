# Paper Runtime Table

This table uses the paper-facing comparison view:
- `Qwen`: fixed-mem runtime points
- `DeepSeek-V2-Lite`: fixed-mem runtime points
- `OLMoE`: coverage-matched fair runtime points

| Model | Comparison View | mem | A gen tok/s | C gen tok/s | gain | C resident |
|---|---|---:|---:|---:|---:|---:|
| Qwen1.5-MoE-A2.7B-Chat | fixed | 0.070 | 2.9266 | 3.2331 | +10.5% | 275 |
| Qwen1.5-MoE-A2.7B-Chat | fixed | 0.100 | 3.0946 | 3.6162 | +16.9% | 464 |
| OLMoE-1B-7B-0924 | fair | 0.012 | 0.3332 | 0.4988 | +49.7% | 316 |
| OLMoE-1B-7B-0924 | fair | 0.014 | 0.3336 | 0.5469 | +63.9% | 388 |
| OLMoE-1B-7B-0924 | fair | 0.016 | 0.3339 | 0.6053 | +81.3% | 462 |
| DeepSeek-V2-Lite | fixed | 0.070 | 0.1597 | 0.1782 | +11.6% | 186 |
| DeepSeek-V2-Lite | fixed | 0.100 | 0.1583 | 0.1776 | +12.2% | 174 |

## Notes

- OLMoE-1B-7B-0924: use coverage-matched fair runtime points for cross-model comparison because fixed mem-ratio is near-full-fit.
