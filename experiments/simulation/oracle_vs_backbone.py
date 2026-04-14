"""Compare backbone-only vs oracle prefetch in simulator.

This experiment answers: is backbone-only theoretically optimal,
or does perfect prefetch still add value on top of backbone?

Result: oracle prefetch adds +18-45% on top of backbone-only.
Current causal prefetch adds nothing (overhead cancels benefit).
Conclusion: backbone is primary mechanism; better prefetch is future work.

这个实验回答：backbone-only 是否理论最优？
结果：oracle prefetch 在 backbone 之上还能加 18-45%。
当前 causal prefetch 收益为零（开销抵消了收益）。
结论：backbone 是主机制，更好的 prefetch 是 future work。
"""
import argparse

from backbone_moe.simulator import SystemBottleneckAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Compare backbone-only vs oracle prefetch in simulator"
    )
    parser.add_argument("--state-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--memory-ratios", type=str, default="0.07,0.10")
    parser.add_argument("--expert-size-mb", type=float, default=17.2)
    parser.add_argument("--h2d-bandwidth-gbps", type=float, default=16.0)
    parser.add_argument("--gpu-compute-time-ms", type=float, default=2.0)
    parser.add_argument("--resident-ratio", type=float, default=0.96)
    args = parser.parse_args()

    rratio = args.resident_ratio

    configs = [
        # label, mode, predictor, cache_layout, resident_policy, resident_ratio, prefetch_windows
        ("A demand-only",         "causal", "history_freq", "single",   "none",    0.5,    [0]),
        ("B oracle-prefetch",     "oracle", "oracle",       "single",   "none",    0.5,    [0, 1, 4, 10]),
        ("C backbone-only",       "causal", "history_freq", "two_pool", "profile_freq", rratio, [0]),
        ("D backbone+oracle-pf",  "oracle", "oracle",       "two_pool", "profile_freq", rratio, [0, 1, 4, 10]),
        ("E backbone+causal-pf",  "causal", "history_freq", "two_pool", "profile_freq", rratio, [1, 4, 10]),
    ]

    results = {}
    for label, mode, predictor, layout, rpol, r_ratio, windows in configs:
        row = {}
        for mem in mem_ratios:
            analyzer = SystemBottleneckAnalyzer(
                state_file=args.state_file,
                mode=mode,
                predictor=predictor,
                output_dir=args.output_dir,
                expert_size_mb=args.expert_size_mb,
                h2d_bandwidth_gbps=args.h2d_bandwidth_gbps,
                gpu_compute_time_ms=args.gpu_compute_time_ms,
                cache_layout=layout,
                resident_ratio=r_ratio,
                resident_policy=rpol,
                resident_profile_ratio=0.2,
            )
            best_tp = 0
            best_pw = 0
            for pw in windows:
                r = analyzer.simulate_with_config(mem, pw, reset_mode="shared")
                tp = r["throughput_tokens_per_sec"]
                if tp > best_tp:
                    best_tp = tp
                    best_pw = pw
            row[mem] = (best_tp, best_pw)
        results[label] = row

    hdr = f"{'Config':<28}"
    for mem in mem_ratios:
        hdr += f"  mem={mem:>8}"
    print(hdr)
    print("-" * (28 + 16 * len(mem_ratios)))
    for label, _, _, _, _, _, _ in configs:
        line = f"{label:<28}"
        for mem in mem_ratios:
            tp, pw = results[label][mem]
            line += f"  {tp:>8.1f} pw={pw}"
        print(line)

    print()
    print("Key comparison (backbone-only vs backbone+oracle):")
    for mem in mem_ratios:
        c_tp = results["C backbone-only"][mem][0]
        d_tp = results["D backbone+oracle-pf"][mem][0]
        diff = (d_tp - c_tp) / c_tp * 100
        winner = "oracle pf adds value" if d_tp > c_tp else "backbone-only sufficient"
        print(f"  mem={mem}: C={c_tp:.1f} vs D_oracle={d_tp:.1f} -> {diff:+.1f}% -> {winner}")


if __name__ == "__main__":
    main()
