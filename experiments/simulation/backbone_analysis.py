import argparse
import json
from pathlib import Path

import numpy as np

from backbone_moe.simulator import SystemBottleneckAnalyzer


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_str_list(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def entries_to_dict(entries):
    return {
        (int(item["layer"]), int(item["expert"])): float(item["value"])
        for item in entries
    }


def resident_entries_to_set(entries):
    return {
        (int(item["layer"]), int(item["expert"]))
        for item in entries
    }


def cumulative_curve(metric_dict):
    values = sorted((float(v) for v in metric_dict.values()), reverse=True)
    total = float(sum(values))
    if total <= 0.0 or not values:
        return {
            "fractions": [0.0, 1.0],
            "coverage": [0.0, 1.0 if values else 0.0],
        }
    cumulative = 0.0
    fractions = []
    coverage = []
    n = len(values)
    for idx, value in enumerate(values, start=1):
        cumulative += value
        fractions.append(idx / n)
        coverage.append(cumulative / total)
    return {
        "fractions": fractions,
        "coverage": coverage,
    }


def coverage_at_fractions(curve, probe_fractions):
    fractions = curve["fractions"]
    coverage = curve["coverage"]
    result = {}
    for probe in probe_fractions:
        index = next((i for i, frac in enumerate(fractions) if frac >= probe), len(fractions) - 1)
        result[str(probe)] = coverage[index] if fractions else 0.0
    return result


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def build_analyzer(args, mode="oracle", predictor="history_freq", cache_layout="single",
                   resident_ratio=0.5, resident_policy="none", resident_profile_ratio=0.1):
    return SystemBottleneckAnalyzer(
        state_file=args.state_file,
        mode=mode,
        predictor=predictor,
        output_dir=args.output_dir,
        expert_size_mb=args.expert_size_mb,
        h2d_bandwidth_gbps=args.h2d_bandwidth_gbps,
        gpu_compute_time_ms=args.gpu_compute_time_ms,
        cache_layout=cache_layout,
        resident_ratio=resident_ratio,
        resident_policy=resident_policy,
        resident_profile_ratio=resident_profile_ratio,
    )


def run_concentration(args):
    analyzer = build_analyzer(args, mode="oracle", cache_layout="single")
    output = {
        "analysis": "backbone_concentration",
        "memory_ratios": args.memory_ratios,
        "probe_fractions": args.probe_fractions,
        "results": [],
    }
    for mem in args.memory_ratios:
        result = analyzer.simulate_with_config(mem, prefetch_window=0, reset_mode=args.reset_mode)
        stall_dict = entries_to_dict(result["per_expert_critical_stall_ms"])
        access_dict = entries_to_dict(result["per_expert_accesses"])
        stall_curve = cumulative_curve(stall_dict)
        access_curve = cumulative_curve(access_dict)
        output["results"].append({
            "device_memory_ratio": mem,
            "total_residual_stall_ms": result["total_residual_stall_ms"],
            "stall_top_coverage": coverage_at_fractions(stall_curve, args.probe_fractions),
            "access_top_coverage": coverage_at_fractions(access_curve, args.probe_fractions),
            "stall_curve": stall_curve,
            "access_curve": access_curve,
            "top_stall_experts": result["per_expert_critical_stall_ms"][:20],
        })
    path = Path(args.output_dir) / "backbone_concentration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    print(f"Saved concentration summary to {path}")


def run_stability(args):
    output = {
        "analysis": "backbone_stability",
        "resident_ratio": args.resident_ratio,
        "prefix_ratios": args.prefix_ratios,
        "resident_policies": args.resident_policies,
        "memory_ratios": args.memory_ratios,
        "results": [],
    }
    for policy in args.resident_policies:
        analyzer = build_analyzer(
            args,
            mode="oracle",
            cache_layout="two_pool",
            resident_ratio=args.resident_ratio,
            resident_policy=policy,
        )
        for mem in args.memory_ratios:
            sets = {}
            for prefix in args.prefix_ratios:
                analyzer.resident_profile_ratio = prefix
                info = analyzer.get_resident_set(mem, reset_mode=args.reset_mode)
                sets[prefix] = resident_entries_to_set(info["resident_set"])
            matrix = []
            for left in args.prefix_ratios:
                row = []
                for right in args.prefix_ratios:
                    row.append(jaccard(sets[left], sets[right]))
                matrix.append(row)
            output["results"].append({
                "resident_policy": policy,
                "device_memory_ratio": mem,
                "resident_set_size": len(sets[args.prefix_ratios[0]]),
                "jaccard_matrix": matrix,
                "prefix_labels": args.prefix_ratios,
                "resident_sets": {
                    str(prefix): analyzer._serialize_expert_pairs(sets[prefix])
                    for prefix in args.prefix_ratios
                },
            })
    path = Path(args.output_dir) / "backbone_stability.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    print(f"Saved stability summary to {path}")


def best_by_throughput(rows):
    return max(rows, key=lambda row: row["throughput_tokens_per_sec"])


def run_tail(args):
    baseline = build_analyzer(
        args,
        mode="oracle",
        cache_layout="single",
    )
    backbone = build_analyzer(
        args,
        mode="oracle",
        cache_layout="two_pool",
        resident_ratio=args.resident_ratio,
        resident_policy=args.backbone_policy,
        resident_profile_ratio=args.backbone_profile_ratio,
    )
    output = {
        "analysis": "tail_anti_predictability",
        "resident_ratio": args.resident_ratio,
        "backbone_policy": args.backbone_policy,
        "backbone_profile_ratio": args.backbone_profile_ratio,
        "memory_ratios": args.memory_ratios,
        "prefetch_windows": args.prefetch_windows,
        "results": [],
    }
    for mem in args.memory_ratios:
        baseline_rows = [
            baseline.simulate_with_config(mem, prefetch_window=window, reset_mode=args.reset_mode)
            for window in args.prefetch_windows
        ]
        backbone_rows = [
            backbone.simulate_with_config(mem, prefetch_window=window, reset_mode=args.reset_mode)
            for window in args.prefetch_windows
        ]
        best_baseline = best_by_throughput(baseline_rows)
        best_backbone = best_by_throughput(backbone_rows)
        output["results"].append({
            "device_memory_ratio": mem,
            "baseline_best": {
                "prefetch_window": best_baseline["prefetch_window"],
                "throughput_tokens_per_sec": best_baseline["throughput_tokens_per_sec"],
                "avg_residual_stall_ms": best_baseline["avg_residual_stall_ms"],
                "funnel_timely_rate": best_baseline["funnel_timely_rate"],
                "funnel_useful_rate": best_baseline["funnel_useful_rate"],
                "tail_funnel_timely_rate": best_baseline["tail_funnel_timely_rate"],
                "tail_funnel_useful_rate": best_baseline["tail_funnel_useful_rate"],
            },
            "backbone_best": {
                "prefetch_window": best_backbone["prefetch_window"],
                "throughput_tokens_per_sec": best_backbone["throughput_tokens_per_sec"],
                "avg_residual_stall_ms": best_backbone["avg_residual_stall_ms"],
                "funnel_timely_rate": best_backbone["funnel_timely_rate"],
                "funnel_useful_rate": best_backbone["funnel_useful_rate"],
                "tail_funnel_timely_rate": best_backbone["tail_funnel_timely_rate"],
                "tail_funnel_useful_rate": best_backbone["tail_funnel_useful_rate"],
                "resident_set_size": len(best_backbone["resident_set"]),
            },
            "stall_removed_ratio": 1.0 - (
                best_backbone["total_residual_stall_ms"] / max(1e-9, best_baseline["total_residual_stall_ms"])
            ),
            "window_table": [
                {
                    "prefetch_window": row["prefetch_window"],
                    "baseline_tp": base_row["throughput_tokens_per_sec"],
                    "baseline_tail_timely_rate": base_row["tail_funnel_timely_rate"],
                    "baseline_tail_useful_rate": base_row["tail_funnel_useful_rate"],
                    "backbone_tp": row["throughput_tokens_per_sec"],
                    "backbone_tail_timely_rate": row["tail_funnel_timely_rate"],
                    "backbone_tail_useful_rate": row["tail_funnel_useful_rate"],
                }
                for row, base_row in zip(backbone_rows, baseline_rows)
            ],
        })
    path = Path(args.output_dir) / "tail_anti_predictability.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    print(f"Saved tail summary to {path}")


def main():
    parser = argparse.ArgumentParser(description="Backbone-oriented analysis for MoE offloading")
    parser.add_argument("--analysis", type=str, default="all",
                        choices=["concentration", "stability", "tail", "all"])
    parser.add_argument("--state-file", type=str,
                        default="../states/Qwen1.5-MoE-A2.7B-Chat~lmsys-chat-1m~8.pkl")
    parser.add_argument("--output-dir", type=str, default="./experiments/results")
    parser.add_argument("--reset-mode", type=str, default="shared", choices=["shared", "per_sequence"])
    parser.add_argument("--expert-size-mb", type=float, default=17.2)
    parser.add_argument("--h2d-bandwidth-gbps", type=float, default=16.0)
    parser.add_argument("--gpu-compute-time-ms", type=float, default=2.0)
    parser.add_argument("--memory-ratios", type=parse_float_list, default=[0.05, 0.07, 0.10])
    parser.add_argument("--prefetch-windows", type=parse_int_list, default=[0, 1, 4, 10])
    parser.add_argument("--probe-fractions", type=parse_float_list, default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--prefix-ratios", type=parse_float_list, default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--resident-ratio", type=float, default=0.9)
    parser.add_argument("--resident-policies", type=parse_str_list,
                        default=["profile_freq", "profile_miss_stall"])
    parser.add_argument("--backbone-policy", type=str, default="profile_freq")
    parser.add_argument("--backbone-profile-ratio", type=float, default=0.2)
    args = parser.parse_args()

    if args.analysis in {"concentration", "all"}:
        run_concentration(args)
    if args.analysis in {"stability", "all"}:
        run_stability(args)
    if args.analysis in {"tail", "all"}:
        run_tail(args)


if __name__ == "__main__":
    main()
