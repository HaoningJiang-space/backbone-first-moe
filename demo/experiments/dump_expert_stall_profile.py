import argparse
import json
from pathlib import Path

from insight_4_system_bottleneck import SystemBottleneckAnalyzer


def parse_entries(entries):
    return {
        (int(item["layer"]), int(item["expert"])): float(item["value"])
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


def top_entries(metric_dict, limit):
    items = sorted(metric_dict.items(), key=lambda item: item[1], reverse=True)
    return [
        {"layer": int(layer), "expert": int(expert), "value": float(value)}
        for (layer, expert), value in items[:limit]
    ]


def build_output_name(args):
    predictor = "oracle" if args.mode == "oracle" else args.predictor
    mem_tag = str(args.memory_ratio).replace(".", "p")
    expert_tag = str(args.expert_size_mb).replace(".", "p")
    parts = [
        f"stall_profile_mode{args.mode}_predictor{predictor}_"
        f"adm{args.prefetch_admission}_mem{mem_tag}_pw{args.prefetch_window}_"
        f"e{expert_tag}_{args.reset_mode}"
    ]
    if args.cache_layout != "single":
        parts.append(f"layout{args.cache_layout}")
        parts.append(f"res{str(args.resident_ratio).replace('.', 'p')}")
        if args.resident_policy != "none":
            parts.append(f"rpol{args.resident_policy}")
        if args.resident_policy in {"profile_freq", "profile_depth_freq", "profile_miss_stall"}:
            parts.append(f"rprof{str(args.resident_profile_ratio).replace('.', 'p')}")
        if args.resident_policy == "profile_depth_freq":
            parts.append(f"rdpow{str(args.resident_depth_power).replace('.', 'p')}")
    return "_".join(parts) + ".json"


def main():
    parser = argparse.ArgumentParser(description="Dump per-expert critical stall profile for one simulator point.")
    parser.add_argument("--state-file", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", type=str, default="oracle", choices=["oracle", "causal"])
    parser.add_argument("--predictor", type=str, default="history_freq",
                        choices=["history_freq", "pl_ctr", "utility_freq"])
    parser.add_argument("--reset-mode", type=str, default="shared", choices=["shared", "per_sequence"])
    parser.add_argument("--prefetch-admission", type=str, default="deadline",
                        choices=["none", "deadline", "value"])
    parser.add_argument("--cache-layout", type=str, default="single", choices=["single", "two_pool"])
    parser.add_argument("--resident-ratio", type=float, default=0.5)
    parser.add_argument("--resident-policy", type=str, default="none",
                        choices=["none", "oracle_freq", "profile_freq", "profile_depth_freq", "profile_miss_stall"])
    parser.add_argument("--resident-profile-ratio", type=float, default=0.1)
    parser.add_argument("--resident-depth-power", type=float, default=1.0)
    parser.add_argument("--memory-ratio", type=float, default=0.10)
    parser.add_argument("--prefetch-window", type=int, default=1)
    parser.add_argument("--expert-size-mb", type=float, default=17.2)
    parser.add_argument("--h2d-bandwidth-gbps", type=float, default=16.0)
    parser.add_argument("--gpu-compute-time-ms", type=float, default=2.0)
    parser.add_argument("--deadline-margin-ms", type=float, default=0.0)
    parser.add_argument("--value-cost-scale", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--probe-fractions", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = SystemBottleneckAnalyzer(
        state_file=args.state_file,
        mode=args.mode,
        predictor=args.predictor,
        output_dir=str(args.output_dir),
        expert_size_mb=args.expert_size_mb,
        h2d_bandwidth_gbps=args.h2d_bandwidth_gbps,
        gpu_compute_time_ms=args.gpu_compute_time_ms,
        prefetch_admission=args.prefetch_admission,
        deadline_margin_ms=args.deadline_margin_ms,
        value_cost_scale=args.value_cost_scale,
        cache_layout=args.cache_layout,
        resident_ratio=args.resident_ratio,
        resident_policy=args.resident_policy,
        resident_profile_ratio=args.resident_profile_ratio,
        resident_depth_power=args.resident_depth_power,
    )
    result = analyzer.simulate_with_config(
        device_memory_ratio=args.memory_ratio,
        prefetch_window=args.prefetch_window,
        reset_mode=args.reset_mode,
    )

    stall_dict = parse_entries(result["per_expert_critical_stall_ms"])
    access_dict = parse_entries(result["per_expert_accesses"])
    miss_dict = parse_entries(result["per_expert_demand_misses"])
    late_dict = parse_entries(result["per_expert_late_prefetches"])
    lorenz = cumulative_curve(stall_dict)
    access_curve = cumulative_curve(access_dict)

    payload = {
        "analysis": "expert_stall_profile",
        "mode": args.mode,
        "predictor": "oracle" if args.mode == "oracle" else args.predictor,
        "memory_ratio": float(args.memory_ratio),
        "prefetch_window": int(args.prefetch_window),
        "reset_mode": args.reset_mode,
        "prefetch_admission": args.prefetch_admission,
        "cache_layout": args.cache_layout,
        "resident_ratio": float(args.resident_ratio),
        "resident_policy": args.resident_policy,
        "resident_profile_ratio": float(args.resident_profile_ratio),
        "resident_depth_power": float(args.resident_depth_power),
        "config": result["config"],
        "summary": {
            "throughput_tokens_per_sec": float(result["throughput_tokens_per_sec"]),
            "avg_residual_stall_ms": float(result["avg_residual_stall_ms"]),
            "total_residual_stall_ms": float(result["total_residual_stall_ms"]),
            "hit_rate": float(result["hit_rate"]),
            "prefetch_hit_rate": float(result["prefetch_hit_rate"]),
            "late_prefetch_rate": float(result["late_prefetch_rate"]),
            "funnel_novel_rate": float(result["funnel_novel_rate"]),
            "funnel_useful_rate": float(result["funnel_useful_rate"]),
        },
        "stall_lorenz": lorenz,
        "access_curve": access_curve,
        "stall_top_coverage": coverage_at_fractions(lorenz, args.probe_fractions),
        "access_top_coverage": coverage_at_fractions(access_curve, args.probe_fractions),
        "top_stall_experts": top_entries(stall_dict, args.top_k),
        "top_access_experts": top_entries(access_dict, args.top_k),
        "per_layer_residual_stall_ms": result["per_layer_residual_stall_ms"],
        "per_expert_critical_stall_ms": result["per_expert_critical_stall_ms"],
        "per_expert_accesses": result["per_expert_accesses"],
        "per_expert_demand_misses": result["per_expert_demand_misses"],
        "per_expert_late_prefetches": result["per_expert_late_prefetches"],
        "resident_set": result["resident_set"],
        "derived_counts": {
            "num_layers": int(analyzer.num_layers),
            "num_experts_with_stall": int(sum(1 for value in stall_dict.values() if value > 0.0)),
            "num_experts_with_access": int(sum(1 for value in access_dict.values() if value > 0.0)),
            "total_demand_misses": int(sum(miss_dict.values())),
            "total_late_prefetches": int(sum(late_dict.values())),
        },
    }

    output_path = args.output_dir / build_output_name(args)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved expert stall profile to {output_path}")


if __name__ == "__main__":
    main()
