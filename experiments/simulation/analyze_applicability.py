import argparse
import json
import tempfile
from pathlib import Path

from backbone_moe.evaluation import (
    cache_capacity_for_mem_ratio,
    parse_float_list,
    rank_resident_candidates,
    summarize_resident_applicability,
)
from backbone_moe.workload import load_state_dict, save_subset_state

from backbone_moe.simulator import SystemBottleneckAnalyzer


def build_profile_subset(state_file, profile_fraction, temp_dir):
    state_dict = load_state_dict(state_file)
    seq_keys = list(state_dict.keys())
    keep = max(1, int(round(len(seq_keys) * profile_fraction)))
    keep_keys = seq_keys[:keep]
    subset_path = Path(temp_dir) / f"profile_prefix_{keep}.pkl"
    save_subset_state(subset_path, state_dict, keep_keys)
    return subset_path


def build_analyzer(state_file, args):
    return SystemBottleneckAnalyzer(
        state_file=str(state_file),
        mode="oracle",
        predictor="history_freq",
        output_dir=str(args.output_dir),
        expert_size_mb=args.expert_size_mb,
        h2d_bandwidth_gbps=args.h2d_bandwidth_gbps,
        gpu_compute_time_ms=args.gpu_compute_time_ms,
        cache_layout="two_pool",
        resident_ratio=0.5,
        resident_policy=args.resident_policy,
        resident_profile_ratio=args.resident_profile_ratio,
        resident_depth_power=args.resident_depth_power,
    )


def entries_to_dict(entries):
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


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose when backbone-only runtime is likely to help: compact resident core + sufficient tail slack."
    )
    parser.add_argument("--state-file", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=str, default="applicability")
    parser.add_argument("--profile-fraction", type=float, default=0.2)
    parser.add_argument("--memory-ratios", type=parse_float_list, default=[0.07, 0.10])
    parser.add_argument("--probe-fractions", type=parse_float_list, default=[0.1, 0.2, 0.3])
    parser.add_argument("--resident-policy", type=str, default="profile_freq")
    parser.add_argument("--resident-profile-ratio", type=float, default=0.2)
    parser.add_argument("--resident-depth-power", type=float, default=1.0)
    parser.add_argument("--frontier-percentile", type=float, default=1.0)
    parser.add_argument("--frontier-horizon", type=int, default=0)
    parser.add_argument("--reset-mode", type=str, default="shared", choices=["shared", "per_sequence"])
    parser.add_argument("--expert-size-mb", type=float, default=17.2)
    parser.add_argument("--h2d-bandwidth-gbps", type=float, default=16.0)
    parser.add_argument("--gpu-compute-time-ms", type=float, default=2.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir) as temp_dir:
        profile_state_file = build_profile_subset(args.state_file, args.profile_fraction, temp_dir)
        analyzer = build_analyzer(profile_state_file, args)

        output = {
            "analysis": "backbone_applicability",
            "selection_method": "frontier_prefix",
            "state_file": str(args.state_file),
            "profile_state_file": str(profile_state_file),
            "profile_fraction": float(args.profile_fraction),
            "resident_policy": args.resident_policy,
            "resident_profile_ratio": float(args.resident_profile_ratio),
            "frontier_percentile": float(args.frontier_percentile),
            "frontier_horizon": int(args.frontier_horizon),
            "probe_fractions": [float(x) for x in args.probe_fractions],
            "memory_ratios": [],
        }

        for mem_ratio in args.memory_ratios:
            cache_capacity = cache_capacity_for_mem_ratio(mem_ratio, args.expert_size_mb)
            ranked = rank_resident_candidates(
                analyzer=analyzer,
                cache_capacity=cache_capacity,
                resident_policy=args.resident_policy,
                resident_profile_ratio=args.resident_profile_ratio,
                resident_depth_power=args.resident_depth_power,
                reset_mode=args.reset_mode,
            )
            runtime_row = analyzer.simulate_with_config(mem_ratio, prefetch_window=0, reset_mode=args.reset_mode)
            stall_curve = cumulative_curve(entries_to_dict(runtime_row["per_expert_critical_stall_ms"]))
            access_curve = cumulative_curve(entries_to_dict(runtime_row["per_expert_accesses"]))
            applicability = summarize_resident_applicability(
                ranked=ranked,
                access_sequence=analyzer.access_sequence,
                cache_capacity=cache_capacity,
                frontier_percentile=args.frontier_percentile,
                frontier_horizon=args.frontier_horizon,
            )
            output["memory_ratios"].append(
                {
                    "device_memory_ratio": float(mem_ratio),
                    "cache_capacity": int(cache_capacity),
                    "stall_top_coverage": coverage_at_fractions(stall_curve, args.probe_fractions),
                    "access_top_coverage": coverage_at_fractions(access_curve, args.probe_fractions),
                    "top20_stall_coverage": float(coverage_at_fractions(stall_curve, [0.2])["0.2"]),
                    "top20_access_coverage": float(coverage_at_fractions(access_curve, [0.2])["0.2"]),
                    "applicability": applicability,
                }
            )

    out_path = args.output_dir / f"{args.output_prefix}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved applicability summary to {out_path}")


if __name__ == "__main__":
    main()
