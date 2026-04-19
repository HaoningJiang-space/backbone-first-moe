import argparse
import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import AutoConfig

from backbone_moe.evaluation import (
    build_batch_union_demand_steps,
    parse_float_list,
    parse_int_list,
)
from backbone_moe.simulator import SystemBottleneckAnalyzer
from backbone_moe.workload import load_state_dict, save_subset_state
from finemoe.utils import infer_routed_expert_size_mb, normalize_runtime_config


def summarize_distribution(values):
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    ordered = sorted(float(v) for v in values)

    def percentile(p):
        idx = int(round((len(ordered) - 1) * p))
        idx = max(0, min(len(ordered) - 1, idx))
        return float(ordered[idx])

    return {
        "count": int(len(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def compute_step_reuse_distances(demand_steps):
    last_seen = {}
    distances = []
    for step_idx, demand in enumerate(demand_steps):
        for expert_key in demand:
            if expert_key in last_seen:
                distances.append(step_idx - last_seen[expert_key])
            last_seen[expert_key] = step_idx
    return distances


def compute_working_set_stats(demand_steps, windows):
    results = {}
    for window in windows:
        window = int(window)
        if window <= 0:
            continue
        sizes = []
        if len(demand_steps) < window:
            sizes.append(len(set().union(*demand_steps)) if demand_steps else 0)
        else:
            for start in range(0, len(demand_steps) - window + 1):
                active = set()
                for step in demand_steps[start : start + window]:
                    active.update(step)
                sizes.append(len(active))
        results[str(window)] = summarize_distribution(sizes)
    return results


def build_profile_subset(state_file, profile_fraction, temp_dir):
    state_dict = load_state_dict(state_file)
    seq_keys = list(state_dict.keys())
    keep = max(1, int(round(len(seq_keys) * profile_fraction)))
    subset_path = Path(temp_dir) / f"profile_prefix_{keep}.pkl"
    save_subset_state(subset_path, state_dict, seq_keys[:keep])
    return subset_path


def resolve_expert_size_mb(args):
    if args.expert_size_mb is not None:
        return float(args.expert_size_mb), "cli"
    if args.model_path:
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        config = normalize_runtime_config(config)
        return float(infer_routed_expert_size_mb(config)), f"model:{args.model_path}"
    return 17.2, "default"


def build_analyzer(
    state_file,
    output_dir,
    expert_size_mb,
    h2d_bandwidth_gbps,
    gpu_compute_time_ms,
    mode="oracle",
    predictor="history_freq",
    cache_layout="single",
    resident_ratio=0.5,
    resident_policy="none",
    resident_profile_ratio=0.2,
):
    return SystemBottleneckAnalyzer(
        state_file=str(state_file),
        mode=mode,
        predictor=predictor,
        output_dir=str(output_dir),
        expert_size_mb=expert_size_mb,
        h2d_bandwidth_gbps=h2d_bandwidth_gbps,
        gpu_compute_time_ms=gpu_compute_time_ms,
        cache_layout=cache_layout,
        resident_ratio=resident_ratio,
        resident_policy=resident_policy,
        resident_profile_ratio=resident_profile_ratio,
    )


def slim_row(row):
    return {
        "device_memory_ratio": float(row["device_memory_ratio"]),
        "prefetch_window": int(row["prefetch_window"]),
        "cache_capacity": int(row["cache_capacity"]),
        "resident_capacity": int(row["resident_capacity"]),
        "speculative_capacity": int(row["speculative_capacity"]),
        "num_pinned_residents": int(row["num_pinned_residents"]),
        "throughput_tokens_per_sec": float(row["throughput_tokens_per_sec"]),
        "avg_residual_stall_ms": float(row["avg_residual_stall_ms"]),
        "total_residual_stall_ms": float(row["total_residual_stall_ms"]),
        "gpu_idle_ratio": float(row["gpu_idle_ratio"]),
        "hit_rate": float(row["hit_rate"]),
        "prefetch_hit_rate": float(row["prefetch_hit_rate"]),
        "late_prefetch_rate": float(row["late_prefetch_rate"]),
        "h2d_bandwidth_gbps": float(row["h2d_bandwidth_gbps"]),
        "tail_funnel_timely_rate": float(row["tail_funnel_timely_rate"]),
        "tail_funnel_useful_rate": float(row["tail_funnel_useful_rate"]),
    }


def best_by_throughput(rows):
    return max(rows, key=lambda row: row["throughput_tokens_per_sec"])


def compute_compute_only_upper_bound(tokens, gpu_compute_time_ms):
    total_compute_ms = float(tokens) * float(gpu_compute_time_ms)
    throughput = float(tokens) / (total_compute_ms / 1000.0) if total_compute_ms > 0 else 0.0
    return {
        "tokens": int(tokens),
        "total_compute_time_ms": float(total_compute_ms),
        "throughput_tokens_per_sec": float(throughput),
    }


def summarize_bottleneck(row, compute_only):
    total_compute_ms = float(compute_only["total_compute_time_ms"])
    stall_ms = float(row["total_residual_stall_ms"])
    denom = total_compute_ms + stall_ms
    return {
        "compute_time_ms": float(total_compute_ms),
        "residual_stall_ms": float(stall_ms),
        "loading_share": float(stall_ms / denom) if denom > 0 else 0.0,
        "compute_share": float(total_compute_ms / denom) if denom > 0 else 0.0,
        "zero_loading_speedup_upper_bound": (
            float(compute_only["throughput_tokens_per_sec"] / row["throughput_tokens_per_sec"])
            if row["throughput_tokens_per_sec"] > 0
            else 0.0
        ),
    }


def run_upper_bound_probe(args, state_file, output_dir):
    baseline = build_analyzer(
        state_file=state_file,
        output_dir=output_dir,
        expert_size_mb=args.expert_size_mb,
        h2d_bandwidth_gbps=args.h2d_bandwidth_gbps,
        gpu_compute_time_ms=args.gpu_compute_time_ms,
        mode="oracle",
        cache_layout="single",
    )
    num_tokens = len(baseline.access_sequence)
    demand_steps = build_batch_union_demand_steps(baseline.access_sequence)
    compute_only = compute_compute_only_upper_bound(num_tokens, args.gpu_compute_time_ms)

    results = {
        "tokens": int(num_tokens),
        "batch_steps": int(len(demand_steps)),
        "compute_only_upper_bound": compute_only,
        "memory_ratios": [],
    }

    for mem_ratio in args.memory_ratios:
        baseline_row = baseline.simulate_with_config(mem_ratio, 0, reset_mode=args.reset_mode)

        single_rows = []
        for window in args.oracle_windows:
            single_rows.append(
                build_analyzer(
                    state_file=state_file,
                    output_dir=output_dir,
                    expert_size_mb=args.expert_size_mb,
                    h2d_bandwidth_gbps=args.h2d_bandwidth_gbps,
                    gpu_compute_time_ms=args.gpu_compute_time_ms,
                    mode="oracle",
                    cache_layout="single",
                ).simulate_with_config(mem_ratio, window, reset_mode=args.reset_mode)
            )
        best_single = best_by_throughput(single_rows)

        two_pool_rows = []
        for resident_ratio in args.oracle_resident_ratios:
            analyzer = build_analyzer(
                state_file=state_file,
                output_dir=output_dir,
                expert_size_mb=args.expert_size_mb,
                h2d_bandwidth_gbps=args.h2d_bandwidth_gbps,
                gpu_compute_time_ms=args.gpu_compute_time_ms,
                mode="oracle",
                cache_layout="two_pool",
                resident_ratio=resident_ratio,
                resident_policy=args.oracle_resident_policy,
                resident_profile_ratio=args.oracle_profile_fraction,
            )
            for window in args.oracle_windows:
                row = analyzer.simulate_with_config(mem_ratio, window, reset_mode=args.reset_mode)
                row["resident_ratio"] = float(resident_ratio)
                two_pool_rows.append(row)
        best_two_pool = best_by_throughput(two_pool_rows)

        results["memory_ratios"].append(
            {
                "device_memory_ratio": float(mem_ratio),
                "baseline_demand_only": slim_row(baseline_row),
                "oracle_single_best": {
                    **slim_row(best_single),
                    "speedup_over_baseline": (
                        float(best_single["throughput_tokens_per_sec"] / baseline_row["throughput_tokens_per_sec"])
                        if baseline_row["throughput_tokens_per_sec"] > 0
                        else 0.0
                    ),
                },
                "oracle_two_pool_best": {
                    **slim_row(best_two_pool),
                    "resident_ratio": float(best_two_pool["resident_ratio"]),
                    "speedup_over_baseline": (
                        float(best_two_pool["throughput_tokens_per_sec"] / baseline_row["throughput_tokens_per_sec"])
                        if baseline_row["throughput_tokens_per_sec"] > 0
                        else 0.0
                    ),
                },
                "bottleneck_breakdown": summarize_bottleneck(baseline_row, compute_only),
            }
        )
    return results, demand_steps


def build_markdown(summary):
    lines = []
    lines.append("# Serving Bottleneck Observation")
    lines.append("")
    lines.append(f"- state_file: `{summary['state_file']}`")
    lines.append(f"- expert_size_mb: `{summary['expert_size_mb']}`")
    lines.append(f"- expert_size_source: `{summary['expert_size_source']}`")
    lines.append(f"- gpu_compute_time_ms: `{summary['gpu_compute_time_ms']}`")
    lines.append(f"- h2d_bandwidth_gbps: `{summary['h2d_bandwidth_gbps']}`")
    lines.append("")
    lines.append("## Upper Bounds")
    lines.append("")
    lines.append("| mem | baseline_tp | oracle_single_tp | oracle_single_speedup | oracle_two_pool_tp | oracle_two_pool_speedup | zero_loading_speedup | loading_share |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["upper_bounds"]["memory_ratios"]:
        baseline = row["baseline_demand_only"]
        oracle_single = row["oracle_single_best"]
        oracle_two_pool = row["oracle_two_pool_best"]
        breakdown = row["bottleneck_breakdown"]
        lines.append(
            f"| {row['device_memory_ratio']:.3f} | "
            f"{baseline['throughput_tokens_per_sec']:.2f} | "
            f"{oracle_single['throughput_tokens_per_sec']:.2f} | "
            f"{oracle_single['speedup_over_baseline']:.2f} | "
            f"{oracle_two_pool['throughput_tokens_per_sec']:.2f} | "
            f"{oracle_two_pool['speedup_over_baseline']:.2f} | "
            f"{breakdown['zero_loading_speedup_upper_bound']:.2f} | "
            f"{breakdown['loading_share']:.3f} |"
        )
    lines.append("")
    lines.append("## Reuse")
    lines.append("")
    reuse = summary["reuse"]
    lines.append(f"- demand_steps: `{reuse['num_demand_steps']}`")
    lines.append(f"- unique_experts: `{reuse['num_unique_experts']}`")
    lines.append(f"- reuse_distance_p50: `{reuse['reuse_distance']['p50']}`")
    lines.append(f"- reuse_distance_p95: `{reuse['reuse_distance']['p95']}`")
    lines.append("")
    lines.append("## Working Set")
    lines.append("")
    lines.append("| window | mean | p50 | p95 | max |")
    lines.append("|---|---:|---:|---:|---:|")
    for window, stats in summary["working_set"].items():
        lines.append(
            f"| {window} | {stats['mean']:.2f} | {stats['p50']:.2f} | {stats['p95']:.2f} | {stats['max']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Observation experiments for MoE serving bottlenecks.")
    parser.add_argument("--state-file", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=str, default="serving_bottleneck")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--expert-size-mb", type=float, default=None)
    parser.add_argument("--profile-fraction", type=float, default=1.0)
    parser.add_argument("--memory-ratios", type=parse_float_list, default=[0.07, 0.10])
    parser.add_argument("--oracle-windows", type=parse_int_list, default=[0, 1, 4, 10])
    parser.add_argument("--oracle-resident-ratios", type=parse_float_list, default=[0.1, 0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--oracle-resident-policy", type=str, default="oracle_freq")
    parser.add_argument("--oracle-profile-fraction", type=float, default=0.2)
    parser.add_argument("--working-set-windows", type=parse_int_list, default=[1, 4, 8, 16, 32])
    parser.add_argument("--h2d-bandwidth-gbps", type=float, default=16.0)
    parser.add_argument("--gpu-compute-time-ms", type=float, default=2.0)
    parser.add_argument("--reset-mode", type=str, default="shared", choices=["shared", "per_sequence"])
    args = parser.parse_args()

    args.expert_size_mb, expert_size_source = resolve_expert_size_mb(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=args.output_dir) as temp_dir:
        profile_state = (
            build_profile_subset(args.state_file, args.profile_fraction, temp_dir)
            if args.profile_fraction < 0.999
            else Path(args.state_file)
        )

        upper_bounds, demand_steps = run_upper_bound_probe(args, profile_state, args.output_dir)
        reuse_distances = compute_step_reuse_distances(demand_steps)
        unique_experts = set()
        for demand in demand_steps:
            unique_experts.update(demand)

        summary = {
            "analysis": "serving_bottleneck_observation",
            "state_file": str(args.state_file),
            "profile_state_file": str(profile_state),
            "profile_fraction": float(args.profile_fraction),
            "expert_size_mb": float(args.expert_size_mb),
            "expert_size_source": expert_size_source,
            "gpu_compute_time_ms": float(args.gpu_compute_time_ms),
            "h2d_bandwidth_gbps": float(args.h2d_bandwidth_gbps),
            "upper_bounds": upper_bounds,
            "reuse": {
                "num_demand_steps": int(len(demand_steps)),
                "num_unique_experts": int(len(unique_experts)),
                "reuse_distance": summarize_distribution(reuse_distances),
            },
            "working_set": compute_working_set_stats(demand_steps, args.working_set_windows),
        }

    json_path = args.output_dir / f"{args.output_prefix}.json"
    md_path = args.output_dir / f"{args.output_prefix}.md"
    json_path.write_text(json.dumps(summary, indent=2))
    md_path.write_text(build_markdown(summary))
    print(f"Saved observation JSON to {json_path}")
    print(f"Saved observation Markdown to {md_path}")


if __name__ == "__main__":
    main()
