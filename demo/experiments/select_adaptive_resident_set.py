import argparse
import json
import tempfile
from pathlib import Path

from finemoe.backbone.evaluation import (
    best_by_throughput,
    compute_capacity_knee,
    evaluate_with_fixed_resident_layout,
    parse_float_list,
    parse_int_list,
)
from finemoe.backbone.workload import load_state_dict, save_subset_state

from insight_4_system_bottleneck import SystemBottleneckAnalyzer


def build_profile_subset(state_file, profile_fraction, temp_dir):
    state_dict = load_state_dict(state_file)
    seq_keys = list(state_dict.keys())
    keep = max(1, int(round(len(seq_keys) * profile_fraction)))
    keep_keys = seq_keys[:keep]
    subset_path = Path(temp_dir) / f"profile_prefix_{keep}.pkl"
    save_subset_state(subset_path, state_dict, keep_keys)
    return subset_path, keep_keys


def build_analyzer(state_file, args, resident_ratio):
    return SystemBottleneckAnalyzer(
        state_file=str(state_file),
        mode=args.mode,
        predictor=args.predictor,
        output_dir=str(args.output_dir),
        expert_size_mb=args.expert_size_mb,
        h2d_bandwidth_gbps=args.h2d_bandwidth_gbps,
        gpu_compute_time_ms=args.gpu_compute_time_ms,
        cache_layout="two_pool",
        resident_ratio=resident_ratio,
        resident_policy=args.resident_policy,
        resident_profile_ratio=args.resident_profile_ratio,
        resident_depth_power=args.resident_depth_power,
    )


def cache_capacity_for_mem_ratio(mem_ratio, expert_size_mb):
    total_gpu_memory_mb = 80 * 1024
    available_memory_mb = total_gpu_memory_mb * mem_ratio
    return int(available_memory_mb / expert_size_mb)


def rank_resident_candidates(analyzer, cache_capacity, args):
    if args.resident_policy == "oracle_freq":
        scores = dict(analyzer.expert_access_count)
    elif args.resident_policy == "profile_freq":
        scores = analyzer._count_expert_accesses(args.resident_profile_ratio, score_mode="freq")
    elif args.resident_policy == "profile_depth_freq":
        scores = analyzer._count_expert_accesses(
            args.resident_profile_ratio,
            score_mode="depth_freq",
        )
    elif args.resident_policy == "profile_miss_stall":
        # Rank candidates by direct stall contribution with zero resident reservation.
        scores = analyzer._profile_miss_stall_scores(
            args.resident_profile_ratio,
            cache_capacity,
            0,
            args.reset_mode,
        )
    else:
        raise ValueError(f"Unsupported resident_policy: {args.resident_policy}")

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )
    return ranked


def search_best_capacity(profile_state_file, mem_ratio, args):
    """Find throughput-optimal resident capacity by sweeping from knee to cache limit.

    Two-stage approach:
      1. Compute utility-curve knee as the structural lower bound.
      2. Sweep ~20 candidate capacities from knee to cache_capacity,
         evaluate simulated throughput at each, and pick the best.

    This is fully data-driven with zero manual ratio tuning.

    两阶段方法：
      1. 从 utility 曲线计算 knee 作为结构下界。
      2. 从 knee 到 cache_capacity 搜索约 20 个候选容量，
         在每个点评估模拟吞吐量，选最佳。
    全数据驱动，无需手动调 ratio。
    """
    cache_capacity = cache_capacity_for_mem_ratio(mem_ratio, args.expert_size_mb)
    profile_analyzer = build_analyzer(profile_state_file, args, resident_ratio=0.5)
    ranked = rank_resident_candidates(profile_analyzer, cache_capacity, args)

    knee_capacity = compute_capacity_knee(ranked, cache_capacity)
    ranked_experts = [expert_key for expert_key, _ in ranked[:cache_capacity]]
    num_ranked = min(cache_capacity, len(ranked_experts))

    # Build candidate capacities: knee, then evenly spaced up to num_ranked
    # 构建候选容量：knee 起步，均匀分布到 num_ranked
    num_steps = min(20, num_ranked - knee_capacity + 1)
    if num_steps <= 1:
        candidate_capacities = [knee_capacity]
    else:
        step = max(1, (num_ranked - knee_capacity) // (num_steps - 1))
        candidate_capacities = list(range(knee_capacity, num_ranked + 1, step))
        if candidate_capacities[-1] != num_ranked:
            candidate_capacities.append(num_ranked)
    # Always include knee and full capacity
    # 始终包含 knee 和满容量
    candidate_capacities = sorted(set(candidate_capacities))

    candidates = []
    for k in candidate_capacities:
        resident_set = set(ranked_experts[:k])
        try:
            rows = evaluate_with_fixed_resident_layout(
                eval_analyzer=profile_analyzer,
                resident_set=resident_set,
                resident_capacity=k,
                cache_capacity=cache_capacity,
                mem_ratio=mem_ratio,
                windows=args.prefetch_windows,
                reset_mode=args.reset_mode,
            )
        except RuntimeError:
            # Demand cache too small at this capacity; skip
            # 该容量下 demand cache 过小，跳过
            continue
        best_row = best_by_throughput(rows)
        candidates.append({
            "resident_capacity": int(k),
            "resident_ratio": float(k / cache_capacity) if cache_capacity > 0 else 0.0,
            "best_prefetch_window": int(best_row["prefetch_window"]),
            "throughput_tokens_per_sec": float(best_row["throughput_tokens_per_sec"]),
            "avg_residual_stall_ms": float(best_row["avg_residual_stall_ms"]),
            "speculative_capacity": int(cache_capacity - k),
            "num_pinned_residents": int(best_row["num_pinned_residents"]),
            "selection_rule": "throughput_sweep",
        })

    candidates.sort(key=lambda c: (-c["throughput_tokens_per_sec"], c["resident_capacity"]))
    best = candidates[0]

    # Record knee for reference
    # 记录 knee 供参考
    best["knee_capacity"] = int(knee_capacity)
    best["knee_ratio"] = float(knee_capacity / cache_capacity) if cache_capacity > 0 else 0.0

    return best, candidates, cache_capacity


def search_best_ratio(profile_state_file, mem_ratio, args):
    candidates = []
    for resident_ratio in args.candidate_ratios:
        analyzer = build_analyzer(profile_state_file, args, resident_ratio)
        rows = [
            analyzer.simulate_with_config(mem_ratio, window, reset_mode=args.reset_mode)
            for window in args.prefetch_windows
        ]
        best_row = best_by_throughput(rows)
        candidates.append(
            {
                "resident_ratio": float(resident_ratio),
                "best_prefetch_window": int(best_row["prefetch_window"]),
                "throughput_tokens_per_sec": float(best_row["throughput_tokens_per_sec"]),
                "avg_residual_stall_ms": float(best_row["avg_residual_stall_ms"]),
                "resident_capacity": int(best_row["resident_capacity"]),
                "speculative_capacity": int(best_row["speculative_capacity"]),
                "num_pinned_residents": int(best_row["num_pinned_residents"]),
            }
        )

    candidates.sort(
        key=lambda row: (
            -row["throughput_tokens_per_sec"],
            row["resident_ratio"],
        )
    )
    return candidates[0], candidates


def export_selected_resident_set(full_state_file, mem_ratio, best_ratio, args, best_capacity=None):
    analyzer = build_analyzer(full_state_file, args, best_ratio)
    if best_capacity is None:
        return analyzer.get_resident_set(mem_ratio, reset_mode=args.reset_mode)

    cache_capacity = cache_capacity_for_mem_ratio(mem_ratio, args.expert_size_mb)
    ranked = rank_resident_candidates(analyzer, cache_capacity, args)
    resident_set = ranked[:best_capacity]
    return {
        "resident_set": [
            {"layer": int(expert_key[0]), "expert": int(expert_key[1])}
            for expert_key, _ in resident_set
        ],
        "resident_capacity": int(best_capacity),
        "speculative_capacity": int(max(0, cache_capacity - best_capacity)),
        "resident_policy": args.resident_policy,
        "resident_ratio": float(best_capacity / cache_capacity) if cache_capacity > 0 else 0.0,
        "cache_capacity": int(cache_capacity),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Automatically select a resident split on a profiling prefix and export the resulting resident set."
    )
    parser.add_argument("--state-file", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./experiments/results"))
    parser.add_argument("--output-prefix", type=str, default="adaptive_resident")
    parser.add_argument("--memory-ratios", type=parse_float_list, default=[0.05, 0.07, 0.10])
    parser.add_argument("--candidate-ratios", type=parse_float_list, default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument(
        "--selection-method",
        type=str,
        choices=["capacity_search", "ratio_grid"],
        default="capacity_search",
    )
    parser.add_argument("--prefetch-windows", type=parse_int_list, default=[0, 1, 4, 10])
    parser.add_argument("--profile-fraction", type=float, default=0.2)
    parser.add_argument("--mode", type=str, default="oracle")
    parser.add_argument("--predictor", type=str, default="history_freq")
    parser.add_argument("--resident-policy", type=str, default="profile_freq")
    parser.add_argument("--resident-profile-ratio", type=float, default=0.2)
    parser.add_argument("--resident-depth-power", type=float, default=1.0)
    parser.add_argument("--reset-mode", type=str, default="shared", choices=["shared", "per_sequence"])
    parser.add_argument("--expert-size-mb", type=float, default=17.2)
    parser.add_argument("--h2d-bandwidth-gbps", type=float, default=16.0)
    parser.add_argument("--gpu-compute-time-ms", type=float, default=2.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "state_file": args.state_file,
        "profile_fraction": float(args.profile_fraction),
        "resident_policy": args.resident_policy,
        "resident_profile_ratio": float(args.resident_profile_ratio),
        "selection_method": args.selection_method,
        "candidate_ratios": [float(x) for x in args.candidate_ratios],
        "prefetch_windows": [int(x) for x in args.prefetch_windows],
        "results": [],
    }

    with tempfile.TemporaryDirectory(prefix="adaptive_resident_") as temp_dir:
        profile_state_file, profile_keys = build_profile_subset(args.state_file, args.profile_fraction, temp_dir)
        summary["profile_sequences"] = len(profile_keys)

        for mem_ratio in args.memory_ratios:
            if args.selection_method == "capacity_search":
                best_row, candidates, cache_capacity = search_best_capacity(profile_state_file, mem_ratio, args)
                resident_info = export_selected_resident_set(
                    args.state_file,
                    mem_ratio,
                    best_row["resident_ratio"],
                    args,
                    best_capacity=best_row["resident_capacity"],
                )
            else:
                best_row, candidates = search_best_ratio(profile_state_file, mem_ratio, args)
                cache_capacity = None
                resident_info = export_selected_resident_set(
                    args.state_file,
                    mem_ratio,
                    best_row["resident_ratio"],
                    args,
                )
            mem_tag = f"{mem_ratio:.2f}".replace(".", "p")
            resident_path = args.output_dir / f"{args.output_prefix}_mem{mem_tag}.json"
            resident_payload = {
                **resident_info,
                "selection_method": args.selection_method,
                "selected_resident_ratio": float(best_row["resident_ratio"]),
                "selected_resident_capacity": int(best_row.get("resident_capacity", resident_info["resident_capacity"])),
                "selection_best_prefetch_window": int(best_row["best_prefetch_window"]),
                "selection_profile_fraction": float(args.profile_fraction),
                "selection_candidates": candidates,
            }
            resident_path.write_text(json.dumps(resident_payload, indent=2))
            summary["results"].append(
                {
                    "device_memory_ratio": float(mem_ratio),
                    "selected_resident_ratio": float(best_row["resident_ratio"]),
                    "selected_resident_capacity": int(best_row.get("resident_capacity", resident_info["resident_capacity"])),
                    "best_prefetch_window": int(best_row["best_prefetch_window"]),
                    "profile_throughput_tokens_per_sec": float(best_row["throughput_tokens_per_sec"]),
                    "cache_capacity": int(cache_capacity or resident_info["cache_capacity"]),
                    "resident_capacity": int(resident_info["resident_capacity"]),
                    "speculative_capacity": int(resident_info["speculative_capacity"]),
                    "resident_count": len(resident_info["resident_set"]),
                    "resident_file": str(resident_path),
                    "candidates": candidates,
                }
            )
            knee_k = best_row.get("knee_capacity", best_row["resident_capacity"])
            knee_r = best_row.get("knee_ratio", best_row["resident_ratio"])
            print(
                f"mem={mem_ratio:.2f}: "
                f"optimal_k={best_row['resident_capacity']} (ratio={best_row['resident_ratio']:.2f}), "
                f"knee_k={knee_k} (ratio={knee_r:.2f}), "
                f"tp={best_row['throughput_tokens_per_sec']:.1f} tok/s, "
                f"searched {len(candidates)} points"
            )

    summary_path = args.output_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved adaptive split summary to {summary_path}")


if __name__ == "__main__":
    main()
