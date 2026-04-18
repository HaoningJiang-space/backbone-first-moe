import argparse
import json
import tempfile
from pathlib import Path

from transformers import AutoConfig

from backbone_moe.evaluation import (
    best_by_throughput,
    cache_capacity_for_mem_ratio,
    compute_capacity_knee,
    evaluate_with_fixed_resident_layout,
    infer_frontier_horizon,
    parse_float_list,
    parse_int_list,
    rank_resident_candidates,
    select_feasible_resident_prefix,
)
from finemoe.utils import infer_routed_expert_size_mb, normalize_runtime_config
from backbone_moe.workload import load_state_dict, save_subset_state

from backbone_moe.simulator import SystemBottleneckAnalyzer


def format_mem_tag(mem_ratio: float) -> str:
    text = f"{mem_ratio:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def resolve_expert_size_mb(args):
    if args.expert_size_mb is not None:
        return float(args.expert_size_mb), "cli"
    if args.model_path:
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        config = normalize_runtime_config(config)
        return float(infer_routed_expert_size_mb(config)), f"model:{args.model_path}"
    return 17.2, "default"


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


def select_frontier_prefix(profile_state_file, mem_ratio, args):
    """
    Select the resident size by trace-level feasibility, not throughput sweep.

    1. Rank experts by profiling utility.
    2. Compute F(k): residual batch-step demand frontier after pinning top-k.
    3. Choose the largest feasible prefix k* such that k + F(k) <= B.
    4. Run one simulator evaluation at k* as a sanity check only.
    """
    cache_capacity = cache_capacity_for_mem_ratio(mem_ratio, args.expert_size_mb)
    profile_analyzer = build_analyzer(profile_state_file, args, resident_ratio=0.5)
    ranked = rank_resident_candidates(
        analyzer=profile_analyzer,
        cache_capacity=cache_capacity,
        resident_policy=args.resident_policy,
        resident_profile_ratio=args.resident_profile_ratio,
        resident_depth_power=args.resident_depth_power,
        reset_mode=args.reset_mode,
    )
    knee_capacity = compute_capacity_knee(ranked, cache_capacity)
    selected = select_feasible_resident_prefix(
        ranked=ranked,
        access_sequence=profile_analyzer.access_sequence,
        cache_capacity=cache_capacity,
        frontier_percentile=args.frontier_percentile,
        frontier_horizon=args.frontier_horizon if args.frontier_horizon > 0 else infer_frontier_horizon(
            access_sequence=profile_analyzer.access_sequence,
            expert_size_mb=args.expert_size_mb,
            h2d_bandwidth_gbps=args.h2d_bandwidth_gbps,
            gpu_compute_time_ms=args.gpu_compute_time_ms,
        ),
    )

    resident_capacity = selected["resident_capacity"]
    resident_set = {expert_key for expert_key, _ in ranked[:resident_capacity]}
    rows = evaluate_with_fixed_resident_layout(
        eval_analyzer=profile_analyzer,
        resident_set=resident_set,
        resident_capacity=resident_capacity,
        cache_capacity=cache_capacity,
        mem_ratio=mem_ratio,
        windows=[0],
        reset_mode=args.reset_mode,
    )
    best_row = best_by_throughput(rows)

    best = {
        **selected,
        "best_prefetch_window": 0,
        "throughput_tokens_per_sec": float(best_row["throughput_tokens_per_sec"]),
        "avg_residual_stall_ms": float(best_row["avg_residual_stall_ms"]),
        "num_pinned_residents": int(best_row["num_pinned_residents"]),
        "knee_capacity": int(knee_capacity),
        "knee_ratio": float(knee_capacity / cache_capacity) if cache_capacity > 0 else 0.0,
    }
    candidates = [
        {
            "resident_capacity": int(best["resident_capacity"]),
            "resident_ratio": float(best["resident_ratio"]),
            "speculative_capacity": int(best["speculative_capacity"]),
            "frontier_capacity": int(best["frontier_capacity"]),
            "best_prefetch_window": 0,
            "throughput_tokens_per_sec": float(best["throughput_tokens_per_sec"]),
            "avg_residual_stall_ms": float(best["avg_residual_stall_ms"]),
            "num_pinned_residents": int(best["num_pinned_residents"]),
            "selection_rule": best["selection_rule"],
        }
    ]
    return best, candidates, cache_capacity


def search_best_capacity(profile_state_file, mem_ratio, args):
    """
    Legacy oracle: throughput-sweep resident capacity search.

    Kept only for comparison against the analytic frontier selector. This is not
    the main method and should not be used for paper claims.
    """
    cache_capacity = cache_capacity_for_mem_ratio(mem_ratio, args.expert_size_mb)
    profile_analyzer = build_analyzer(profile_state_file, args, resident_ratio=0.5)
    ranked = rank_resident_candidates(
        analyzer=profile_analyzer,
        cache_capacity=cache_capacity,
        resident_policy=args.resident_policy,
        resident_profile_ratio=args.resident_profile_ratio,
        resident_depth_power=args.resident_depth_power,
        reset_mode=args.reset_mode,
    )

    knee_capacity = compute_capacity_knee(ranked, cache_capacity)
    ranked_experts = [expert_key for expert_key, _ in ranked[:cache_capacity]]
    num_ranked = min(cache_capacity, len(ranked_experts))

    num_steps = min(20, num_ranked - knee_capacity + 1)
    if num_steps <= 1:
        candidate_capacities = [knee_capacity]
    else:
        step = max(1, (num_ranked - knee_capacity) // (num_steps - 1))
        candidate_capacities = list(range(knee_capacity, num_ranked + 1, step))
        if candidate_capacities[-1] != num_ranked:
            candidate_capacities.append(num_ranked)
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
                "selection_rule": "ratio_grid",
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
    ranked = rank_resident_candidates(
        analyzer=analyzer,
        cache_capacity=cache_capacity,
        resident_policy=args.resident_policy,
        resident_profile_ratio=args.resident_profile_ratio,
        resident_depth_power=args.resident_depth_power,
        reset_mode=args.reset_mode,
    )
    resident_prefix = ranked[:best_capacity]
    resident_order = [
        {"layer": int(expert_key[0]), "expert": int(expert_key[1])}
        for expert_key, _ in resident_prefix
    ]
    return {
        "resident_set": list(resident_order),
        "resident_selection_order": list(resident_order),
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
        choices=["frontier_prefix", "capacity_search", "ratio_grid"],
        default="frontier_prefix",
    )
    parser.add_argument("--frontier-percentile", type=float, default=1.0)
    parser.add_argument("--frontier-horizon", type=int, default=0,
                        help="Residual demand burst horizon in batch-steps; <=0 means auto-infer from transfer/compute overlap.")
    parser.add_argument("--prefetch-windows", type=parse_int_list, default=[0, 1, 4, 10])
    parser.add_argument("--profile-fraction", type=float, default=0.2)
    parser.add_argument("--mode", type=str, default="oracle")
    parser.add_argument("--predictor", type=str, default="history_freq")
    parser.add_argument("--resident-policy", type=str, default="profile_freq")
    parser.add_argument("--resident-profile-ratio", type=float, default=0.2)
    parser.add_argument("--resident-depth-power", type=float, default=1.0)
    parser.add_argument("--reset-mode", type=str, default="shared", choices=["shared", "per_sequence"])
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--expert-size-mb", type=float, default=None)
    parser.add_argument("--h2d-bandwidth-gbps", type=float, default=16.0)
    parser.add_argument("--gpu-compute-time-ms", type=float, default=2.0)
    args = parser.parse_args()
    args.expert_size_mb, expert_size_source = resolve_expert_size_mb(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "state_file": args.state_file,
        "profile_fraction": float(args.profile_fraction),
        "resident_policy": args.resident_policy,
        "resident_profile_ratio": float(args.resident_profile_ratio),
        "selection_method": args.selection_method,
        "expert_size_mb": float(args.expert_size_mb),
        "expert_size_source": expert_size_source,
        "frontier_percentile": float(args.frontier_percentile),
        "frontier_horizon": int(args.frontier_horizon),
        "candidate_ratios": [float(x) for x in args.candidate_ratios],
        "prefetch_windows": [int(x) for x in args.prefetch_windows],
        "results": [],
    }

    with tempfile.TemporaryDirectory(prefix="adaptive_resident_") as temp_dir:
        profile_state_file, profile_keys = build_profile_subset(args.state_file, args.profile_fraction, temp_dir)
        summary["profile_sequences"] = len(profile_keys)

        for mem_ratio in args.memory_ratios:
            if args.selection_method == "frontier_prefix":
                best_row, candidates, cache_capacity = select_frontier_prefix(profile_state_file, mem_ratio, args)
                resident_info = export_selected_resident_set(
                    args.state_file,
                    mem_ratio,
                    best_row["resident_ratio"],
                    args,
                    best_capacity=best_row["resident_capacity"],
                )
            elif args.selection_method == "capacity_search":
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

            mem_tag = format_mem_tag(mem_ratio)
            resident_path = args.output_dir / f"{args.output_prefix}_mem{mem_tag}.json"
            resident_payload = {
                **resident_info,
                "selection_method": args.selection_method,
                "selected_resident_ratio": float(best_row["resident_ratio"]),
                "selected_resident_capacity": int(best_row.get("resident_capacity", resident_info["resident_capacity"])),
                "selection_best_prefetch_window": int(best_row.get("best_prefetch_window", 0)),
                "selection_profile_fraction": float(args.profile_fraction),
                "selection_frontier_capacity": int(best_row.get("frontier_capacity", 0)),
                "selection_frontier_percentile": float(best_row.get("frontier_percentile", args.frontier_percentile)),
                "selection_frontier_horizon": int(best_row.get("frontier_horizon", args.frontier_horizon)),
                "selection_knee_capacity": int(best_row.get("knee_capacity", 0)),
                "expert_size_mb": float(args.expert_size_mb),
                "expert_size_source": expert_size_source,
                "selection_candidates": candidates,
            }
            resident_path.write_text(json.dumps(resident_payload, indent=2))
            summary["results"].append(
                {
                    "device_memory_ratio": float(mem_ratio),
                    "selected_resident_ratio": float(best_row["resident_ratio"]),
                    "selected_resident_capacity": int(best_row.get("resident_capacity", resident_info["resident_capacity"])),
                    "best_prefetch_window": int(best_row.get("best_prefetch_window", 0)),
                    "profile_throughput_tokens_per_sec": float(best_row["throughput_tokens_per_sec"]),
                    "frontier_capacity": int(best_row.get("frontier_capacity", 0)),
                    "frontier_horizon": int(best_row.get("frontier_horizon", args.frontier_horizon)),
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
                f"mem={mem_ratio:g}: "
                f"frontier_k={best_row['resident_capacity']} (ratio={best_row['resident_ratio']:.2f}), "
                f"frontier={best_row.get('frontier_capacity', 0)}, "
                f"horizon={best_row.get('frontier_horizon', args.frontier_horizon)}, "
                f"knee_k={knee_k} (ratio={knee_r:.2f}), "
                f"tp={best_row['throughput_tokens_per_sec']:.1f} tok/s"
            )

    summary_path = args.output_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved adaptive split summary to {summary_path}")


if __name__ == "__main__":
    main()
