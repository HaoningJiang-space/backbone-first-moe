import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

from finemoe.backbone.evaluation import (
    best_by_throughput,
    build_single_cache_analyzer,
    build_two_pool_analyzer,
    evaluate_with_fixed_resident_set,
    parse_float_list,
    parse_int_list,
    resident_set_from_analyzer,
)
from finemoe.backbone.metrics import mean_and_ci95
from finemoe.backbone.workload import (
    build_kfold_splits,
    load_state_dict,
    save_subset_state,
    split_sequence_keys,
)
from insight_4_system_bottleneck import SystemBottleneckAnalyzer


def evaluate_train_test_pair(args, train_state, test_state, train_keys, test_keys):
    pair_results = []
    for mem in args.memory_ratios:
        train_analyzer = build_two_pool_analyzer(SystemBottleneckAnalyzer, args, str(train_state))
        train_resident = resident_set_from_analyzer(train_analyzer, mem, args.reset_mode)
        del train_analyzer
        gc.collect()

        test_analyzer = build_two_pool_analyzer(SystemBottleneckAnalyzer, args, str(test_state))
        test_resident = resident_set_from_analyzer(test_analyzer, mem, args.reset_mode)
        transfer_rows = evaluate_with_fixed_resident_set(
            test_analyzer, train_resident, mem, args.prefetch_windows, args.reset_mode
        )
        oracle_rows = [
            test_analyzer.simulate_with_config(mem, window, reset_mode=args.reset_mode)
            for window in args.prefetch_windows
        ]
        del test_analyzer
        gc.collect()

        test_oracle_native_analyzer = build_two_pool_analyzer(
            SystemBottleneckAnalyzer, args, str(test_state), resident_policy="oracle_freq"
        )
        oracle_native_rows = [
            test_oracle_native_analyzer.simulate_with_config(mem, window, reset_mode=args.reset_mode)
            for window in args.prefetch_windows
        ]
        del test_oracle_native_analyzer
        gc.collect()

        single_baseline = build_single_cache_analyzer(SystemBottleneckAnalyzer, args, test_state)
        single_rows = [
            single_baseline.simulate_with_config(mem, window, reset_mode=args.reset_mode)
            for window in args.prefetch_windows
        ]
        del single_baseline
        gc.collect()

        best_transfer = best_by_throughput(transfer_rows)
        best_oracle = best_by_throughput(oracle_rows)
        best_oracle_native = best_by_throughput(oracle_native_rows)
        best_single = best_by_throughput(single_rows)

        overlap = len(train_resident & test_resident) / max(1, len(train_resident | test_resident))
        native_gain = best_oracle["throughput_tokens_per_sec"] - best_single["throughput_tokens_per_sec"]
        oracle_native_gain = best_oracle_native["throughput_tokens_per_sec"] - best_single["throughput_tokens_per_sec"]
        transfer_gain = best_transfer["throughput_tokens_per_sec"] - best_single["throughput_tokens_per_sec"]
        retained_gain_fraction = None
        retained_gain_fraction_vs_oracle_native = None
        if native_gain > 0:
            retained_gain_fraction = transfer_gain / native_gain
        if oracle_native_gain > 0:
            retained_gain_fraction_vs_oracle_native = transfer_gain / oracle_native_gain

        pair_results.append({
            "device_memory_ratio": mem,
            "train_resident_size": len(train_resident),
            "test_resident_size": len(test_resident),
            "train_test_jaccard": overlap,
            "best_transfer": {
                "prefetch_window": best_transfer["prefetch_window"],
                "throughput_tokens_per_sec": best_transfer["throughput_tokens_per_sec"],
                "avg_residual_stall_ms": best_transfer["avg_residual_stall_ms"],
            },
            "best_test_native": {
                "prefetch_window": best_oracle["prefetch_window"],
                "throughput_tokens_per_sec": best_oracle["throughput_tokens_per_sec"],
                "avg_residual_stall_ms": best_oracle["avg_residual_stall_ms"],
            },
            "best_test_oracle_native": {
                "prefetch_window": best_oracle_native["prefetch_window"],
                "throughput_tokens_per_sec": best_oracle_native["throughput_tokens_per_sec"],
                "avg_residual_stall_ms": best_oracle_native["avg_residual_stall_ms"],
            },
            "best_single": {
                "prefetch_window": best_single["prefetch_window"],
                "throughput_tokens_per_sec": best_single["throughput_tokens_per_sec"],
                "avg_residual_stall_ms": best_single["avg_residual_stall_ms"],
            },
            "native_gain_over_single": native_gain,
            "oracle_native_gain_over_single": oracle_native_gain,
            "transfer_gain_over_single": transfer_gain,
            "transfer_regret_vs_native": best_oracle["throughput_tokens_per_sec"] - best_transfer["throughput_tokens_per_sec"],
            "transfer_regret_vs_oracle_native": best_oracle_native["throughput_tokens_per_sec"] - best_transfer["throughput_tokens_per_sec"],
            "retained_gain_fraction": retained_gain_fraction,
            "retained_gain_fraction_vs_oracle_native": retained_gain_fraction_vs_oracle_native,
        })
    return pair_results


def main():
    parser = argparse.ArgumentParser(description="Generalization analysis for resident backbone")
    parser.add_argument("--state-file", type=str,
                        default="../states/Qwen1.5-MoE-A2.7B-Chat~lmsys-chat-1m~8.pkl")
    parser.add_argument("--train-state-file", type=str, default=None)
    parser.add_argument("--test-state-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./experiments/results")
    parser.add_argument("--reset-mode", type=str, default="shared", choices=["shared", "per_sequence"])
    parser.add_argument("--expert-size-mb", type=float, default=17.2)
    parser.add_argument("--h2d-bandwidth-gbps", type=float, default=16.0)
    parser.add_argument("--gpu-compute-time-ms", type=float, default=2.0)
    parser.add_argument("--memory-ratios", type=parse_float_list, default=[0.05, 0.07, 0.10])
    parser.add_argument("--prefetch-windows", type=parse_int_list, default=[0, 1, 4, 10])
    parser.add_argument("--resident-ratio", type=float, default=0.9)
    parser.add_argument("--resident-policy", type=str, default="profile_freq")
    parser.add_argument("--resident-profile-ratio", type=float, default=0.2)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--cv-folds", type=int, default=0)
    args = parser.parse_args()

    explicit_transfer = args.train_state_file is not None and args.test_state_file is not None
    if explicit_transfer:
        train_state = Path(args.train_state_file)
        test_state = Path(args.test_state_file)
        train_state_dict = load_state_dict(train_state)
        test_state_dict = load_state_dict(test_state)
        train_keys = list(train_state_dict.keys())
        test_keys = list(test_state_dict.keys())
    else:
        full_state = load_state_dict(args.state_file)
        seq_keys = list(full_state.keys())
        train_keys, test_keys = split_sequence_keys(seq_keys, args.train_fraction)

        tmp_dir = Path(args.output_dir) / "generalization_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        train_state = tmp_dir / "train_state.pkl"
        test_state = tmp_dir / "test_state.pkl"
        save_subset_state(train_state, full_state, train_keys)
        save_subset_state(test_state, full_state, test_keys)

    output = {
        "analysis": "backbone_generalization",
        "mode": "explicit_transfer" if explicit_transfer else "split_transfer",
        "train_fraction": args.train_fraction,
        "train_sequences": train_keys,
        "test_sequences": test_keys,
        "train_state_file": str(train_state),
        "test_state_file": str(test_state),
        "resident_policy": args.resident_policy,
        "resident_ratio": args.resident_ratio,
        "resident_profile_ratio": args.resident_profile_ratio,
        "memory_ratios": args.memory_ratios,
        "prefetch_windows": args.prefetch_windows,
        "results": [],
    }

    if not explicit_transfer and args.cv_folds and args.cv_folds > 1:
        full_state = load_state_dict(args.state_file)
        seq_keys = list(full_state.keys())
        tmp_dir = Path(args.output_dir) / "generalization_tmp_cv"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        output["mode"] = "kfold_cv"
        output["cv_folds"] = args.cv_folds
        output["folds"] = []

        metric_buckets = defaultdict(list)

        for fold in build_kfold_splits(seq_keys, args.cv_folds):
            fold_idx = fold["fold_index"]
            train_keys = fold["train_sequences"]
            test_keys = fold["test_sequences"]

            train_state = tmp_dir / f"train_fold_{fold_idx}.pkl"
            test_state = tmp_dir / f"test_fold_{fold_idx}.pkl"
            save_subset_state(train_state, full_state, train_keys)
            save_subset_state(test_state, full_state, test_keys)

            fold_results = evaluate_train_test_pair(args, train_state, test_state, train_keys, test_keys)
            output["folds"].append({
                "fold_index": fold_idx,
                "train_sequences": train_keys,
                "test_sequences": test_keys,
                "results": fold_results,
            })
            for row in fold_results:
                mem = row["device_memory_ratio"]
                metric_buckets[(mem, "train_test_jaccard")].append(row["train_test_jaccard"])
                metric_buckets[(mem, "native_gain_over_single")].append(row["native_gain_over_single"])
                metric_buckets[(mem, "oracle_native_gain_over_single")].append(row["oracle_native_gain_over_single"])
                metric_buckets[(mem, "transfer_gain_over_single")].append(row["transfer_gain_over_single"])
                metric_buckets[(mem, "transfer_regret_vs_native")].append(row["transfer_regret_vs_native"])
                metric_buckets[(mem, "transfer_regret_vs_oracle_native")].append(row["transfer_regret_vs_oracle_native"])
                if row["retained_gain_fraction"] is not None:
                    metric_buckets[(mem, "retained_gain_fraction")].append(row["retained_gain_fraction"])
                if row["retained_gain_fraction_vs_oracle_native"] is not None:
                    metric_buckets[(mem, "retained_gain_fraction_vs_oracle_native")].append(
                        row["retained_gain_fraction_vs_oracle_native"]
                    )

            output["results"].extend(fold_results)

        output["aggregate"] = []
        for mem in args.memory_ratios:
            output["aggregate"].append({
                "device_memory_ratio": mem,
                "train_test_jaccard": mean_and_ci95(metric_buckets[(mem, "train_test_jaccard")]),
                "native_gain_over_single": mean_and_ci95(metric_buckets[(mem, "native_gain_over_single")]),
                "oracle_native_gain_over_single": mean_and_ci95(metric_buckets[(mem, "oracle_native_gain_over_single")]),
                "transfer_gain_over_single": mean_and_ci95(metric_buckets[(mem, "transfer_gain_over_single")]),
                "transfer_regret_vs_native": mean_and_ci95(metric_buckets[(mem, "transfer_regret_vs_native")]),
                "transfer_regret_vs_oracle_native": mean_and_ci95(metric_buckets[(mem, "transfer_regret_vs_oracle_native")]),
                "retained_gain_fraction": mean_and_ci95(metric_buckets[(mem, "retained_gain_fraction")]),
                "retained_gain_fraction_vs_oracle_native": mean_and_ci95(
                    metric_buckets[(mem, "retained_gain_fraction_vs_oracle_native")]
                ),
            })
    else:
        output["results"] = evaluate_train_test_pair(args, train_state, test_state, train_keys, test_keys)

    output_path = Path(args.output_dir) / "backbone_generalization.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved generalization summary to {output_path}")


if __name__ == "__main__":
    main()
