import argparse
import gc
import json
from pathlib import Path

from backbone_moe.evaluation import (
    best_by_throughput,
    build_two_pool_analyzer,
    parse_float_list,
    parse_int_list,
)
from backbone_moe.metrics import mean_and_ci95
from backbone_moe.workload import load_state_dict, save_subset_state
from backbone_moe.simulator import SystemBottleneckAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Augment backbone CV JSON with oracle-native metrics.")
    parser.add_argument("--base-json", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=str, default="./experiments/results")
    parser.add_argument("--expert-size-mb", type=float, default=17.2)
    parser.add_argument("--h2d-bandwidth-gbps", type=float, default=16.0)
    parser.add_argument("--gpu-compute-time-ms", type=float, default=2.0)
    parser.add_argument("--resident-ratio", type=float, default=0.9)
    parser.add_argument("--resident-profile-ratio", type=float, default=0.2)
    parser.add_argument("--memory-ratios", type=parse_float_list, default=None)
    parser.add_argument("--prefetch-windows", type=parse_int_list, default=None)
    parser.add_argument("--reset-mode", type=str, default="shared", choices=["shared", "per_sequence"])
    args = parser.parse_args()

    data = json.loads(args.base_json.read_text())
    full_state = load_state_dict(args.state_file)

    memory_ratios = args.memory_ratios or data.get("memory_ratios", [0.05, 0.07, 0.10])
    prefetch_windows = args.prefetch_windows or data.get("prefetch_windows", [0, 1, 4, 10])

    tmp_dir = args.output_json.parent / "oracle_native_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    metric_buckets = {}
    for mem in memory_ratios:
        metric_buckets[(mem, "oracle_native_gain_over_single")] = []
        metric_buckets[(mem, "transfer_regret_vs_oracle_native")] = []
        metric_buckets[(mem, "retained_gain_fraction_vs_oracle_native")] = []

    if "folds" not in data:
        raise ValueError("base-json must be a k-fold CV result with 'folds'")

    for fold in data["folds"]:
        test_keys = fold["test_sequences"]
        test_state = tmp_dir / f"test_fold_{fold['fold_index']}.pkl"
        save_subset_state(test_state, full_state, test_keys)

        analyzer = build_two_pool_analyzer(
            SystemBottleneckAnalyzer,
            args,
            test_state,
            resident_policy="oracle_freq",
        )
        rows_by_mem = {}
        for mem in memory_ratios:
            rows = [
                analyzer.simulate_with_config(mem, window, reset_mode=args.reset_mode)
                for window in prefetch_windows
            ]
            rows_by_mem[mem] = best_by_throughput(rows)
        del analyzer
        gc.collect()

        for row in fold["results"]:
            mem = float(row["device_memory_ratio"])
            best_oracle_native = rows_by_mem[mem]
            oracle_native_gain = (
                best_oracle_native["throughput_tokens_per_sec"]
                - row["best_single"]["throughput_tokens_per_sec"]
            )
            transfer_regret_vs_oracle_native = (
                best_oracle_native["throughput_tokens_per_sec"]
                - row["best_transfer"]["throughput_tokens_per_sec"]
            )
            retained_vs_oracle_native = None
            if oracle_native_gain > 0:
                retained_vs_oracle_native = row["transfer_gain_over_single"] / oracle_native_gain

            row["best_test_oracle_native"] = {
                "prefetch_window": best_oracle_native["prefetch_window"],
                "throughput_tokens_per_sec": best_oracle_native["throughput_tokens_per_sec"],
                "avg_residual_stall_ms": best_oracle_native["avg_residual_stall_ms"],
            }
            row["oracle_native_gain_over_single"] = oracle_native_gain
            row["transfer_regret_vs_oracle_native"] = transfer_regret_vs_oracle_native
            row["retained_gain_fraction_vs_oracle_native"] = retained_vs_oracle_native

            metric_buckets[(mem, "oracle_native_gain_over_single")].append(oracle_native_gain)
            metric_buckets[(mem, "transfer_regret_vs_oracle_native")].append(transfer_regret_vs_oracle_native)
            if retained_vs_oracle_native is not None:
                metric_buckets[(mem, "retained_gain_fraction_vs_oracle_native")].append(retained_vs_oracle_native)

    data["results"] = [row for fold in data["folds"] for row in fold["results"]]
    if "aggregate" in data:
        for row in data["aggregate"]:
            mem = float(row["device_memory_ratio"])
            row["oracle_native_gain_over_single"] = mean_and_ci95(metric_buckets[(mem, "oracle_native_gain_over_single")])
            row["transfer_regret_vs_oracle_native"] = mean_and_ci95(metric_buckets[(mem, "transfer_regret_vs_oracle_native")])
            row["retained_gain_fraction_vs_oracle_native"] = mean_and_ci95(
                metric_buckets[(mem, "retained_gain_fraction_vs_oracle_native")]
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(data, indent=2))
    print(f"Saved augmented CV JSON to {args.output_json}")


if __name__ == "__main__":
    main()
