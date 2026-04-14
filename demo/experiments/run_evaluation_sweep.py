import argparse
import gc
import json
from pathlib import Path

from insight_4_system_bottleneck import SystemBottleneckAnalyzer


CONFIGS = [
    {
        "name": "baseline_single_history",
        "mode": "causal",
        "predictor": "history_freq",
        "cache_layout": "single",
        "resident_policy": "none",
        "resident_ratio": 0.5,
        "resident_profile_ratio": 0.1,
    },
    {
        "name": "baseline_single_oracle",
        "mode": "oracle",
        "predictor": "history_freq",
        "cache_layout": "single",
        "resident_policy": "none",
        "resident_ratio": 0.5,
        "resident_profile_ratio": 0.1,
    },
    {
        "name": "backbone_two_pool_history",
        "mode": "causal",
        "predictor": "history_freq",
        "cache_layout": "two_pool",
        "resident_policy": "profile_freq",
        "resident_ratio": 0.9,
        "resident_profile_ratio": 0.2,
    },
    {
        "name": "backbone_two_pool_oracle",
        "mode": "oracle",
        "predictor": "history_freq",
        "cache_layout": "two_pool",
        "resident_policy": "profile_freq",
        "resident_ratio": 0.9,
        "resident_profile_ratio": 0.2,
    },
]


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def slim_row(row):
    return {
        "device_memory_ratio": float(row["device_memory_ratio"]),
        "prefetch_window": int(row["prefetch_window"]),
        "predictor": row["predictor"],
        "cache_capacity": int(row["cache_capacity"]),
        "resident_capacity": int(row["resident_capacity"]),
        "speculative_capacity": int(row["speculative_capacity"]),
        "resident_policy": row["resident_policy"],
        "num_pinned_residents": int(row["num_pinned_residents"]),
        "throughput_tokens_per_sec": float(row["throughput_tokens_per_sec"]),
        "p95_latency_ms": float(row["p95_latency_ms"]),
        "avg_residual_stall_ms": float(row["avg_residual_stall_ms"]),
        "total_residual_stall_ms": float(row["total_residual_stall_ms"]),
        "hit_rate": float(row["hit_rate"]),
        "prefetch_hit_rate": float(row["prefetch_hit_rate"]),
        "late_prefetch_rate": float(row["late_prefetch_rate"]),
        "funnel_novel_rate": float(row["funnel_novel_rate"]),
        "funnel_useful_rate": float(row["funnel_useful_rate"]),
        "tail_funnel_novel_rate": float(row["tail_funnel_novel_rate"]),
        "tail_funnel_useful_rate": float(row["tail_funnel_useful_rate"]),
    }


def best_by_mem(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(float(row["device_memory_ratio"]), []).append(row)
    return {
        mem: max(group, key=lambda row: row["throughput_tokens_per_sec"])
        for mem, group in grouped.items()
    }


def build_analyzer(args, config):
    return SystemBottleneckAnalyzer(
        state_file=args.state_file,
        mode=config["mode"],
        predictor=config["predictor"],
        output_dir=str(args.output_dir),
        expert_size_mb=args.expert_size_mb,
        h2d_bandwidth_gbps=args.h2d_bandwidth_gbps,
        gpu_compute_time_ms=args.gpu_compute_time_ms,
        prefetch_admission=args.prefetch_admission,
        deadline_margin_ms=args.deadline_margin_ms,
        value_cost_scale=args.value_cost_scale,
        cache_layout=config["cache_layout"],
        resident_ratio=config["resident_ratio"],
        resident_policy=config["resident_policy"],
        resident_profile_ratio=config["resident_profile_ratio"],
        output_tag=args.output_tag,
    )


def run_light_sweep(args, config):
    rows = []
    for mem in args.memory_ratios:
        analyzer = build_analyzer(args, config)
        for window in args.prefetch_windows:
            full_row = analyzer.simulate_with_config(mem, window, reset_mode=args.reset_mode)
            rows.append(slim_row(full_row))
            del full_row
            gc.collect()
        del analyzer
        gc.collect()
    result_dir = args.output_dir / "section5_runs"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{config['name']}_{args.output_tag}.json"
    result_file.write_text(json.dumps(rows, indent=2))
    return rows, result_file


def run_or_load(args, config):
    result_dir = args.output_dir / "section5_runs"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{config['name']}_{args.output_tag}.json"
    if args.skip_existing and result_file.exists():
        rows = load_json(result_file)
    else:
        rows, result_file = run_light_sweep(args, config)
    return rows, result_file


def row_summary(row):
    return {
        "prefetch_window": int(row["prefetch_window"]),
        "throughput_tokens_per_sec": float(row["throughput_tokens_per_sec"]),
        "p95_latency_ms": float(row["p95_latency_ms"]),
        "avg_residual_stall_ms": float(row["avg_residual_stall_ms"]),
        "hit_rate": float(row["hit_rate"]),
        "prefetch_hit_rate": float(row["prefetch_hit_rate"]),
        "late_prefetch_rate": float(row["late_prefetch_rate"]),
        "funnel_novel_rate": float(row["funnel_novel_rate"]),
        "funnel_useful_rate": float(row["funnel_useful_rate"]),
    }


def save_markdown(summary, path):
    lines = []
    lines.append("# Section 5 Evaluation Sweep")
    lines.append("")
    lines.append(f"- state_file: `{summary['state_file']}`")
    lines.append(f"- reset_mode: `{summary['reset_mode']}`")
    lines.append(f"- prefetch_admission: `{summary['prefetch_admission']}`")
    lines.append(f"- memory_ratios: `{summary['memory_ratios']}`")
    lines.append(f"- prefetch_windows: `{summary['prefetch_windows']}`")
    lines.append("")
    lines.append("## Best-Throughput Table")
    lines.append("")
    lines.append("| config | mem | best_w | throughput | p95_ms | stall_ms | hit_rate | prefetch_hit | late_prefetch |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mem_block in summary["best_by_mem"]:
        mem = mem_block["device_memory_ratio"]
        for config_name, row in mem_block["configs"].items():
            lines.append(
                f"| {config_name} | {mem:.2f} | {row['prefetch_window']} | "
                f"{row['throughput_tokens_per_sec']:.2f} | {row['p95_latency_ms']:.2f} | "
                f"{row['avg_residual_stall_ms']:.3f} | {row['hit_rate']:.3f} | "
                f"{row['prefetch_hit_rate']:.3f} | {row['late_prefetch_rate']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Section 5 end-to-end evaluation sweep.")
    parser.add_argument("--state-file", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./experiments/results"))
    parser.add_argument("--summary-json", type=Path, default=Path("./experiments/results/section5_evaluation_summary.json"))
    parser.add_argument("--summary-md", type=Path, default=Path("./experiments/results/section5_evaluation_summary.md"))
    parser.add_argument("--reset-mode", type=str, default="shared", choices=["shared", "per_sequence"])
    parser.add_argument("--prefetch-admission", type=str, default="deadline", choices=["none", "deadline", "value"])
    parser.add_argument("--deadline-margin-ms", type=float, default=0.0)
    parser.add_argument("--value-cost-scale", type=float, default=1.0)
    parser.add_argument("--expert-size-mb", type=float, default=17.2)
    parser.add_argument("--h2d-bandwidth-gbps", type=float, default=16.0)
    parser.add_argument("--gpu-compute-time-ms", type=float, default=2.0)
    parser.add_argument("--memory-ratios", type=parse_float_list, default=[0.05, 0.07, 0.10, 0.15, 0.20])
    parser.add_argument("--prefetch-windows", type=parse_int_list, default=[0, 1, 4, 10])
    parser.add_argument("--output-tag", type=str, default="section5_n64")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)

    config_results = {}
    for config in CONFIGS:
        rows, result_file = run_or_load(args, config)
        config_results[config["name"]] = {
            "result_file": str(result_file),
            "rows": rows,
            "best_by_mem": {str(mem): row_summary(row) for mem, row in best_by_mem(rows).items()},
            "config": config,
        }

    best_table = []
    for mem in args.memory_ratios:
        row = {"device_memory_ratio": float(mem), "configs": {}}
        for config in CONFIGS:
            config_name = config["name"]
            best_row = best_by_mem(config_results[config_name]["rows"])[float(mem)]
            row["configs"][config_name] = row_summary(best_row)
        best_table.append(row)

    summary = {
        "state_file": args.state_file,
        "reset_mode": args.reset_mode,
        "prefetch_admission": args.prefetch_admission,
        "expert_size_mb": args.expert_size_mb,
        "memory_ratios": args.memory_ratios,
        "prefetch_windows": args.prefetch_windows,
        "configs": config_results,
        "best_by_mem": best_table,
    }

    args.summary_json.write_text(json.dumps(summary, indent=2))
    save_markdown(summary, args.summary_md)
    print(f"Saved JSON summary to {args.summary_json}")
    print(f"Saved markdown summary to {args.summary_md}")


if __name__ == "__main__":
    main()
