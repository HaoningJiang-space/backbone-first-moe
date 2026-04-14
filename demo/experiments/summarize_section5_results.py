import argparse
import json
from pathlib import Path


CONFIG_ORDER = [
    "baseline_single_history",
    "baseline_single_oracle",
    "backbone_two_pool_history",
    "backbone_two_pool_oracle",
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_config_file(text):
    name, rhs = text.split("=", 1)
    files = [Path(item.strip()) for item in rhs.split(",") if item.strip()]
    return name.strip(), files


def slim_row(row):
    return {
        "device_memory_ratio": float(row["device_memory_ratio"]),
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


def normalize_rows(rows):
    return [slim_row(row) for row in rows]


def best_by_mem(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(float(row["device_memory_ratio"]), []).append(row)
    return {
        mem: max(group, key=lambda row: row["throughput_tokens_per_sec"])
        for mem, group in grouped.items()
    }


def save_markdown(summary, path):
    lines = []
    lines.append("# Section 5 Evaluation Summary")
    lines.append("")
    lines.append("| config | mem | best_w | throughput | p95_ms | stall_ms | hit_rate | prefetch_hit | late_prefetch | novel_rate | useful_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mem_block in summary["best_by_mem"]:
        mem = mem_block["device_memory_ratio"]
        for config_name in CONFIG_ORDER:
            row = mem_block["configs"][config_name]
            lines.append(
                f"| {config_name} | {mem:.2f} | {row['prefetch_window']} | "
                f"{row['throughput_tokens_per_sec']:.2f} | {row['p95_latency_ms']:.2f} | "
                f"{row['avg_residual_stall_ms']:.3f} | {row['hit_rate']:.3f} | "
                f"{row['prefetch_hit_rate']:.3f} | {row['late_prefetch_rate']:.3f} | "
                f"{row['funnel_novel_rate']:.3f} | {row['funnel_useful_rate']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize chunked Section 5 result JSON files.")
    parser.add_argument("--config-file", action="append", required=True,
                        help="Format: config_name=file1.json,file2.json")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    config_rows = {}
    for item in args.config_file:
        config_name, files = parse_config_file(item)
        rows = []
        for path in files:
            rows.extend(normalize_rows(load_json(path)))
        rows.sort(key=lambda row: (row["device_memory_ratio"], row["prefetch_window"]))
        config_rows[config_name] = rows

    all_mems = sorted({row["device_memory_ratio"] for rows in config_rows.values() for row in rows})
    best_table = []
    for mem in all_mems:
        block = {"device_memory_ratio": mem, "configs": {}}
        for config_name in CONFIG_ORDER:
            row = best_by_mem(config_rows[config_name])[mem]
            block["configs"][config_name] = row
        best_table.append(block)

    summary = {
        "configs": config_rows,
        "best_by_mem": best_table,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    save_markdown(summary, args.output_md)
    print(f"Saved JSON summary to {args.output_json}")
    print(f"Saved markdown summary to {args.output_md}")


if __name__ == "__main__":
    main()
