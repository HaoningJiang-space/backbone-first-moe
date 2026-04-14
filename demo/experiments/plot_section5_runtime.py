import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


CONFIG_ORDER = [
    ("A", "A_demand", "Demand-only"),
    ("B", "B_lru_prefetch", "LRU+Prefetch"),
    ("C", "C_backbone_only", "Backbone-only"),
    ("D", "D_backbone", "Backbone-first"),
]


def parse_mem(label):
    token = label.split("_mem", 1)[1].replace("p", ".")
    return float(token)


def config_key(label):
    for short, prefix, title in CONFIG_ORDER:
        if label.startswith(prefix):
            return short, title
    raise KeyError(f"Unknown config label: {label}")


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def collect_rows(summary):
    rows = {}
    for label, payload in summary["results"].items():
        mem = parse_mem(label)
        short, title = config_key(label)
        rows[(mem, short)] = {
            "label": label,
            "config_title": title,
            "generated_tokens_per_sec": float(payload["generated_tokens_per_sec"]),
            "end_to_end_tokens_per_sec": float(payload["end_to_end_tokens_per_sec"]),
            "total_elapsed_sec": float(payload["total_elapsed_sec"]),
            "peak_memory_mb": None if payload["peak_memory_mb"] is None else float(payload["peak_memory_mb"]),
            "resident_count": int(payload.get("resident_count", 0)),
        }
    return rows


def build_markdown(rows, mems):
    lines = []
    lines.append("# Section 5 Runtime Summary")
    lines.append("")
    lines.append("| mem | config | gen tok/s | e2e tok/s | elapsed(s) | peak MB | resident | delta vs A (e2e) | delta vs B (e2e) |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for mem in mems:
        base_a = rows.get((mem, "A"))
        base_b = rows.get((mem, "B"))
        for short, _, title in CONFIG_ORDER:
            row = rows.get((mem, short))
            if row is None:
                continue
            delta_a = ""
            delta_b = ""
            if base_a is not None:
                delta_a = f"{row['end_to_end_tokens_per_sec'] - base_a['end_to_end_tokens_per_sec']:+.2f}"
            if base_b is not None:
                delta_b = f"{row['end_to_end_tokens_per_sec'] - base_b['end_to_end_tokens_per_sec']:+.2f}"
            peak = "N/A" if row["peak_memory_mb"] is None else f"{row['peak_memory_mb']:.0f}"
            lines.append(
                f"| {mem:.2f} | {title} | {row['generated_tokens_per_sec']:.3f} | "
                f"{row['end_to_end_tokens_per_sec']:.2f} | {row['total_elapsed_sec']:.2f} | "
                f"{peak} | {row['resident_count']} | {delta_a} | {delta_b} |"
            )
    return "\n".join(lines) + "\n"


def plot_metric(rows, mems, metric, ylabel, output_path):
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = range(len(mems))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for (offset, (short, _, title)) in zip(offsets, CONFIG_ORDER):
        values = [rows[(mem, short)][metric] if (mem, short) in rows else 0.0 for mem in mems]
        ax.bar([i + offset for i in x], values, width=width, label=title)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{mem:.2f}" for mem in mems])
    ax.set_xlabel("Device Memory Ratio")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Summarize and plot runtime Section 5 results.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="section5_runtime")
    args = parser.parse_args()

    summary = load_summary(args.summary_json)
    rows = collect_rows(summary)
    mems = sorted({mem for mem, _ in rows})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / f"{args.prefix}_summary.md"
    md_path.write_text(build_markdown(rows, mems))

    plot_metric(
        rows,
        mems,
        "generated_tokens_per_sec",
        "Generated Tokens / Sec",
        args.output_dir / f"{args.prefix}_generated_tps.png",
    )
    plot_metric(
        rows,
        mems,
        "end_to_end_tokens_per_sec",
        "End-to-End Tokens / Sec",
        args.output_dir / f"{args.prefix}_e2e_tps.png",
    )
    plot_metric(
        rows,
        mems,
        "peak_memory_mb",
        "Peak GPU Memory (MB)",
        args.output_dir / f"{args.prefix}_peak_memory.png",
    )
    print(f"Saved markdown summary to {md_path}")
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
