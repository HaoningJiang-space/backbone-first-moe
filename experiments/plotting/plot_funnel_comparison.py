import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PREDICTOR_STYLES = {
    "history_freq": {"label": "History-freq", "color": "#607D8B"},
    "pl_ctr": {"label": "PL-CTR", "color": "#43A047"},
    "oracle": {"label": "Oracle", "color": "#E64A19"},
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def best_by_mem(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(float(row["device_memory_ratio"]), []).append(row)
    return {mem: max(group, key=lambda x: x["throughput_tokens_per_sec"]) for mem, group in grouped.items()}


def stage_breakdown(row):
    predicted = float(row["funnel_predicted"])
    novel = float(row["funnel_novel"])
    admission = float(row["funnel_admission_dropped"])
    deadline = float(row["funnel_deadline_rejected"])
    late = float(row["funnel_late"])
    useful = float(row["funnel_useful"])
    timely = float(row["funnel_timely"])

    already_cached = max(0.0, predicted - novel)
    timely_not_useful = max(0.0, timely - useful)
    pieces = {
        "Already cached": already_cached,
        "Admission drop": admission,
        "Deadline reject": deadline,
        "Late prefetch": late,
        "Timely but not useful": timely_not_useful,
        "Useful": useful,
    }
    return pieces, predicted


def plot_novel_rate(best_rows, output_path):
    mems = sorted(set.intersection(*(set(rows.keys()) for rows in best_rows.values())))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for predictor, rows in best_rows.items():
        style = PREDICTOR_STYLES[predictor]
        ax.plot(
            mems,
            [rows[mem]["funnel_novel_rate"] for mem in mems],
            marker="o",
            lw=2,
            color=style["color"],
            label=style["label"],
        )

    ax.set_xlabel("Memory Ratio")
    ax.set_ylabel("Novel Rate")
    ax.set_title("Best-Window Novel Rate by Predictor")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    ax.set_xticks(mems)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stacked_funnel(best_rows, focus_mem, output_path):
    labels = []
    stage_names = [
        "Already cached",
        "Admission drop",
        "Deadline reject",
        "Late prefetch",
        "Timely but not useful",
        "Useful",
    ]
    colors = {
        "Already cached": "#B0BEC5",
        "Admission drop": "#8E24AA",
        "Deadline reject": "#FB8C00",
        "Late prefetch": "#EF5350",
        "Timely but not useful": "#5C6BC0",
        "Useful": "#26A69A",
    }

    normalized = {stage: [] for stage in stage_names}
    annotations = []

    for predictor in ["history_freq", "pl_ctr", "oracle"]:
        row = best_rows[predictor][focus_mem]
        pieces, predicted = stage_breakdown(row)
        labels.append(
            f"{PREDICTOR_STYLES[predictor]['label']}\n"
            f"w={row['prefetch_window']}, mem={focus_mem:.2f}"
        )
        denom = max(1.0, predicted)
        for stage in stage_names:
            normalized[stage].append(pieces[stage] / denom)
        annotations.append({
            "novel_rate": row["funnel_novel_rate"],
            "useful_rate": row["funnel_useful_rate"],
            "throughput": row["throughput_tokens_per_sec"],
        })

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    for stage in stage_names:
        vals = np.array(normalized[stage])
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=colors[stage],
            edgecolor="white",
            linewidth=0.6,
            label=stage,
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of Predicted Experts")
    ax.set_title(f"Funnel Breakdown at mem={focus_mem:.2f} (Best Window per Predictor)")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)

    for idx, ann in enumerate(annotations):
        ax.text(
            idx,
            1.02,
            f"novel={ann['novel_rate']:.3f}\nuseful={ann['useful_rate']:.3f}\ntput={ann['throughput']:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot cross-predictor funnel comparison figures.")
    parser.add_argument("--history-file", type=Path, required=True)
    parser.add_argument("--pl-ctr-file", type=Path, required=True)
    parser.add_argument("--oracle-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="admdeadline")
    parser.add_argument("--focus-mem", type=float, default=0.10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_rows = {
        "history_freq": best_by_mem(load_json(args.history_file)),
        "pl_ctr": best_by_mem(load_json(args.pl_ctr_file)),
        "oracle": best_by_mem(load_json(args.oracle_file)),
    }

    plot_novel_rate(best_rows, args.output_dir / f"funnel_novel_rate_{args.tag}.png")
    plot_stacked_funnel(best_rows, args.focus_mem, args.output_dir / f"funnel_breakdown_mem{args.focus_mem:.2f}_{args.tag}.png")

    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
