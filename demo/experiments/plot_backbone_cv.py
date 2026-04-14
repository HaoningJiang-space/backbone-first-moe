import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    0.05: "#9E9E9E",
    0.07: "#1E88E5",
    0.10: "#E53935",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def aggregate_by_mem(data):
    agg = {}
    for row in data["aggregate"]:
        agg[float(row["device_memory_ratio"])] = row
    return agg


def fold_points_by_mem(data):
    by_mem = {}
    for fold in data["folds"]:
        for row in fold["results"]:
            mem = float(row["device_memory_ratio"])
            by_mem.setdefault(mem, []).append(row)
    return by_mem


def aggregate_metric(agg, mems, key):
    means = [agg[mem][key]["mean"] for mem in mems]
    cis = [agg[mem][key]["ci95"] for mem in mems]
    return means, cis


def plot_retained_gain(agg, out_path, key="retained_gain_fraction", title_suffix=""):
    mems = sorted(agg)
    means, cis = aggregate_metric(agg, mems, key)
    colors = [COLORS.get(mem, "#607D8B") for mem in mems]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.bar(range(len(mems)), means, yerr=cis, color=colors, capsize=5, alpha=0.9)
    ax.set_xticks(range(len(mems)))
    ax.set_xticklabels([f"{mem:.2f}" for mem in mems])
    ax.set_xlabel("Memory Ratio")
    ax.set_ylabel("Retained Gain Fraction")
    ax.set_title(f"Resident Backbone Generalization{title_suffix}")
    ax.axhline(1.0, color="#424242", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_ylim(0.0, max(1.1, max(m + c for m, c in zip(means, cis)) + 0.05))
    ax.grid(axis="y", alpha=0.25)

    for idx, (mean, ci) in enumerate(zip(means, cis)):
        ax.text(idx, mean + ci + 0.02, f"{mean:.3f}\n±{ci:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_native_transfer_scatter(by_mem, out_path, x_key="best_test_native", title_suffix="", x_label="Native Backbone Throughput (tokens/s)"):
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    x_all = []
    y_all = []

    for mem in sorted(by_mem):
        rows = by_mem[mem]
        xs = [row[x_key]["throughput_tokens_per_sec"] for row in rows]
        ys = [row["best_transfer"]["throughput_tokens_per_sec"] for row in rows]
        x_all.extend(xs)
        y_all.extend(ys)
        ax.scatter(
            xs,
            ys,
            s=60,
            alpha=0.85,
            color=COLORS.get(mem, "#607D8B"),
            label=f"mem={mem:.2f}",
            edgecolors="white",
            linewidths=0.6,
        )

    lo = min(x_all + y_all)
    hi = max(x_all + y_all)
    margin = 0.04 * (hi - lo)
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "--", color="#424242", linewidth=1.2)
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Transfer Backbone Throughput (tokens/s)")
    ax.set_title(f"Held-out Transfer vs Reference Backbone{title_suffix}")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_regret_forest(agg, out_path, key="transfer_regret_vs_native", title_suffix=""):
    mems = sorted(agg)
    means, cis = aggregate_metric(agg, mems, key)
    y = np.arange(len(mems))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axvline(0.0, color="#424242", linestyle="--", linewidth=1.2, alpha=0.8)
    for idx, mem in enumerate(mems):
        ax.errorbar(
            means[idx],
            y[idx],
            xerr=cis[idx],
            fmt="o",
            color=COLORS.get(mem, "#607D8B"),
            ecolor=COLORS.get(mem, "#607D8B"),
            elinewidth=2.0,
            capsize=4,
            markersize=7,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"mem={mem:.2f}" for mem in mems])
    ax.set_xlabel("Transfer Regret vs Reference (tokens/s)")
    ax.set_title(f"Backbone Transfer Regret{title_suffix}")
    ax.grid(axis="x", alpha=0.25)

    for idx, (mean, ci) in enumerate(zip(means, cis)):
        ax.text(mean + ci + 2.0, y[idx], f"{mean:.1f} ± {ci:.1f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot CV figures for backbone generalization.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="profile_freq_cv8_n64")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_json(args.input_json)
    agg = aggregate_by_mem(data)
    by_mem = fold_points_by_mem(data)

    plot_retained_gain(
        agg,
        args.output_dir / f"backbone_retained_gain_{args.tag}.png",
        key="retained_gain_fraction",
        title_suffix=" vs Profiled Native (8-fold CV, n=64)",
    )
    plot_native_transfer_scatter(
        by_mem,
        args.output_dir / f"backbone_native_vs_transfer_{args.tag}.png",
        x_key="best_test_native",
        title_suffix=" (Profiled Native)",
        x_label="Profiled Native Throughput (tokens/s)",
    )
    plot_regret_forest(
        agg,
        args.output_dir / f"backbone_regret_{args.tag}.png",
        key="transfer_regret_vs_native",
        title_suffix=" vs Profiled Native (95% CI)",
    )

    if all("retained_gain_fraction_vs_oracle_native" in row for row in data.get("aggregate", [])):
        plot_retained_gain(
            agg,
            args.output_dir / f"backbone_retained_gain_vs_oracle_native_{args.tag}.png",
            key="retained_gain_fraction_vs_oracle_native",
            title_suffix=" vs Oracle-Freq Native (8-fold CV, n=64)",
        )
        plot_native_transfer_scatter(
            by_mem,
            args.output_dir / f"backbone_oracle_native_vs_transfer_{args.tag}.png",
            x_key="best_test_oracle_native",
            title_suffix=" (Oracle-Freq Native)",
            x_label="Oracle-Freq Native Throughput (tokens/s)",
        )
        plot_regret_forest(
            agg,
            args.output_dir / f"backbone_regret_vs_oracle_native_{args.tag}.png",
            key="transfer_regret_vs_oracle_native",
            title_suffix=" vs Oracle-Freq Native (95% CI)",
        )

    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
