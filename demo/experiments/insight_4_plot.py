"""
Insight 4 plotting utilities for predictor comparison.

Generates:
  1. Gap-closure figure: history_freq / pl_ctr / utility_freq / oracle
  2. Correlation figure: hit rate vs residual stall against p95 latency
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).parent
DEFAULT_RESULTS = BASE / "experiments" / "results"
DEFAULT_FIGURES = BASE / "figures"


def pick_existing(candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No result file found among: {candidates}")


def load_results(path):
    with open(path) as f:
        return json.load(f)


def build_summary(configs):
    by_mem = {}
    for config in configs:
        by_mem.setdefault(config["device_memory_ratio"], []).append(config)

    best_per_mem = {}
    for mem, group in by_mem.items():
        best_per_mem[mem] = max(group, key=lambda item: item["throughput_tokens_per_sec"])
    return by_mem, best_per_mem


def plot_gap_closure(history_best, pl_ctr_best, utility_best, oracle_best, figure_dir, figure_tag):
    memory_ratios = sorted(set(history_best) & set(pl_ctr_best) & set(utility_best) & set(oracle_best))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        memory_ratios,
        [history_best[mem]["throughput_tokens_per_sec"] for mem in memory_ratios],
        marker="o",
        lw=2,
        color="#607D8B",
        label="History-freq best-window",
    )
    ax.plot(
        memory_ratios,
        [pl_ctr_best[mem]["throughput_tokens_per_sec"] for mem in memory_ratios],
        marker="^",
        lw=2,
        color="#4CAF50",
        label="PL-CTR best-window",
    )
    ax.plot(
        memory_ratios,
        [utility_best[mem]["throughput_tokens_per_sec"] for mem in memory_ratios],
        marker="s",
        lw=2,
        color="#3F51B5",
        label="Utility-freq best-window",
    )
    ax.plot(
        memory_ratios,
        [oracle_best[mem]["throughput_tokens_per_sec"] for mem in memory_ratios],
        marker="*",
        lw=2,
        color="#FF5722",
        label="Oracle best-window",
    )

    oracle_windows = [oracle_best[mem]["prefetch_window"] for mem in memory_ratios]
    for mem, tput, win in zip(
        memory_ratios,
        [oracle_best[mem]["throughput_tokens_per_sec"] for mem in memory_ratios],
        oracle_windows,
    ):
        ax.annotate(
            f"w={win}",
            (mem, tput),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="#FF5722",
        )

    x_min = min(memory_ratios)
    x_max = max(memory_ratios)
    x_margin = max(0.01, 0.02 * (x_max - x_min))
    ax.axvspan(max(0.0, x_min - x_margin), min(0.2, x_max + x_margin), alpha=0.06, color="red")
    if x_max > 0.2:
        ax.axvspan(max(0.2, x_min - x_margin), min(0.35, x_max + x_margin), alpha=0.06, color="gold")
    if x_max > 0.35:
        ax.axvspan(max(0.35, x_min - x_margin), x_max + x_margin, alpha=0.06, color="blue")

    ax.set_xlim(max(0.0, x_min - x_margin), x_max + x_margin)
    ax.set_xticks(memory_ratios)
    ax.set_xlabel("Memory Ratio")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Insight 4 Gap Closure: History-Freq vs PL-CTR vs Utility-Freq vs Oracle")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    out = figure_dir / f"insight4_gap_closure{figure_tag}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_correlation_disconnect(oracle_configs, figure_dir, figure_tag):
    mem_vals = [config["device_memory_ratio"] for config in oracle_configs]
    y_p95 = [config["p95_latency_ms"] for config in oracle_configs]
    x_hit = [config["hit_rate"] for config in oracle_configs]
    x_stall = [config["avg_residual_stall_ms"] for config in oracle_configs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    scatter_kwargs = dict(c=mem_vals, cmap="plasma", s=40, alpha=0.8, edgecolors="none")

    ax1.scatter(x_hit, y_p95, **scatter_kwargs)
    ax1.set_xlabel("Hit Rate")
    ax1.set_ylabel("P95 Latency (ms)")
    ax1.set_title("Hit Rate vs P95 Latency")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(x_stall, y_p95, **scatter_kwargs)
    ax2.set_xlabel("Avg Residual Stall (ms)")
    ax2.set_ylabel("P95 Latency (ms)")
    ax2.set_title("Residual Stall vs P95 Latency")
    ax2.grid(True, alpha=0.3)

    z_hit = np.polyfit(x_hit, y_p95, 1)
    z_stall = np.polyfit(x_stall, y_p95, 1)
    x_hit_line = np.linspace(min(x_hit), max(x_hit), 100)
    x_stall_line = np.linspace(min(x_stall), max(x_stall), 100)
    ax1.plot(x_hit_line, np.poly1d(z_hit)(x_hit_line), "--", color="gray", lw=1.5)
    ax2.plot(x_stall_line, np.poly1d(z_stall)(x_stall_line), "--", color="gray", lw=1.5)

    outlier = next(
        (
            config for config in oracle_configs
            if config["device_memory_ratio"] == 0.2 and config["prefetch_window"] == 10
        ),
        None,
    )
    if outlier is not None:
        ax1.annotate(
            f"mem=0.2 w=10\nhit={outlier['hit_rate']:.3f}\np95={outlier['p95_latency_ms']:.0f}ms",
            (outlier["hit_rate"], outlier["p95_latency_ms"]),
            textcoords="offset points",
            xytext=(-55, -22),
            fontsize=8,
            color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        )
        ax2.annotate(
            f"mem=0.2 w=10\nstall={outlier['avg_residual_stall_ms']:.1f}ms\np95={outlier['p95_latency_ms']:.0f}ms",
            (outlier["avg_residual_stall_ms"], outlier["p95_latency_ms"]),
            textcoords="offset points",
            xytext=(-80, -28),
            fontsize=8,
            color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        )

    cbar = fig.colorbar(ax1.collections[0], ax=(ax1, ax2), orientation="vertical", pad=0.02, aspect=30)
    cbar.set_label("Memory Ratio")

    fig.tight_layout()
    out = figure_dir / f"insight4_correlation_disconnect{figure_tag}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def print_best_by_mem(name, best_per_mem):
    print(f"\n{name} best-window per memory:")
    print(f"  {'mem':>5} {'w*':>3} {'tput':>8} {'p95':>7} {'stall':>8} {'hit':>7}")
    for mem in sorted(best_per_mem):
        config = best_per_mem[mem]
        print(
            f"  {mem:>5.2f} {config['prefetch_window']:>3} "
            f"{config['throughput_tokens_per_sec']:>8.2f} "
            f"{config['p95_latency_ms']:>7.2f} "
            f"{config['avg_residual_stall_ms']:>8.2f} "
            f"{config['hit_rate']:>7.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Plot Insight 4 predictor-comparison figures")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--history-file", type=Path, default=None)
    parser.add_argument("--pl-ctr-file", type=Path, default=None)
    parser.add_argument("--utility-file", type=Path, default=None)
    parser.add_argument("--oracle-file", type=Path, default=None)
    parser.add_argument("--figure-tag", type=str, default="")
    args = parser.parse_args()

    results_dir = args.results_dir
    figures_dir = args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_tag = f"_{args.figure_tag}" if args.figure_tag else ""

    history_file = args.history_file or pick_existing([
        results_dir / "insight_4_modecausal_predictorhistory_freq_resetshared_e17p2_lowmem.json",
        results_dir / "insight_4_modecausal_predictorhistory_freq_resetshared_e17p2.json",
        results_dir / "insight_4_modecausal_predictorhistory_freq_resetshared.json",
        results_dir / "insight_4_modecausal_resetshared.json",
    ])
    pl_ctr_file = args.pl_ctr_file or pick_existing([
        results_dir / "insight_4_modecausal_predictorpl_ctr_resetshared_e17p2_lowmem.json",
        results_dir / "insight_4_modecausal_predictorpl_ctr_resetshared_e17p2.json",
        results_dir / "insight_4_modecausal_predictorpl_ctr_resetshared.json",
    ])
    utility_file = args.utility_file or pick_existing([
        results_dir / "insight_4_modecausal_predictorutility_freq_resetshared_e17p2_lowmem.json",
        results_dir / "insight_4_modecausal_predictorutility_freq_resetshared_e17p2.json",
        results_dir / "insight_4_modecausal_predictorutility_freq_resetshared.json",
    ])
    oracle_file = args.oracle_file or pick_existing([
        results_dir / "insight_4_modeoracle_resetshared_e17p2_lowmem.json",
        results_dir / "insight_4_modeoracle_resetshared_e17p2.json",
        results_dir / "insight_4_modeoracle_resetshared.json",
    ])

    print(f"History-freq file: {history_file}")
    print(f"PL-CTR file:      {pl_ctr_file}")
    print(f"Utility-freq file:{utility_file}")
    print(f"Oracle file:      {oracle_file}")

    history_configs = load_results(history_file)
    pl_ctr_configs = load_results(pl_ctr_file)
    utility_configs = load_results(utility_file)
    oracle_configs = load_results(oracle_file)

    _, history_best = build_summary(history_configs)
    _, pl_ctr_best = build_summary(pl_ctr_configs)
    _, utility_best = build_summary(utility_configs)
    _, oracle_best = build_summary(oracle_configs)

    print_best_by_mem("History-freq", history_best)
    print_best_by_mem("PL-CTR", pl_ctr_best)
    print_best_by_mem("Utility-freq", utility_best)
    print_best_by_mem("Oracle", oracle_best)

    plot_gap_closure(history_best, pl_ctr_best, utility_best, oracle_best, figures_dir, figure_tag)
    plot_correlation_disconnect(oracle_configs, figures_dir, figure_tag)


if __name__ == "__main__":
    main()
