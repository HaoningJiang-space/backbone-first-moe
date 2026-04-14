import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STYLE_BY_KEY = {
    "oracle_single": {"label": "Oracle / single", "color": "#E64A19"},
    "oracle_two_pool_profile_freq": {"label": "Oracle / resident-pinned", "color": "#1E88E5"},
    "history_freq_single": {"label": "History-freq / single", "color": "#607D8B"},
    "history_freq_two_pool_profile_freq": {"label": "History-freq / resident-pinned", "color": "#3949AB"},
    "pl_ctr_single": {"label": "PL-CTR / single", "color": "#43A047"},
    "utility_freq_single": {"label": "Utility-freq / single", "color": "#8E24AA"},
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def style_for_profile(profile):
    predictor = profile["predictor"]
    layout = profile.get("cache_layout") or profile.get("config", {}).get("cache_layout", "single")
    resident_policy = profile.get("resident_policy") or profile.get("config", {}).get("resident_policy", "none")
    if layout == "two_pool" and resident_policy == "profile_freq":
        key = f"{predictor}_two_pool_profile_freq"
        if key in STYLE_BY_KEY:
            return STYLE_BY_KEY[key]
    key = f"{predictor}_single"
    if key in STYLE_BY_KEY:
        return STYLE_BY_KEY[key]
    key = predictor
    if key in STYLE_BY_KEY:
        return STYLE_BY_KEY[key]
    return {"label": predictor, "color": "#424242"}


def metric_entries_to_matrix(entries):
    if not entries:
        return np.zeros((1, 1), dtype=float)
    max_layer = max(int(item["layer"]) for item in entries)
    max_expert = max(int(item["expert"]) for item in entries)
    matrix = np.zeros((max_layer + 1, max_expert + 1), dtype=float)
    for item in entries:
        matrix[int(item["layer"]), int(item["expert"])] = float(item["value"])
    return matrix


def plot_lorenz(profiles, output_path):
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#BDBDBD", label="Uniform")
    for profile in profiles:
        style = style_for_profile(profile)
        curve = profile["stall_lorenz"]
        coverage20 = profile["stall_top_coverage"].get("0.2", 0.0)
        ax.plot(
            curve["fractions"],
            curve["coverage"],
            lw=2.2,
            color=style["color"],
            label=f"{style['label']} (top20%={coverage20:.2f})",
        )
    ax.set_xlabel("Fraction of Experts")
    ax.set_ylabel("Fraction of Critical Stall")
    ax.set_title("Critical Stall Lorenz Curve")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(profiles, output_path):
    matrices = [metric_entries_to_matrix(profile["per_expert_critical_stall_ms"]) for profile in profiles]
    vmax = max(float(np.log1p(matrix).max()) for matrix in matrices) if matrices else 1.0
    fig, axes = plt.subplots(
        1,
        len(profiles),
        figsize=(6.4 * len(profiles), 4.8),
        squeeze=False,
        constrained_layout=True,
    )

    image = None
    for idx, (profile, matrix) in enumerate(zip(profiles, matrices)):
        ax = axes[0, idx]
        log_matrix = np.log1p(matrix)
        style = style_for_profile(profile)
        image = ax.imshow(log_matrix, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(
            f"{style['label']}\n"
            f"tput={profile['summary']['throughput_tokens_per_sec']:.1f}, "
            f"stall={profile['summary']['avg_residual_stall_ms']:.1f}ms"
        )
        ax.set_xlabel("Expert")
        ax.set_ylabel("Layer")

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("log(1 + critical stall ms)")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Lorenz and heatmap for per-expert stall profiles.")
    parser.add_argument("--profiles", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="mem0p10_pw1_deadline_e17p2")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles = [load_json(path) for path in args.profiles]
    def sort_key(item):
        style = style_for_profile(item)
        keys = list(STYLE_BY_KEY.values())
        for idx, entry in enumerate(keys):
            if entry["label"] == style["label"]:
                return idx
        return len(keys)
    profiles = sorted(profiles, key=sort_key)

    plot_lorenz(profiles, args.output_dir / f"critical_stall_lorenz_{args.tag}.png")
    plot_heatmaps(profiles, args.output_dir / f"critical_stall_heatmap_{args.tag}.png")
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
