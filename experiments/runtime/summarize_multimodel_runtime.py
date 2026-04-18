import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "runtime_formal"


MODEL_MANIFEST = {
    "qwen": {
        "title": "Qwen1.5-MoE-A2.7B-Chat",
        "role": "strong_positive",
        "runtime": {
            "0.07": {
                "A": RESULTS_DIR / "qwen" / "qwen_A_mem0p07.json",
                "C": RESULTS_DIR / "qwen" / "qwen_C_mem0p07.json",
            },
            "0.10": {
                "A": RESULTS_DIR / "qwen" / "qwen_A_mem0p10.json",
                "C": RESULTS_DIR / "qwen" / "qwen_C_mem0p10.json",
            },
        },
        "applicability": RESULTS_DIR / "applicability" / "qwen_applicability.json",
    },
    "olmoe": {
        "title": "OLMoE-1B-7B-0924",
        "role": "strong_positive",
        "runtime": {
            "0.07": {
                "A": RESULTS_DIR / "olmoe" / "olmoe_A_mem0p07.json",
                "C": RESULTS_DIR / "olmoe" / "olmoe_C_mem0p07.json",
            },
            "0.10": {
                "A": RESULTS_DIR / "olmoe" / "olmoe_A_mem0p10.json",
                "C": RESULTS_DIR / "olmoe" / "olmoe_C_mem0p10.json",
            },
        },
        "applicability": None,
    },
    "deepseek_v2_lite": {
        "title": "DeepSeek-V2-Lite",
        "role": "weak_positive_boundary",
        "runtime": {
            "0.07": {
                "A": RESULTS_DIR / "deepseek" / "deepseek_v2_A_mem0p07.json",
                "C": RESULTS_DIR / "deepseek" / "deepseek_v2_C_mem0p07.json",
            },
            "0.10": {
                "A": RESULTS_DIR / "deepseek" / "deepseek_v2_A_mem0p10.json",
                "C": RESULTS_DIR / "deepseek" / "deepseek_v2_C_mem0p10.json",
            },
        },
        "applicability": RESULTS_DIR / "applicability" / "deepseek_v2_lite_applicability.json",
    },
    "mixtral": {
        "title": "Mixtral-8x7B",
        "role": "applicability_boundary",
        "runtime": {},
        "applicability": None,
        "boundary_assets": {
            "generalization": RESULTS_DIR / "mixtral" / "mixtral_backbone_generalization.json",
            "adaptive": RESULTS_DIR / "mixtral" / "mixtral_adaptive_summary.json",
            "tiny_A": RESULTS_DIR / "mixtral" / "mixtral_tiny_A.json",
            "tiny_C": RESULTS_DIR / "mixtral" / "mixtral_tiny_C.json",
        },
    },
}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def summarize_runtime_pair(a_path: Path, c_path: Path):
    a = load_json(a_path)
    c = load_json(c_path)
    a_gen = float(a["generated_tokens_per_sec"])
    c_gen = float(c["generated_tokens_per_sec"])
    gain = ((c_gen / a_gen) - 1.0) * 100.0 if a_gen > 0 else 0.0
    return {
        "A": {
            "generated_tokens_per_sec": a_gen,
            "end_to_end_tokens_per_sec": float(a["end_to_end_tokens_per_sec"]),
            "resident_count": int(a.get("resident_count", 0)),
            "path": str(a_path.relative_to(ROOT)),
        },
        "C": {
            "generated_tokens_per_sec": c_gen,
            "end_to_end_tokens_per_sec": float(c["end_to_end_tokens_per_sec"]),
            "resident_count": int(c.get("resident_count", 0)),
            "path": str(c_path.relative_to(ROOT)),
        },
        "gain_percent": gain,
    }


def summarize_applicability(path: Path | None):
    if path is None:
        return None
    payload = load_json(path)
    rows = {}
    for row in payload["memory_ratios"]:
        mem = f"{float(row['device_memory_ratio']):.2f}"
        applicability = row["applicability"]
        rows[mem] = {
            "top20_access_coverage": float(row["top20_access_coverage"]),
            "top20_stall_coverage": float(row["top20_stall_coverage"]),
            "knee_ratio": float(applicability["knee_ratio"]),
            "frontier_selected_ratio": float(applicability["frontier_selected_ratio"]),
            "frontier_horizon": int(applicability["frontier_horizon"]),
            "slack_utilization": float(applicability["frontier_selected_slack_utilization"]),
            "path": str(path.relative_to(ROOT)),
        }
    return rows


def summarize_mixtral_boundary(assets: dict | None):
    if not assets:
        return None
    generalization = load_json(assets["generalization"])
    adaptive = load_json(assets["adaptive"])
    tiny_a = load_json(assets["tiny_A"])
    tiny_c = load_json(assets["tiny_C"])
    generalization_rows = {}
    for row in generalization["results"]:
        mem = f"{float(row['device_memory_ratio']):.2f}"
        generalization_rows[mem] = {
            "jaccard": float(row["train_test_jaccard"]),
            "retained_gain_fraction": float(row["retained_gain_fraction"]),
            "single_tput": float(row["best_single"]["throughput_tokens_per_sec"]),
            "native_tput": float(row["best_test_native"]["throughput_tokens_per_sec"]),
        }
    adaptive_rows = {}
    for row in adaptive["results"]:
        mem = f"{float(row['device_memory_ratio']):.2f}"
        adaptive_rows[mem] = {
            "selected_resident_ratio": float(row["selected_resident_ratio"]),
            "selected_resident_capacity": int(row["selected_resident_capacity"]),
            "profile_tput": float(row["profile_throughput_tokens_per_sec"]),
            "cache_capacity": int(row["cache_capacity"]),
        }
    return {
        "generalization": generalization_rows,
        "adaptive": adaptive_rows,
        "tiny_runtime": {
            "A_resident_count": int(tiny_a.get("resident_count", 0)),
            "C_resident_count": int(tiny_c.get("resident_count", 0)),
            "A_generated_tokens_per_sec": (
                float(tiny_a["generated_tokens_per_sec"])
                if tiny_a.get("generated_tokens_per_sec") is not None
                else None
            ),
            "C_generated_tokens_per_sec": (
                float(tiny_c["generated_tokens_per_sec"])
                if tiny_c.get("generated_tokens_per_sec") is not None
                else None
            ),
            "gain_percent": (
                ((float(tiny_c["generated_tokens_per_sec"]) / float(tiny_a["generated_tokens_per_sec"])) - 1.0) * 100.0
                if tiny_a.get("generated_tokens_per_sec") not in (None, 0)
                and tiny_c.get("generated_tokens_per_sec") is not None
                else None
            ),
            "A_path": str(assets["tiny_A"].relative_to(ROOT)),
            "C_path": str(assets["tiny_C"].relative_to(ROOT)),
        },
    }


def build_summary():
    summary = {"models": []}
    for model_id, spec in MODEL_MANIFEST.items():
        runtime = {
            mem: summarize_runtime_pair(pair["A"], pair["C"])
            for mem, pair in spec["runtime"].items()
        }
        summary["models"].append(
            {
                "model_id": model_id,
                "title": spec["title"],
                "role": spec["role"],
                "runtime": runtime,
                "applicability": summarize_applicability(spec["applicability"]),
                "boundary": summarize_mixtral_boundary(spec.get("boundary_assets")),
            }
        )
    return summary


def render_markdown(summary):
    lines = []
    lines.append("# Multi-Model Runtime Summary")
    lines.append("")
    lines.append("Current paper-facing split:")
    lines.append("- `Qwen` / `OLMoE`: strong positive cases")
    lines.append("- `DeepSeek-V2-Lite`: weak positive / boundary case")
    lines.append("- `Mixtral`: applicability / boundary case; tiny packed-runtime probe is positive, but full-model runtime is still missing")
    lines.append("")
    lines.append("## Runtime Table")
    lines.append("")
    lines.append("| Model | Role | mem | A gen tok/s | C gen tok/s | gain | C resident |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for model in summary["models"]:
        if not model["runtime"]:
            continue
        for mem, runtime in sorted(model["runtime"].items(), key=lambda item: float(item[0])):
            lines.append(
                f"| {model['title']} | {model['role']} | {float(mem):.2f} | "
                f"{runtime['A']['generated_tokens_per_sec']:.4f} | "
                f"{runtime['C']['generated_tokens_per_sec']:.4f} | "
                f"{runtime['gain_percent']:+.1f}% | "
                f"{runtime['C']['resident_count']} |"
            )
    lines.append("")
    lines.append("## Applicability Table")
    lines.append("")
    lines.append("| Model | mem | top20 access cov. | top20 stall cov. | knee ratio | frontier ratio | horizon | slack util. |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model in summary["models"]:
        applicability = model["applicability"]
        if applicability is None:
            continue
        for mem, row in sorted(applicability.items(), key=lambda item: float(item[0])):
            lines.append(
                f"| {model['title']} | {float(mem):.2f} | "
                f"{row['top20_access_coverage']:.3f} | "
                f"{row['top20_stall_coverage']:.3f} | "
                f"{row['knee_ratio']:.3f} | "
                f"{row['frontier_selected_ratio']:.3f} | "
                f"{row['frontier_horizon']} | "
                f"{row['slack_utilization']:.3f} |"
            )
    lines.append("")
    lines.append("## Mixtral Boundary Summary")
    lines.append("")
    lines.append("| mem | retained | Jaccard | adaptive ratio | adaptive k | profile tput |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    mixtral = next((m for m in summary["models"] if m["model_id"] == "mixtral"), None)
    if mixtral and mixtral["boundary"]:
        gen = mixtral["boundary"]["generalization"]
        ada = mixtral["boundary"]["adaptive"]
        for mem in sorted(gen.keys(), key=float):
            lines.append(
                f"| {float(mem):.2f} | "
                f"{gen[mem]['retained_gain_fraction']:.3f} | "
                f"{gen[mem]['jaccard']:.3f} | "
                f"{ada[mem]['selected_resident_ratio']:.3f} | "
                f"{ada[mem]['selected_resident_capacity']} | "
                f"{ada[mem]['profile_tput']:.1f} |"
            )
        lines.append("")
        lines.append("Tiny packed-runtime probe:")
        lines.append(
            f"- `A`: gen tok/s={mixtral['boundary']['tiny_runtime']['A_generated_tokens_per_sec']:.4f}, resident_count={mixtral['boundary']['tiny_runtime']['A_resident_count']}"
        )
        lines.append(
            f"- `C`: gen tok/s={mixtral['boundary']['tiny_runtime']['C_generated_tokens_per_sec']:.4f}, resident_count={mixtral['boundary']['tiny_runtime']['C_resident_count']}"
        )
        lines.append(
            f"- tiny probe gain: {mixtral['boundary']['tiny_runtime']['gain_percent']:+.1f}%"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `Qwen` remains the clearest compact-backbone positive case.")
    lines.append("- `OLMoE` is also strongly positive, but mainly because resident pinning dominates under the current small-expert regime.")
    lines.append("- `DeepSeek-V2-Lite` is not a negative case: `C > A` on real hardware, but the gains stay modest, so it should be framed as a weak positive / boundary case.")
    lines.append("- `Mixtral` has non-trivial backbone structure in simulation (`retained≈0.96-0.97` at `mem=0.07/0.10`) and a positive tiny packed-runtime probe, but without a full-model runtime asset it should still stay as an applicability / boundary model rather than a formal positive runtime case.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    summary = build_summary()
    out_json = RESULTS_DIR / "multimodel_runtime_summary.json"
    out_md = RESULTS_DIR / "multimodel_runtime_summary.md"
    out_json.write_text(json.dumps(summary, indent=2))
    out_md.write_text(render_markdown(summary))
    print(f"Saved JSON summary to {out_json}")
    print(f"Saved markdown summary to {out_md}")


if __name__ == "__main__":
    main()
