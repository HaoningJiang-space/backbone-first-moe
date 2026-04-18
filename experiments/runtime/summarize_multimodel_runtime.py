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
    lines.append("- `Mixtral`: applicability case until a formal packed-runtime probe is completed")
    lines.append("")
    lines.append("## Runtime Table")
    lines.append("")
    lines.append("| Model | Role | mem | A gen tok/s | C gen tok/s | gain | C resident |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for model in summary["models"]:
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
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `Qwen` remains the clearest compact-backbone positive case.")
    lines.append("- `OLMoE` is also strongly positive, but mainly because resident pinning dominates under the current small-expert regime.")
    lines.append("- `DeepSeek-V2-Lite` is not a negative case: `C > A` on real hardware, but the gains stay modest, so it should be framed as a weak positive / boundary case.")
    lines.append("- `Mixtral` should stay outside the formal positive table until a full packed-runtime probe shows a clear resident-backbone gain.")
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
