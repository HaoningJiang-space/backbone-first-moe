import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "results" / "runtime_formal" / "multimodel_runtime_summary.json"
OUT_JSON = ROOT / "results" / "runtime_formal" / "paper_runtime_table.json"
OUT_MD = ROOT / "results" / "runtime_formal" / "paper_runtime_table.md"


PAPER_SELECTION = {
    "qwen": {"view": "runtime", "mems": ["0.07", "0.10"]},
    "olmoe": {"view": "fair_runtime", "mems": ["0.012", "0.014", "0.016"]},
    "deepseek_v2_lite": {"view": "runtime", "mems": ["0.07", "0.10"]},
}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def build_table(summary: dict):
    model_index = {model["model_id"]: model for model in summary["models"]}
    rows = []
    notes = []
    for model_id, spec in PAPER_SELECTION.items():
        model = model_index[model_id]
        selected_view = spec["view"]
        runtime = model[selected_view]
        if selected_view == "fair_runtime":
            notes.append(
                f"{model['title']}: use coverage-matched fair runtime points for cross-model comparison because fixed mem-ratio is near-full-fit."
            )
        for mem in spec["mems"]:
            row = runtime[mem]
            rows.append(
                {
                    "model_id": model_id,
                    "title": model["title"],
                    "role": model["role"],
                    "view": selected_view,
                    "mem": mem,
                    "A_gen_tok_s": row["A"]["generated_tokens_per_sec"],
                    "C_gen_tok_s": row["C"]["generated_tokens_per_sec"],
                    "gain_percent": row["gain_percent"],
                    "resident_count": row["C"]["resident_count"],
                    "A_path": row["A"]["path"],
                    "C_path": row["C"]["path"],
                }
            )
    return {"rows": rows, "notes": notes}


def render_markdown(table: dict):
    lines = []
    lines.append("# Paper Runtime Table")
    lines.append("")
    lines.append("This table uses the paper-facing comparison view:")
    lines.append("- `Qwen`: fixed-mem runtime points")
    lines.append("- `DeepSeek-V2-Lite`: fixed-mem runtime points")
    lines.append("- `OLMoE`: coverage-matched fair runtime points")
    lines.append("")
    lines.append("| Model | Comparison View | mem | A gen tok/s | C gen tok/s | gain | C resident |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in table["rows"]:
        view = "fair" if row["view"] == "fair_runtime" else "fixed"
        lines.append(
            f"| {row['title']} | {view} | {float(row['mem']):.3f} | "
            f"{row['A_gen_tok_s']:.4f} | {row['C_gen_tok_s']:.4f} | "
            f"{row['gain_percent']:+.1f}% | {row['resident_count']} |"
        )
    if table["notes"]:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        for note in table["notes"]:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main():
    summary = load_json(SUMMARY_PATH)
    table = build_table(summary)
    OUT_JSON.write_text(json.dumps(table, indent=2) + "\n")
    OUT_MD.write_text(render_markdown(table))
    print(f"Saved JSON table to {OUT_JSON}")
    print(f"Saved markdown table to {OUT_MD}")


if __name__ == "__main__":
    main()
