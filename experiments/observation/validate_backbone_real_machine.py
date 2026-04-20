import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.observation.analyze_backbone_compute_regularity import (  # noqa: E402
    analyze_trace,
    load_resident_set,
)


def run_cmd(cmd, cwd):
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_prompt_items(prompt_file):
    with open(prompt_file, "r") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"Prompt file must contain a list: {prompt_file}")
    return items


def write_prompt_items(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2))


def summarize(values):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return {"count": 0, "min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
    }


def resident_jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / max(1, len(union))


def derive_state_path(model_path, dataset_name, sample_size):
    model_name = Path(model_path.rstrip("/")).name
    return REPO_ROOT / "states" / f"{model_name}~{dataset_name}~{sample_size}.pkl"


def derive_resident_path(output_dir, output_prefix, mem_ratio):
    mem_tag = f"{mem_ratio:.6f}".split(".")
    whole, frac = mem_tag[0], mem_tag[1].rstrip("0")
    if len(frac) < 2:
        frac = frac.ljust(2, "0")
    return output_dir / f"{output_prefix}_mem{whole}p{frac}.json"


def build_shard(args, shard_idx, shard_items):
    shard_name = f"{args.dataset_prefix}_shard{shard_idx}"
    shard_dir = args.output_dir / f"shard_{shard_idx}"
    prompts_path = shard_dir / "prompts.json"
    states_dir = shard_dir / "states"
    resident_dir = shard_dir / "resident"

    write_prompt_items(prompts_path, shard_items)

    generated_state = derive_state_path(args.model_path, shard_name, len(shard_items))
    local_state = states_dir / generated_state.name
    resident_path = derive_resident_path(resident_dir, "resident", args.memory_ratio)

    if not (
        args.reuse_existing
        and local_state.exists()
        and resident_path.exists()
    ):
        prepare_cmd = [
            sys.executable,
            "demo/prepare_custom_data.py",
            "--prompt-file",
            str(prompts_path),
            "--dataset-name",
            shard_name,
            "--sample-size",
            str(len(shard_items)),
            "--batch-size",
            str(args.batch_size),
            "--model-path",
            args.model_path,
        ]
        run_cmd(prepare_cmd, REPO_ROOT)

        if not generated_state.exists():
            raise FileNotFoundError(f"Expected generated state: {generated_state}")

        states_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_state, local_state)

        select_cmd = [
            sys.executable,
            "-m",
            "experiments.simulation.select_adaptive_resident_set",
            "--state-file",
            str(local_state),
            "--model-path",
            args.model_path,
            "--output-dir",
            str(resident_dir),
            "--output-prefix",
            "resident",
            "--memory-ratios",
            str(args.memory_ratio),
            "--selection-method",
            "frontier_prefix",
            "--profile-fraction",
            str(args.profile_fraction),
            "--prefetch-windows",
            "0",
        ]
        if args.sparse_budget_bytes is not None:
            select_cmd.extend(["--sparse-budget-bytes", str(args.sparse_budget_bytes)])
        run_cmd(select_cmd, REPO_ROOT)

    if not resident_path.exists():
        raise FileNotFoundError(f"Expected resident file: {resident_path}")

    resident_set, resident_payload = load_resident_set(resident_path)

    from backbone_moe.workload import load_state_dict  # local import to keep startup light

    state_dict = load_state_dict(local_state)
    native_stats = analyze_trace(state_dict, resident_set)

    return {
        "shard_index": shard_idx,
        "shard_name": shard_name,
        "prompt_file": str(prompts_path),
        "state_file": str(local_state),
        "resident_file": str(resident_path),
        "resident_payload": resident_payload,
        "resident_set": resident_set,
        "native_stats": native_stats,
    }


def build_markdown(summary):
    lines = []
    lines.append("# Real-Machine Backbone Validation")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        f"- model: `{summary['model_path']}`"
    )
    lines.append(
        f"- prompt_file: `{summary['prompt_file']}`"
    )
    lines.append(
        f"- shards: `{summary['num_shards']}` x `{summary['shard_size']}` prompts"
    )
    lines.append(
        f"- memory_ratio: `{summary['memory_ratio']}`"
    )
    if summary["sparse_budget_bytes"] is not None:
        lines.append(
            f"- sparse_budget_bytes: `{summary['sparse_budget_bytes']}`"
        )
    lines.append("")
    lines.append("## Native Per-Shard")
    lines.append("")
    for shard in summary["shards"]:
        lines.append(
            f"- shard {shard['shard_index']}: resident_count={shard['resident_count']}, "
            f"assignment_fraction_per_token_mean={shard['assignment_fraction_per_token_mean']:.4f}, "
            f"backbone_any_hit_token_coverage={shard['backbone_any_hit_token_coverage']:.4f}, "
            f"active_reduction_mean={shard['active_reduction_mean']:.4f}"
        )
    lines.append("")
    lines.append("## Cross-Shard Stability")
    lines.append("")
    agg = summary["aggregate"]
    lines.append(
        f"- resident_jaccard: mean={agg['resident_jaccard']['mean']:.4f}, "
        f"min={agg['resident_jaccard']['min']:.4f}, max={agg['resident_jaccard']['max']:.4f}"
    )
    lines.append(
        f"- transfer/native assignment fraction retained: mean={agg['retained_assignment_fraction']['mean']:.4f}, "
        f"min={agg['retained_assignment_fraction']['min']:.4f}, max={agg['retained_assignment_fraction']['max']:.4f}"
    )
    lines.append(
        f"- transfer/native any-hit retained: mean={agg['retained_any_hit_fraction']['mean']:.4f}, "
        f"min={agg['retained_any_hit_fraction']['min']:.4f}, max={agg['retained_any_hit_fraction']['max']:.4f}"
    )
    lines.append(
        f"- transfer/native active-reduction retained: mean={agg['retained_active_reduction_fraction']['mean']:.4f}, "
        f"min={agg['retained_active_reduction_fraction']['min']:.4f}, max={agg['retained_active_reduction_fraction']['max']:.4f}"
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        "- `backbone exists` if fresh real-machine shards repeatedly produce non-empty resident sets with substantial native assignment coverage."
    )
    lines.append(
        "- `backbone is stable` if train-derived resident sets retain a large fraction of held-out assignment coverage and overlap non-trivially across shards."
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Validate backbone existence and stability from fresh real-machine traces.")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-prefix", type=str, default="qwen_rm_backbone")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-ratio", type=float, default=0.10)
    parser.add_argument("--profile-fraction", type=float, default=0.2)
    parser.add_argument("--sparse-budget-bytes", type=int, default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    items = load_prompt_items(args.prompt_file)
    need = args.num_shards * args.shard_size
    if len(items) < need:
        raise ValueError(f"Need at least {need} prompts, got {len(items)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    shards = []
    for shard_idx in range(args.num_shards):
        start = shard_idx * args.shard_size
        end = start + args.shard_size
        shard_items = items[start:end]
        shards.append(build_shard(args, shard_idx, shard_items))

    from backbone_moe.workload import load_state_dict  # local import to keep startup light

    pair_rows = []
    resident_jaccards = []
    retained_assignment_fractions = []
    retained_any_hit_fractions = []
    retained_active_reduction_fractions = []

    state_cache = {
        shard["shard_index"]: load_state_dict(shard["state_file"])
        for shard in shards
    }

    for train in shards:
        for test in shards:
            if train["shard_index"] == test["shard_index"]:
                continue
            transfer_stats = analyze_trace(state_cache[test["shard_index"]], train["resident_set"])
            native_stats = test["native_stats"]
            j = resident_jaccard(train["resident_set"], test["resident_set"])
            retained_assignment = (
                transfer_stats["assignment_fraction"]["per_token"]["mean"]
                / max(1e-12, native_stats["assignment_fraction"]["per_token"]["mean"])
            )
            retained_any_hit = (
                transfer_stats["coverage"]["backbone_any_hit_token_coverage"]
                / max(1e-12, native_stats["coverage"]["backbone_any_hit_token_coverage"])
            )
            native_active = native_stats["active_expert_count_before_after_backbone"]["reduction_fraction"]["mean"]
            retained_active = (
                transfer_stats["active_expert_count_before_after_backbone"]["reduction_fraction"]["mean"]
                / max(1e-12, native_active)
                if native_active > 0
                else 0.0
            )
            row = {
                "train_shard": train["shard_index"],
                "test_shard": test["shard_index"],
                "resident_jaccard": j,
                "transfer_assignment_fraction_per_token_mean": transfer_stats["assignment_fraction"]["per_token"]["mean"],
                "native_assignment_fraction_per_token_mean": native_stats["assignment_fraction"]["per_token"]["mean"],
                "retained_assignment_fraction": retained_assignment,
                "transfer_any_hit_token_coverage": transfer_stats["coverage"]["backbone_any_hit_token_coverage"],
                "native_any_hit_token_coverage": native_stats["coverage"]["backbone_any_hit_token_coverage"],
                "retained_any_hit_fraction": retained_any_hit,
                "transfer_active_reduction_mean": transfer_stats["active_expert_count_before_after_backbone"]["reduction_fraction"]["mean"],
                "native_active_reduction_mean": native_active,
                "retained_active_reduction_fraction": retained_active,
            }
            pair_rows.append(row)
            resident_jaccards.append(j)
            retained_assignment_fractions.append(retained_assignment)
            retained_any_hit_fractions.append(retained_any_hit)
            retained_active_reduction_fractions.append(retained_active)

    summary = {
        "analysis": "real_machine_backbone_validation",
        "model_path": args.model_path,
        "prompt_file": str(args.prompt_file),
        "num_shards": args.num_shards,
        "shard_size": args.shard_size,
        "batch_size": args.batch_size,
        "memory_ratio": args.memory_ratio,
        "profile_fraction": args.profile_fraction,
        "sparse_budget_bytes": args.sparse_budget_bytes,
        "shards": [],
        "pairwise_transfer": pair_rows,
        "aggregate": {
            "resident_jaccard": summarize(resident_jaccards),
            "retained_assignment_fraction": summarize(retained_assignment_fractions),
            "retained_any_hit_fraction": summarize(retained_any_hit_fractions),
            "retained_active_reduction_fraction": summarize(retained_active_reduction_fractions),
        },
    }

    for shard in shards:
        native = shard["native_stats"]
        summary["shards"].append({
            "shard_index": shard["shard_index"],
            "shard_name": shard["shard_name"],
            "prompt_file": shard["prompt_file"],
            "state_file": shard["state_file"],
            "resident_file": shard["resident_file"],
            "resident_count": len(shard["resident_set"]),
            "assignment_fraction_per_token_mean": native["assignment_fraction"]["per_token"]["mean"],
            "backbone_any_hit_token_coverage": native["coverage"]["backbone_any_hit_token_coverage"],
            "active_reduction_mean": native["active_expert_count_before_after_backbone"]["reduction_fraction"]["mean"],
            "backbone_group_mean": native["group_size_before_after_backbone"]["per_step_mean"]["backbone_only"]["mean"],
            "tail_group_mean": native["group_size_before_after_backbone"]["per_step_mean"]["tail_only"]["mean"],
        })

    json_path = args.output_dir / "real_machine_backbone_validation.json"
    md_path = args.output_dir / "real_machine_backbone_validation.md"
    json_path.write_text(json.dumps(summary, indent=2))
    md_path.write_text(build_markdown(summary))
    print(f"Saved JSON to {json_path}")
    print(f"Saved Markdown to {md_path}")


if __name__ == "__main__":
    main()
