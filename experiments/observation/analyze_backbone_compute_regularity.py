import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from backbone_moe.workload import load_state_dict


def summarize_distribution(values):
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    ordered = sorted(float(v) for v in values)

    def percentile(p):
        idx = int(round((len(ordered) - 1) * p))
        idx = max(0, min(len(ordered) - 1, idx))
        return float(ordered[idx])

    return {
        "count": int(len(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def load_resident_set(path):
    payload = json.loads(Path(path).read_text())
    entries = payload.get("resident_set", [])
    resident_set = {
        (int(item["layer"]), int(item["expert"]))
        for item in entries
    }
    return resident_set, payload


def analyze_trace(state_dict, resident_set):
    grouped = defaultdict(lambda: defaultdict(int))
    total_tokens = 0
    layer_token_steps = 0
    total_assignments = 0
    resident_assignments = 0
    token_hits = 0
    layer_token_hits = 0

    for trace_entry in state_dict.values():
        iters = trace_entry.get("iters", [])
        for iter_idx, it in enumerate(iters):
            nodes = it["nodes"]
            total_tokens += 1
            token_has_backbone = False
            num_layers = int(nodes.shape[0])
            for layer_idx in range(num_layers):
                experts = torch.nonzero(nodes[layer_idx] > 0).squeeze(-1).tolist()
                if not experts:
                    continue
                layer_token_steps += 1
                layer_has_backbone = False
                step_counts = grouped[(iter_idx, layer_idx)]
                for expert_idx in experts:
                    expert_key = (layer_idx, int(expert_idx))
                    step_counts[expert_key] += 1
                    total_assignments += 1
                    if expert_key in resident_set:
                        resident_assignments += 1
                        layer_has_backbone = True
                        token_has_backbone = True
                if layer_has_backbone:
                    layer_token_hits += 1
            if token_has_backbone:
                token_hits += 1

    total_blocks = 0
    resident_blocks = 0
    all_active_experts = []
    tail_active_experts = []
    active_reductions = []
    all_group_sizes = []
    tail_group_sizes = []
    backbone_group_sizes = []
    mean_group_size_all = []
    mean_group_size_tail = []
    mean_group_size_backbone = []
    layer_shape_patterns = defaultdict(list)
    layer_exact_patterns = defaultdict(list)

    for (iter_idx, layer_idx), counts in sorted(grouped.items()):
        total_blocks += len(counts)
        resident_counts = {key: value for key, value in counts.items() if key in resident_set}
        tail_counts = {key: value for key, value in counts.items() if key not in resident_set}
        resident_blocks += len(resident_counts)

        active_all = len(counts)
        active_tail = len(tail_counts)
        all_active_experts.append(active_all)
        tail_active_experts.append(active_tail)
        active_reductions.append((active_all - active_tail) / max(1, active_all))

        all_values = list(counts.values())
        tail_values = list(tail_counts.values())
        resident_values = list(resident_counts.values())

        all_group_sizes.extend(all_values)
        tail_group_sizes.extend(tail_values)
        backbone_group_sizes.extend(resident_values)

        mean_group_size_all.append(sum(all_values) / max(1, len(all_values)))
        mean_group_size_tail.append(sum(tail_values) / max(1, len(tail_values)) if tail_values else 0.0)
        mean_group_size_backbone.append(sum(resident_values) / max(1, len(resident_values)) if resident_values else 0.0)

        layer_exact_patterns[layer_idx].append(
            tuple(sorted((expert_key[1], int(value)) for expert_key, value in counts.items()))
        )
        layer_shape_patterns[layer_idx].append(
            tuple(sorted((int(value) for value in counts.values()), reverse=True))
        )

    def reuse_rate(patterns):
        if not patterns:
            return 0.0
        return 1.0 - (len(set(patterns)) / max(1, len(patterns)))

    per_layer_shape_reuse = {
        str(layer_idx): reuse_rate(patterns)
        for layer_idx, patterns in sorted(layer_shape_patterns.items())
    }
    per_layer_exact_reuse = {
        str(layer_idx): reuse_rate(patterns)
        for layer_idx, patterns in sorted(layer_exact_patterns.items())
    }

    weighted_shape_reuse = 0.0
    total_pattern_steps = 0
    for layer_idx, patterns in layer_shape_patterns.items():
        weighted_shape_reuse += reuse_rate(patterns) * len(patterns)
        total_pattern_steps += len(patterns)
    weighted_shape_reuse /= max(1, total_pattern_steps)

    access_coverage = resident_assignments / max(1, total_assignments)
    token_coverage = token_hits / max(1, total_tokens)
    layer_token_coverage = layer_token_hits / max(1, layer_token_steps)
    expert_block_coverage = resident_blocks / max(1, total_blocks)

    return {
        "totals": {
            "token_iterations": int(total_tokens),
            "layer_token_steps": int(layer_token_steps),
            "expert_assignments": int(total_assignments),
            "expert_blocks": int(total_blocks),
            "resident_assignments": int(resident_assignments),
            "resident_blocks": int(resident_blocks),
            "resident_set_size": int(len(resident_set)),
        },
        "coverage": {
            "backbone_access_coverage": float(access_coverage),
            "backbone_token_coverage": float(token_coverage),
            "backbone_layer_token_coverage": float(layer_token_coverage),
            "backbone_expert_block_coverage": float(expert_block_coverage),
            "backbone_flop_coverage": float(access_coverage),
            "backbone_flop_coverage_assumption": "homogeneous_expert_ffn_cost",
        },
        "active_expert_count_before_after_backbone": {
            "all": summarize_distribution(all_active_experts),
            "tail_only": summarize_distribution(tail_active_experts),
            "reduction_fraction": summarize_distribution(active_reductions),
        },
        "group_size_before_after_backbone": {
            "per_block": {
                "all": summarize_distribution(all_group_sizes),
                "tail_only": summarize_distribution(tail_group_sizes),
                "backbone_only": summarize_distribution(backbone_group_sizes),
            },
            "per_step_mean": {
                "all": summarize_distribution(mean_group_size_all),
                "tail_only": summarize_distribution(mean_group_size_tail),
                "backbone_only": summarize_distribution(mean_group_size_backbone),
            },
        },
        "assignment_shape_reuse_rate": {
            "weighted_mean": float(weighted_shape_reuse),
            "per_layer_shape": per_layer_shape_reuse,
            "per_layer_exact": per_layer_exact_reuse,
        },
    }


def build_markdown(summary):
    lines = []
    lines.append("# Backbone Compute Regularity Observation")
    lines.append("")
    lines.append(f"- state_file: `{summary['state_file']}`")
    lines.append(f"- resident_file: `{summary['resident_file']}`")
    lines.append(f"- resident_set_size: `{summary['resident_summary']['resident_set_size']}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    coverage = summary["coverage"]
    lines.append(f"- backbone_access_coverage: `{coverage['backbone_access_coverage']:.4f}`")
    lines.append(f"- backbone_token_coverage: `{coverage['backbone_token_coverage']:.4f}`")
    lines.append(f"- backbone_layer_token_coverage: `{coverage['backbone_layer_token_coverage']:.4f}`")
    lines.append(f"- backbone_expert_block_coverage: `{coverage['backbone_expert_block_coverage']:.4f}`")
    lines.append(f"- backbone_flop_coverage: `{coverage['backbone_flop_coverage']:.4f}`")
    lines.append(f"- backbone_flop_coverage_assumption: `{coverage['backbone_flop_coverage_assumption']}`")
    lines.append("")
    lines.append("## Active Expert Reduction")
    lines.append("")
    reduction = summary["active_expert_count_before_after_backbone"]
    lines.append(
        f"- mean active experts per step: `all={reduction['all']['mean']:.2f}`, "
        f"`tail={reduction['tail_only']['mean']:.2f}`"
    )
    lines.append(
        f"- reduction_fraction: `mean={reduction['reduction_fraction']['mean']:.4f}`, "
        f"`p50={reduction['reduction_fraction']['p50']:.4f}`, "
        f"`p95={reduction['reduction_fraction']['p95']:.4f}`"
    )
    lines.append("")
    lines.append("## Group Size")
    lines.append("")
    group_stats = summary["group_size_before_after_backbone"]["per_step_mean"]
    lines.append(
        f"- mean group size per step: `all={group_stats['all']['mean']:.2f}`, "
        f"`tail={group_stats['tail_only']['mean']:.2f}`, "
        f"`backbone={group_stats['backbone_only']['mean']:.2f}`"
    )
    lines.append("")
    lines.append("## Assignment Shape Reuse")
    lines.append("")
    lines.append(
        f"- weighted_mean assignment_shape_reuse_rate: "
        f"`{summary['assignment_shape_reuse_rate']['weighted_mean']:.4f}`"
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Analyze backbone compute regularity from routed traces.")
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--resident-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=str, default="backbone_compute_regularity")
    args = parser.parse_args()

    state_dict = load_state_dict(args.state_file)
    resident_set, resident_payload = load_resident_set(args.resident_file)
    summary = analyze_trace(state_dict, resident_set)
    summary["analysis"] = "backbone_compute_regularity"
    summary["state_file"] = str(args.state_file)
    summary["resident_file"] = str(args.resident_file)
    summary["resident_summary"] = {
        "resident_set_size": int(len(resident_set)),
        "selection_budget_bytes": resident_payload.get("selection_budget_bytes"),
        "resident_capacity": resident_payload.get("resident_capacity"),
        "resident_policy": resident_payload.get("resident_policy"),
        "selection_method": resident_payload.get("selection_method"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_prefix}.json"
    md_path = args.output_dir / f"{args.output_prefix}.md"
    json_path.write_text(json.dumps(summary, indent=2))
    md_path.write_text(build_markdown(summary))
    print(f"Saved compute regularity JSON to {json_path}")
    print(f"Saved compute regularity Markdown to {md_path}")


if __name__ == "__main__":
    main()
