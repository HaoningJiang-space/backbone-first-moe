import argparse
import json
from collections import defaultdict
from copy import deepcopy
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


def subset_state_dict(state_dict, keep_keys):
    return {key: deepcopy(state_dict[key]) for key in keep_keys}


def analyze_trace(state_dict, resident_set):
    grouped = defaultdict(lambda: defaultdict(int))
    total_tokens = 0
    layer_token_steps = 0
    total_assignments = 0
    resident_assignments = 0
    token_hits = 0
    layer_token_hits = 0
    token_assignment_fractions = []
    layer_token_assignment_fractions = []
    per_layer_assignments = defaultdict(int)
    per_layer_resident_assignments = defaultdict(int)

    for trace_entry in state_dict.values():
        iters = trace_entry.get("iters", [])
        for iter_idx, it in enumerate(iters):
            nodes = it["nodes"]
            total_tokens += 1
            token_has_backbone = False
            token_assignments = 0
            token_resident_assignments = 0
            num_layers = int(nodes.shape[0])
            for layer_idx in range(num_layers):
                experts = torch.nonzero(nodes[layer_idx] > 0).squeeze(-1).tolist()
                if not experts:
                    continue
                layer_token_steps += 1
                layer_has_backbone = False
                layer_assignments = 0
                layer_resident_assignments = 0
                step_counts = grouped[(iter_idx, layer_idx)]
                for expert_idx in experts:
                    expert_key = (layer_idx, int(expert_idx))
                    step_counts[expert_key] += 1
                    total_assignments += 1
                    token_assignments += 1
                    layer_assignments += 1
                    per_layer_assignments[layer_idx] += 1
                    if expert_key in resident_set:
                        resident_assignments += 1
                        token_resident_assignments += 1
                        layer_resident_assignments += 1
                        per_layer_resident_assignments[layer_idx] += 1
                        layer_has_backbone = True
                        token_has_backbone = True
                if layer_has_backbone:
                    layer_token_hits += 1
                layer_token_assignment_fractions.append(
                    layer_resident_assignments / max(1, layer_assignments)
                )
            if token_has_backbone:
                token_hits += 1
            token_assignment_fractions.append(
                token_resident_assignments / max(1, token_assignments)
            )

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
    per_layer_active_all = defaultdict(list)
    per_layer_active_tail = defaultdict(list)
    per_layer_active_reduction = defaultdict(list)
    per_layer_step_mean_group_all = defaultdict(list)
    per_layer_step_mean_group_tail = defaultdict(list)
    per_layer_step_mean_group_backbone = defaultdict(list)

    for (iter_idx, layer_idx), counts in sorted(grouped.items()):
        total_blocks += len(counts)
        resident_counts = {key: value for key, value in counts.items() if key in resident_set}
        tail_counts = {key: value for key, value in counts.items() if key not in resident_set}
        resident_blocks += len(resident_counts)

        active_all = len(counts)
        active_tail = len(tail_counts)
        all_active_experts.append(active_all)
        tail_active_experts.append(active_tail)
        reduction = (active_all - active_tail) / max(1, active_all)
        active_reductions.append(reduction)
        per_layer_active_all[layer_idx].append(active_all)
        per_layer_active_tail[layer_idx].append(active_tail)
        per_layer_active_reduction[layer_idx].append(reduction)

        all_values = list(counts.values())
        tail_values = list(tail_counts.values())
        resident_values = list(resident_counts.values())

        all_group_sizes.extend(all_values)
        tail_group_sizes.extend(tail_values)
        backbone_group_sizes.extend(resident_values)

        mean_group_size_all.append(sum(all_values) / max(1, len(all_values)))
        mean_group_size_tail.append(sum(tail_values) / max(1, len(tail_values)) if tail_values else 0.0)
        mean_group_size_backbone.append(sum(resident_values) / max(1, len(resident_values)) if resident_values else 0.0)
        per_layer_step_mean_group_all[layer_idx].append(mean_group_size_all[-1])
        per_layer_step_mean_group_tail[layer_idx].append(mean_group_size_tail[-1])
        per_layer_step_mean_group_backbone[layer_idx].append(mean_group_size_backbone[-1])

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
    per_layer_assignment_coverage = {
        str(layer_idx): (
            per_layer_resident_assignments[layer_idx] / max(1, per_layer_assignments[layer_idx])
        )
        for layer_idx in sorted(per_layer_assignments.keys())
    }
    per_layer_summary = {}
    all_layers = sorted(
        set(per_layer_assignments.keys())
        | set(per_layer_active_all.keys())
        | set(layer_shape_patterns.keys())
    )
    for layer_idx in all_layers:
        per_layer_summary[str(layer_idx)] = {
            "assignment_coverage": float(
                per_layer_resident_assignments[layer_idx] / max(1, per_layer_assignments[layer_idx])
            ),
            "active_experts_all": summarize_distribution(per_layer_active_all[layer_idx]),
            "active_experts_tail_only": summarize_distribution(per_layer_active_tail[layer_idx]),
            "active_reduction_fraction": summarize_distribution(per_layer_active_reduction[layer_idx]),
            "group_size_per_step_mean_all": summarize_distribution(per_layer_step_mean_group_all[layer_idx]),
            "group_size_per_step_mean_tail_only": summarize_distribution(per_layer_step_mean_group_tail[layer_idx]),
            "group_size_per_step_mean_backbone_only": summarize_distribution(per_layer_step_mean_group_backbone[layer_idx]),
            "shape_reuse_rate": float(reuse_rate(layer_shape_patterns[layer_idx])),
            "exact_reuse_rate": float(reuse_rate(layer_exact_patterns[layer_idx])),
        }

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
            "backbone_any_hit_token_coverage": float(token_coverage),
            "backbone_any_hit_layer_token_coverage": float(layer_token_coverage),
            "backbone_expert_block_coverage": float(expert_block_coverage),
            "backbone_flop_coverage": float(access_coverage),
            "backbone_flop_coverage_assumption": "homogeneous_expert_ffn_cost",
            "per_layer_assignment_coverage": per_layer_assignment_coverage,
        },
        "assignment_fraction": {
            "per_token": summarize_distribution(token_assignment_fractions),
            "per_layer_token": summarize_distribution(layer_token_assignment_fractions),
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
        "per_layer": per_layer_summary,
    }


def analyze_chunks(state_dict, resident_set, chunk_size):
    seq_keys = list(state_dict.keys())
    if chunk_size <= 0 or len(seq_keys) <= chunk_size:
        return {}

    chunk_rows = []
    for start in range(0, len(seq_keys), chunk_size):
        keys = seq_keys[start : start + chunk_size]
        if not keys:
            continue
        subset = subset_state_dict(state_dict, keys)
        result = analyze_trace(subset, resident_set)
        chunk_rows.append(
            {
                "chunk_index": int(len(chunk_rows)),
                "num_sequences": int(len(keys)),
                "sequence_keys": [str(key) for key in keys],
                "backbone_access_coverage": float(result["coverage"]["backbone_access_coverage"]),
                "assignment_fraction_per_token_mean": float(result["assignment_fraction"]["per_token"]["mean"]),
                "active_reduction_mean": float(
                    result["active_expert_count_before_after_backbone"]["reduction_fraction"]["mean"]
                ),
                "backbone_group_mean": float(
                    result["group_size_before_after_backbone"]["per_step_mean"]["backbone_only"]["mean"]
                ),
                "tail_group_mean": float(
                    result["group_size_before_after_backbone"]["per_step_mean"]["tail_only"]["mean"]
                ),
                "shape_reuse_weighted_mean": float(result["assignment_shape_reuse_rate"]["weighted_mean"]),
            }
        )

    return {
        "chunk_size": int(chunk_size),
        "num_chunks": int(len(chunk_rows)),
        "rows": chunk_rows,
        "summary": {
            "backbone_access_coverage": summarize_distribution(
                [row["backbone_access_coverage"] for row in chunk_rows]
            ),
            "assignment_fraction_per_token_mean": summarize_distribution(
                [row["assignment_fraction_per_token_mean"] for row in chunk_rows]
            ),
            "active_reduction_mean": summarize_distribution(
                [row["active_reduction_mean"] for row in chunk_rows]
            ),
            "backbone_group_mean": summarize_distribution(
                [row["backbone_group_mean"] for row in chunk_rows]
            ),
            "tail_group_mean": summarize_distribution(
                [row["tail_group_mean"] for row in chunk_rows]
            ),
            "shape_reuse_weighted_mean": summarize_distribution(
                [row["shape_reuse_weighted_mean"] for row in chunk_rows]
            ),
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
    lines.append(f"- backbone_any_hit_token_coverage: `{coverage['backbone_any_hit_token_coverage']:.4f}`")
    lines.append(f"- backbone_any_hit_layer_token_coverage: `{coverage['backbone_any_hit_layer_token_coverage']:.4f}`")
    lines.append(f"- backbone_expert_block_coverage: `{coverage['backbone_expert_block_coverage']:.4f}`")
    lines.append(f"- backbone_flop_coverage: `{coverage['backbone_flop_coverage']:.4f}`")
    lines.append(f"- backbone_flop_coverage_assumption: `{coverage['backbone_flop_coverage_assumption']}`")
    lines.append("")
    lines.append("## Assignment Fraction")
    lines.append("")
    assignment_fraction = summary["assignment_fraction"]
    lines.append(
        f"- backbone assignment fraction per token: "
        f"`mean={assignment_fraction['per_token']['mean']:.4f}`, "
        f"`p50={assignment_fraction['per_token']['p50']:.4f}`, "
        f"`p95={assignment_fraction['per_token']['p95']:.4f}`"
    )
    lines.append(
        f"- backbone assignment fraction per layer-token: "
        f"`mean={assignment_fraction['per_layer_token']['mean']:.4f}`, "
        f"`p50={assignment_fraction['per_layer_token']['p50']:.4f}`, "
        f"`p95={assignment_fraction['per_layer_token']['p95']:.4f}`"
    )
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
    if summary.get("chunk_stability"):
        chunk = summary["chunk_stability"]
        lines.append("")
        lines.append("## Chunk Stability")
        lines.append("")
        lines.append(
            f"- chunk_size: `{chunk['chunk_size']}`"
        )
        lines.append(
            f"- assignment_fraction_per_token_mean across chunks: "
            f"`mean={chunk['summary']['assignment_fraction_per_token_mean']['mean']:.4f}`, "
            f"`min={chunk['summary']['assignment_fraction_per_token_mean']['min']:.4f}`, "
            f"`p95={chunk['summary']['assignment_fraction_per_token_mean']['p95']:.4f}`"
        )
        lines.append(
            f"- active_reduction_mean across chunks: "
            f"`mean={chunk['summary']['active_reduction_mean']['mean']:.4f}`, "
            f"`min={chunk['summary']['active_reduction_mean']['min']:.4f}`, "
            f"`p95={chunk['summary']['active_reduction_mean']['p95']:.4f}`"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Analyze backbone compute regularity from routed traces.")
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--resident-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=str, default="backbone_compute_regularity")
    parser.add_argument("--chunk-size", type=int, default=0)
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
    summary["chunk_stability"] = analyze_chunks(state_dict, resident_set, args.chunk_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_prefix}.json"
    md_path = args.output_dir / f"{args.output_prefix}.md"
    json_path.write_text(json.dumps(summary, indent=2))
    md_path.write_text(build_markdown(summary))
    print(f"Saved compute regularity JSON to {json_path}")
    print(f"Saved compute regularity Markdown to {md_path}")


if __name__ == "__main__":
    main()
