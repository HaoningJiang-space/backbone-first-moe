#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finemoe.backbone.runtime_eval import (
    RuntimeEvalConfig,
    build_model,
    build_tokenizer,
    cleanup_model,
    evaluate_runtime_with_components,
    prepare_prompts,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run Qwen modulelist control/no-control ceiling comparison in a single Python process."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--offload-path", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device-memory-ratio", type=float, default=0.10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--store-capacity", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-profile-fraction", type=float, default=0.2)
    return parser


def log_stage(output_root: Path, stage: str):
    print(f"[ceiling-singleproc] {stage}", flush=True)
    launcher_log = output_root / "launcher.log"
    launcher_log.parent.mkdir(parents=True, exist_ok=True)
    with launcher_log.open("a") as fh:
        fh.write(stage + "\n")


def make_cfg(args, output_path: Path, *, resident_file: str = "", budget_override: int = 0,
             no_control_mode: bool = False, no_tail_wait_mode: bool = False,
             tag: str = "runtime_eval") -> RuntimeEvalConfig:
    return RuntimeEvalConfig(
        model_path=args.model_path,
        prompt_file=args.prompt_file,
        output=output_path,
        offload_path=args.offload_path,
        device_memory_ratio=args.device_memory_ratio,
        prefetch_distance=0,
        store_prefix="",
        resident_expert_ids_file=resident_file,
        sparse_budget_bytes_override=budget_override,
        device=args.device,
        eval_mode="offline",
        batch_size=args.batch_size,
        num_prompts=args.num_prompts,
        seed=args.seed,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        store_capacity=args.store_capacity,
        batch_prefetch=False,
        no_control_mode=no_control_mode,
        no_tail_wait_mode=no_tail_wait_mode,
        tag=tag,
    )


def run_eval(cfg: RuntimeEvalConfig, prompts, tokenizer):
    model = build_model(cfg)
    try:
        return evaluate_runtime_with_components(cfg, model, tokenizer, prompts)
    finally:
        cleanup_model(model)


def run_eval_pair_with_reused_model(warmup_cfg: RuntimeEvalConfig, pair_cfg: RuntimeEvalConfig, prompts, tokenizer):
    model = build_model(warmup_cfg)
    try:
        warmup = evaluate_runtime_with_components(warmup_cfg, model, tokenizer, prompts)
        pair = evaluate_runtime_with_components(pair_cfg, model, tokenizer, prompts)
        return warmup, pair
    finally:
        cleanup_model(model)


def run_no_tail_wait_pair(warmup_cfg: RuntimeEvalConfig, pair_cfg: RuntimeEvalConfig, prompts, tokenizer):
    model = build_model(warmup_cfg)
    try:
        start_capture = getattr(model.engine, "start_no_tail_wait_capture", None)
        stop_capture = getattr(model.engine, "stop_no_tail_wait_capture", None)
        activate = getattr(model.engine, "activate_no_tail_wait_mode", None)
        deactivate = getattr(model.engine, "deactivate_no_tail_wait_mode", None)
        if start_capture is None or stop_capture is None or activate is None or deactivate is None:
            raise RuntimeError("Engine does not expose required no-tail-wait controls")
        start_capture()
        warmup = evaluate_runtime_with_components(warmup_cfg, model, tokenizer, prompts)
        stop_capture()
        activate()
        try:
            pair = evaluate_runtime_with_components(pair_cfg, model, tokenizer, prompts)
        finally:
            deactivate()
        return warmup, pair
    finally:
        cleanup_model(model)


def select_resident(args, budget_bytes: int, output_dir: Path) -> Path:
    resident_json = output_dir / "qwen_lane_mem0p10.json"
    cmd = [
        "python",
        "-m",
        "experiments.simulation.select_adaptive_resident_set",
        "--state-file",
        str(args.state_file),
        "--model-path",
        args.model_path,
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        "qwen_lane",
        "--memory-ratios",
        str(args.device_memory_ratio),
        "--selection-method",
        "frontier_prefix",
        "--profile-fraction",
        str(args.selection_profile_fraction),
        "--prefetch-windows",
        "0",
        "--sparse-budget-bytes",
        str(budget_bytes),
    ]
    subprocess.run(cmd, check=True)
    return resident_json


def run_mode(args, output_root: Path, prompts, tokenizer, budget_bytes: int, resident_json: Path, *,
             mode_name: str, no_control_mode: bool, no_tail_wait_mode: bool = False):
    mode_root = output_root / mode_name
    warmup_dir = mode_root / "warmup"
    pair_dir = mode_root / "pair1"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)

    a_warmup_cfg = make_cfg(
        args,
        warmup_dir / "qwen_A_warmup.json",
        budget_override=budget_bytes,
        no_control_mode=no_control_mode,
        no_tail_wait_mode=no_tail_wait_mode,
        tag=f"{mode_name}_warmup_A",
    )
    c_warmup_cfg = make_cfg(
        args,
        warmup_dir / "qwen_C_warmup.json",
        resident_file=str(resident_json),
        budget_override=budget_bytes,
        no_control_mode=no_control_mode,
        no_tail_wait_mode=no_tail_wait_mode,
        tag=f"{mode_name}_warmup_C",
    )
    a_pair_cfg = make_cfg(
        args,
        pair_dir / "qwen_A_mem0p10_lane_long.json",
        budget_override=budget_bytes,
        no_control_mode=no_control_mode,
        no_tail_wait_mode=no_tail_wait_mode,
        tag=f"{mode_name}_pair1_A",
    )
    c_pair_cfg = make_cfg(
        args,
        pair_dir / "qwen_C_mem0p10_lane_long.json",
        resident_file=str(resident_json),
        budget_override=budget_bytes,
        no_control_mode=no_control_mode,
        no_tail_wait_mode=no_tail_wait_mode,
        tag=f"{mode_name}_pair1_C",
    )

    if no_tail_wait_mode:
        log_stage(output_root, f"{mode_name}:warmup_a:start")
        log_stage(output_root, f"{mode_name}:pair1_a:start")
        a_warmup, a_pair = run_no_tail_wait_pair(a_warmup_cfg, a_pair_cfg, prompts, tokenizer)
        log_stage(output_root, f"{mode_name}:warmup_a:done")
        log_stage(output_root, f"{mode_name}:pair1_a:done")

        log_stage(output_root, f"{mode_name}:warmup_c:start")
        log_stage(output_root, f"{mode_name}:pair1_c:start")
        c_warmup, c_pair = run_no_tail_wait_pair(c_warmup_cfg, c_pair_cfg, prompts, tokenizer)
        log_stage(output_root, f"{mode_name}:warmup_c:done")
        log_stage(output_root, f"{mode_name}:pair1_c:done")
    else:
        log_stage(output_root, f"{mode_name}:warmup_a:start")
        a_warmup = run_eval(a_warmup_cfg, prompts, tokenizer)
        log_stage(output_root, f"{mode_name}:warmup_a:done")

        log_stage(output_root, f"{mode_name}:warmup_c:start")
        c_warmup = run_eval(c_warmup_cfg, prompts, tokenizer)
        log_stage(output_root, f"{mode_name}:warmup_c:done")

        log_stage(output_root, f"{mode_name}:pair1_a:start")
        a_pair = run_eval(a_pair_cfg, prompts, tokenizer)
        log_stage(output_root, f"{mode_name}:pair1_a:done")

        log_stage(output_root, f"{mode_name}:pair1_c:start")
        c_pair = run_eval(c_pair_cfg, prompts, tokenizer)
        log_stage(output_root, f"{mode_name}:pair1_c:done")

    gain = 0.0
    if a_pair["generated_tokens_per_sec"] > 0:
        gain = (
            c_pair["generated_tokens_per_sec"] / a_pair["generated_tokens_per_sec"] - 1.0
        ) * 100.0

    summary = {
        "mode": mode_name,
        "no_control_mode": bool(no_control_mode),
        "no_tail_wait_mode": bool(no_tail_wait_mode),
        "sparse_budget_bytes": int(budget_bytes),
        "resident_expert_ids_file": str(resident_json),
        "warmup": {"A": a_warmup, "C": c_warmup},
        "pair1": {"A": a_pair, "C": c_pair, "gain_percent": gain},
    }
    (mode_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    args = build_parser().parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    plan_dir = output_root / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_cfg = make_cfg(
        args,
        plan_dir / "qwen_A_plan_mem0p10.json",
        budget_override=0,
        no_control_mode=False,
        tag="plan_A",
    )

    prompts = prepare_prompts(plan_cfg)
    tokenizer = build_tokenizer(plan_cfg)

    log_stage(output_root, "plan_a:start")
    plan_payload = run_eval(plan_cfg, prompts, tokenizer)
    log_stage(output_root, "plan_a:done")

    budget_bytes = int(plan_payload["sparse_budget_bytes"])
    log_stage(output_root, f"resident_select:start budget_bytes={budget_bytes}")
    resident_json = select_resident(args, budget_bytes, plan_dir)
    log_stage(output_root, f"resident_select:done resident_file={resident_json}")

    control_summary = run_mode(
        args,
        output_root,
        prompts,
        tokenizer,
        budget_bytes,
        resident_json,
        mode_name="control",
        no_control_mode=False,
    )
    no_control_summary = run_mode(
        args,
        output_root,
        prompts,
        tokenizer,
        budget_bytes,
        resident_json,
        mode_name="no_control",
        no_control_mode=True,
    )
    no_tail_wait_summary = run_mode(
        args,
        output_root,
        prompts,
        tokenizer,
        budget_bytes,
        resident_json,
        mode_name="no_tail_wait",
        no_control_mode=False,
        no_tail_wait_mode=True,
    )

    summary = {
        "budget_bytes": budget_bytes,
        "resident_expert_ids_file": str(resident_json),
        "control": control_summary,
        "no_control": no_control_summary,
        "no_tail_wait": no_tail_wait_summary,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
