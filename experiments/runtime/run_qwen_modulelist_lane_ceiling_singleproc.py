#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
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
    load_prompts,
    reset_runtime_measurement_state,
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
    parser.add_argument("--num-measured-slices", type=int, default=4)
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


def prepare_prompt_subsets(prompt_file: Path, *, seed: int, num_prompts: int, num_measured_slices: int):
    prompts = load_prompts(prompt_file)
    random.Random(seed).shuffle(prompts)
    required = num_prompts * (1 + num_measured_slices)
    if len(prompts) < required:
        raise RuntimeError(
            f"Need at least {required} prompts for warmup + {num_measured_slices} measured slices, got {len(prompts)}"
        )
    warmup_prompts = prompts[:num_prompts]
    measured_slices = []
    for slice_idx in range(num_measured_slices):
        start = num_prompts * (slice_idx + 1)
        measured_slices.append(prompts[start:start + num_prompts])
    return warmup_prompts, measured_slices


def _aggregate_window_results(warmup, measured_results):
    total_generated_tokens = sum(item["total_generated_tokens"] for item in measured_results)
    total_prompt_tokens = sum(item["total_prompt_tokens"] for item in measured_results)
    total_elapsed = sum(item["total_elapsed_sec"] for item in measured_results)
    generated_tps = total_generated_tokens / total_elapsed if total_elapsed > 0 else 0.0
    end_to_end_tps = (total_prompt_tokens + total_generated_tokens) / total_elapsed if total_elapsed > 0 else 0.0
    return {
        "warmup": warmup,
        "measured_slices": measured_results,
        "num_measured_slices": len(measured_results),
        "total_elapsed_sec": total_elapsed,
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": total_generated_tokens,
        "generated_tokens_per_sec": generated_tps,
        "end_to_end_tokens_per_sec": end_to_end_tps,
    }


def run_persistent_model_window(
    model,
    warmup_cfg: RuntimeEvalConfig,
    measured_cfg_factory,
    warmup_prompts,
    measured_prompt_slices,
    tokenizer,
):
    warmup = evaluate_runtime_with_components(warmup_cfg, model, tokenizer, warmup_prompts)
    reset_runtime_measurement_state(model, clear_dynamic_sparse_state=True)
    measured_results = []
    for slice_idx, measured_prompts in enumerate(measured_prompt_slices, start=1):
        measured_cfg = measured_cfg_factory(slice_idx)
        measured_results.append(
            evaluate_runtime_with_components(measured_cfg, model, tokenizer, measured_prompts)
        )
    return _aggregate_window_results(warmup, measured_results)


def run_no_tail_wait_persistent_window(
    model,
    warmup_cfg: RuntimeEvalConfig,
    measured_cfg_factory,
    warmup_prompts,
    measured_prompt_slices,
    tokenizer,
):
    start_capture = getattr(model.engine, "start_no_tail_wait_capture", None)
    stop_capture = getattr(model.engine, "stop_no_tail_wait_capture", None)
    activate = getattr(model.engine, "activate_no_tail_wait_mode", None)
    deactivate = getattr(model.engine, "deactivate_no_tail_wait_mode", None)
    if start_capture is None or stop_capture is None or activate is None or deactivate is None:
        raise RuntimeError("Engine does not expose required no-tail-wait controls")
    start_capture()
    warmup = evaluate_runtime_with_components(warmup_cfg, model, tokenizer, warmup_prompts)
    stop_capture()
    reset_runtime_measurement_state(model, clear_dynamic_sparse_state=True)
    activate()
    try:
        reset_runtime_measurement_state(model)
        measured_results = []
        for slice_idx, measured_prompts in enumerate(measured_prompt_slices, start=1):
            measured_cfg = measured_cfg_factory(slice_idx)
            measured_results.append(
                evaluate_runtime_with_components(measured_cfg, model, tokenizer, measured_prompts)
            )
    finally:
        deactivate()
    return _aggregate_window_results(warmup, measured_results)


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


def run_mode(args, output_root: Path, warmup_prompts, measured_prompt_slices, tokenizer, budget_bytes: int, resident_json: Path, *,
             mode_name: str, no_control_mode: bool, no_tail_wait_mode: bool = False):
    mode_root = output_root / mode_name
    warmup_dir = mode_root / "warmup"
    measured_dir = mode_root / "measured"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    measured_dir.mkdir(parents=True, exist_ok=True)

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
    def make_a_measured_cfg(slice_idx: int):
        return make_cfg(
            args,
            measured_dir / f"slice{slice_idx}" / "qwen_A_mem0p10_lane_long.json",
            budget_override=budget_bytes,
            no_control_mode=no_control_mode,
            no_tail_wait_mode=no_tail_wait_mode,
            tag=f"{mode_name}_slice{slice_idx}_A",
        )

    def make_c_measured_cfg(slice_idx: int):
        return make_cfg(
            args,
            measured_dir / f"slice{slice_idx}" / "qwen_C_mem0p10_lane_long.json",
            resident_file=str(resident_json),
            budget_override=budget_bytes,
            no_control_mode=no_control_mode,
            no_tail_wait_mode=no_tail_wait_mode,
            tag=f"{mode_name}_slice{slice_idx}_C",
        )

    log_stage(output_root, f"{mode_name}:build_a:start")
    a_model = build_model(a_warmup_cfg)
    log_stage(output_root, f"{mode_name}:build_a:done")
    try:
        log_stage(output_root, f"{mode_name}:warmup_a:start")
        log_stage(output_root, f"{mode_name}:measured_a:start")
        if no_tail_wait_mode:
            a_window = run_no_tail_wait_persistent_window(
                a_model,
                a_warmup_cfg,
                make_a_measured_cfg,
                warmup_prompts,
                measured_prompt_slices,
                tokenizer,
            )
        else:
            a_window = run_persistent_model_window(
                a_model,
                a_warmup_cfg,
                make_a_measured_cfg,
                warmup_prompts,
                measured_prompt_slices,
                tokenizer,
            )
        log_stage(output_root, f"{mode_name}:warmup_a:done")
        log_stage(output_root, f"{mode_name}:measured_a:done")
    finally:
        cleanup_model(a_model)
        log_stage(output_root, f"{mode_name}:cleanup_a:done")

    log_stage(output_root, f"{mode_name}:build_c:start")
    c_model = build_model(c_warmup_cfg)
    log_stage(output_root, f"{mode_name}:build_c:done")
    try:
        log_stage(output_root, f"{mode_name}:warmup_c:start")
        log_stage(output_root, f"{mode_name}:measured_c:start")
        if no_tail_wait_mode:
            c_window = run_no_tail_wait_persistent_window(
                c_model,
                c_warmup_cfg,
                make_c_measured_cfg,
                warmup_prompts,
                measured_prompt_slices,
                tokenizer,
            )
        else:
            c_window = run_persistent_model_window(
                c_model,
                c_warmup_cfg,
                make_c_measured_cfg,
                warmup_prompts,
                measured_prompt_slices,
                tokenizer,
            )
        log_stage(output_root, f"{mode_name}:warmup_c:done")
        log_stage(output_root, f"{mode_name}:measured_c:done")
    finally:
        cleanup_model(c_model)
        log_stage(output_root, f"{mode_name}:cleanup_c:done")

    gain = 0.0
    if a_window["generated_tokens_per_sec"] > 0:
        gain = (
            c_window["generated_tokens_per_sec"] / a_window["generated_tokens_per_sec"] - 1.0
        ) * 100.0

    summary = {
        "mode": mode_name,
        "no_control_mode": bool(no_control_mode),
        "no_tail_wait_mode": bool(no_tail_wait_mode),
        "sparse_budget_bytes": int(budget_bytes),
        "resident_expert_ids_file": str(resident_json),
        "num_measured_slices": len(measured_prompt_slices),
        "window": {"A": a_window, "C": c_window, "gain_percent": gain},
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

    warmup_prompts, measured_prompt_slices = prepare_prompt_subsets(
        args.prompt_file,
        seed=args.seed,
        num_prompts=args.num_prompts,
        num_measured_slices=args.num_measured_slices,
    )
    tokenizer = build_tokenizer(plan_cfg)

    log_stage(output_root, "plan_a:start")
    plan_payload = run_eval(plan_cfg, measured_prompt_slices[0], tokenizer)
    log_stage(output_root, "plan_a:done")

    budget_bytes = int(plan_payload["sparse_budget_bytes"])
    log_stage(output_root, f"resident_select:start budget_bytes={budget_bytes}")
    resident_json = select_resident(args, budget_bytes, plan_dir)
    log_stage(output_root, f"resident_select:done resident_file={resident_json}")

    control_summary = run_mode(
        args,
        output_root,
        warmup_prompts,
        measured_prompt_slices,
        tokenizer,
        budget_bytes,
        resident_json,
        mode_name="control",
        no_control_mode=False,
    )
    no_control_summary = run_mode(
        args,
        output_root,
        warmup_prompts,
        measured_prompt_slices,
        tokenizer,
        budget_bytes,
        resident_json,
        mode_name="no_control",
        no_control_mode=True,
    )
    no_tail_wait_summary = run_mode(
        args,
        output_root,
        warmup_prompts,
        measured_prompt_slices,
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
