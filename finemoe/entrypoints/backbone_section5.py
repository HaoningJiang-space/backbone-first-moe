"""
Section 5 real-hardware evaluation sweep.
"""

import argparse
import json
import sys
from pathlib import Path

from finemoe.backbone.section5 import (
    DEFAULT_EVAL_SCRIPT,
    DEFAULT_MODEL_PATH,
    DEFAULT_OFFLOAD_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_FILE,
    DEFAULT_PYTHON,
    DEFAULT_RESIDENT_DIR,
    DEFAULT_STORE_PREFIX,
    RuntimeSweepArgs,
    build_runtime_sweep_configs,
    format_runtime_summary,
    run_runtime_config,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Section 5 sweep: run real-hardware eval across configs and memory ratios."
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--offload-path", type=str, default=DEFAULT_OFFLOAD_PATH)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--store-prefix", type=str, default=DEFAULT_STORE_PREFIX)
    parser.add_argument("--resident-dir", type=str, default=DEFAULT_RESIDENT_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python-bin", type=str, default=DEFAULT_PYTHON)
    parser.add_argument("--eval-script", type=str, default=DEFAULT_EVAL_SCRIPT)

    parser.add_argument(
        "--memory-ratios",
        type=str,
        default="0.05,0.07,0.10",
        help="Comma-separated list of device_memory_ratio values",
    )
    parser.add_argument("--prefetch-distance", type=int, default=6)
    parser.add_argument("--store-capacity", type=int, default=1000)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-new-tokens", type=int, default=1)

    parser.add_argument(
        "--configs",
        type=str,
        default="A,B,D",
        help="Comma-separated config labels to run (A=demand, B=lru+prefetch, C=backbone-only, D=backbone)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    sweep_args = RuntimeSweepArgs(
        model_path=args.model_path,
        offload_path=args.offload_path,
        prompt_file=args.prompt_file,
        store_prefix=args.store_prefix,
        resident_dir=args.resident_dir,
        output_dir=args.output_dir,
        python_bin=args.python_bin,
        eval_script=args.eval_script,
        memory_ratios=args.memory_ratios,
        prefetch_distance=args.prefetch_distance,
        store_capacity=args.store_capacity,
        device=args.device,
        batch_size=args.batch_size,
        num_prompts=args.num_prompts,
        seed=args.seed,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
    )

    configs = build_runtime_sweep_configs(sweep_args)
    enabled = set(args.configs.split(","))
    configs = [(label, cli) for label, cli in configs if label.split("_")[0] in enabled]

    if not configs:
        print("No configurations to run. Check --configs and --memory-ratios.")
        sys.exit(1)

    print(f"Will run {len(configs)} configurations:")
    for label, _ in configs:
        print(f"  - {label}")

    if args.dry_run:
        for label, cli in configs:
            output_path = Path(args.output_dir) / f"{label}.json"
            cmd = [args.python_bin, args.eval_script] + cli + ["--output", str(output_path)]
            print(f"\n[DRY RUN] {label}:")
            print(f"  {' '.join(cmd)}")
        return

    all_results = []
    for label, cli in configs:
        data = run_runtime_config(args.python_bin, args.eval_script, label, cli, args.output_dir)
        if data is not None:
            all_results.append((label, data))

    agg_path = Path(args.output_dir) / "section5_summary.json"
    agg_payload = {
        "sweep_args": {k: str(v) for k, v in vars(args).items()},
        "results": {label: data for label, data in all_results},
    }
    agg_path.write_text(json.dumps(agg_payload, indent=2))
    print(f"\nAggregated results saved to {agg_path}")
    print(format_runtime_summary(all_results))


if __name__ == "__main__":
    main()
