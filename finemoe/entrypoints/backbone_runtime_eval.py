import argparse
import json
from pathlib import Path

from finemoe.backbone import RuntimeEvalConfig, evaluate_runtime


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run minimal real-machine runtime evaluation for demand/cache/backbone settings."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offload-path", type=str, required=True)
    parser.add_argument("--device-memory-ratio", type=float, required=True)
    parser.add_argument("--prefetch-distance", type=int, default=6)
    parser.add_argument("--store-prefix", type=str, default="")
    parser.add_argument("--resident-expert-ids-file", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-mode", type=str, default="offline")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--store-capacity", type=int, default=1000)
    parser.add_argument("--batch-prefetch", action="store_true",
                        help="Enable batch-aware prefetch (one call per layer instead of per-sequence)")
    parser.add_argument("--tag", type=str, default="runtime_eval")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    runtime_cfg = RuntimeEvalConfig(
        model_path=args.model_path,
        prompt_file=args.prompt_file,
        output=args.output,
        offload_path=args.offload_path,
        device_memory_ratio=args.device_memory_ratio,
        prefetch_distance=args.prefetch_distance,
        store_prefix=args.store_prefix,
        resident_expert_ids_file=args.resident_expert_ids_file,
        device=args.device,
        eval_mode=args.eval_mode,
        batch_size=args.batch_size,
        num_prompts=args.num_prompts,
        seed=args.seed,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        store_capacity=args.store_capacity,
        batch_prefetch=args.batch_prefetch,
        tag=args.tag,
    )
    payload = evaluate_runtime(runtime_cfg)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
