import argparse
import json
import time
from pathlib import Path

import torch

from finemoe.entrypoints.big_modeling import MoE


def parse_args():
    parser = argparse.ArgumentParser(description="Probe tiny packed-MoE runtime throughput.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--offload-path", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resident-file", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device-memory-ratio", type=float, default=0.10)
    parser.add_argument("--store-capacity", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=255)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="packed_tiny_probe")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = {
        "offload_path": args.offload_path,
        "device_memory_ratio": args.device_memory_ratio,
        "resident_expert_ids_file": args.resident_file,
        "prefetch_distance": 0,
        "store_capacity": args.store_capacity,
        "device": args.device,
        "eval_batch_size": args.batch_size,
        "eval_max_length": args.prompt_length + args.max_new_tokens,
        "eval_mode": "offline",
    }
    model = MoE(args.model_path, cfg)

    input_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.prompt_length),
        device=args.device,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=args.device)
        torch.cuda.synchronize(device=args.device)

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=False,
            pad_token_id=0,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device=args.device)
        peak = torch.cuda.max_memory_allocated(device=args.device) / (1024 ** 2)
    else:
        peak = None
    elapsed = time.perf_counter() - start

    total_prompt_tokens = int(input_ids.numel())
    total_tokens = int(out.numel())
    generated_tokens = max(total_tokens - total_prompt_tokens, 0)
    payload = {
        "tag": args.tag,
        "model_path": args.model_path,
        "offload_path": args.offload_path,
        "resident_file": args.resident_file,
        "device": args.device,
        "device_memory_ratio": args.device_memory_ratio,
        "batch_size": args.batch_size,
        "prompt_length": args.prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": generated_tokens,
        "generated_tokens_per_sec": float(generated_tokens / elapsed) if elapsed > 0 else 0.0,
        "end_to_end_tokens_per_sec": float(total_tokens / elapsed) if elapsed > 0 else 0.0,
        "total_elapsed_sec": float(elapsed),
        "resident_count": len(getattr(model.engine, "resident_expert_ids", [])),
        "peak_memory_mb": peak,
        "input_shape": list(input_ids.shape),
        "output_shape": list(out.shape),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(args.output)

    if hasattr(model.engine, "archer_engine"):
        model.engine.archer_engine.clean_up_resources()


if __name__ == "__main__":
    main()
