import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers

from finemoe import MoE


@dataclass
class RuntimeEvalConfig:
    model_path: str
    prompt_file: Path
    output: Path
    offload_path: str
    device_memory_ratio: float
    prefetch_distance: int = 6
    store_prefix: str = ""
    resident_expert_ids_file: str = ""
    device: str = "cuda:0"
    eval_mode: str = "offline"
    batch_size: int = 1
    num_prompts: int = 8
    seed: int = 42
    max_length: int = 256
    max_new_tokens: int = 64
    min_new_tokens: int = 1
    store_capacity: int = 1000
    batch_prefetch: bool = False
    tag: str = "runtime_eval"


def load_prompts(prompt_file):
    with open(prompt_file, "r") as f:
        payload = json.load(f)
    prompts = []
    for item in payload:
        if isinstance(item, dict):
            prompts.append(item["prompt"])
        else:
            prompts.append(str(item))
    return prompts


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def build_model(runtime_cfg):
    model = MoE(
        runtime_cfg.model_path,
        {
            "offload_path": runtime_cfg.offload_path,
            "device_memory_ratio": runtime_cfg.device_memory_ratio,
            "resident_expert_ids_file": runtime_cfg.resident_expert_ids_file,
            "prefetch_distance": runtime_cfg.prefetch_distance,
            "store_capacity": runtime_cfg.store_capacity,
            "device": runtime_cfg.device,
            "eval_batch_size": runtime_cfg.batch_size,
            "eval_max_length": runtime_cfg.max_length,
            "eval_mode": runtime_cfg.eval_mode,
        },
    )
    if runtime_cfg.store_prefix and runtime_cfg.prefetch_distance > 0:
        model.engine.expert_tracer.expert_map_store.import_store_data(runtime_cfg.store_prefix)

    # Enable batch-aware prefetch mode on all MoE layers
    # 在所有 MoE 层上启用 batch-aware 预取模式
    if runtime_cfg.batch_prefetch:
        for moe_layer in model.engine.moe_layers:
            moe_layer.batch_prefetch_mode = True

    return model


def build_tokenizer(runtime_cfg):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        runtime_cfg.model_path,
        device=runtime_cfg.device,
        clean_up_tokenization_spaces=True,
        trust_remote_code=True,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def evaluate_runtime(runtime_cfg):
    prompts = load_prompts(runtime_cfg.prompt_file)
    random.Random(runtime_cfg.seed).shuffle(prompts)
    prompts = prompts[: runtime_cfg.num_prompts]

    model = build_model(runtime_cfg)
    tokenizer = build_tokenizer(runtime_cfg)
    generate_config = {"pad_token_id": tokenizer.pad_token_id}

    per_batch = []
    total_generated_tokens = 0
    total_prompt_tokens = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=runtime_cfg.device)

    for batch_prompts in batched(prompts, runtime_cfg.batch_size):
        inputs = tokenizer(
            batch_prompts,
            truncation=True,
            padding=True,
            max_length=runtime_cfg.max_length,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(runtime_cfg.device)
        attention_mask = inputs.attention_mask.to(runtime_cfg.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=runtime_cfg.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=runtime_cfg.max_new_tokens,
                min_new_tokens=runtime_cfg.min_new_tokens,
                do_sample=False,
                **generate_config,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=runtime_cfg.device)
        elapsed = time.perf_counter() - t0

        input_lengths = inputs.attention_mask.sum(dim=1).tolist()
        output_lengths = (output_ids != tokenizer.pad_token_id).sum(dim=1).tolist()
        generated_lengths = [max(0, int(o - i)) for i, o in zip(input_lengths, output_lengths)]

        total_generated_tokens += sum(generated_lengths)
        total_prompt_tokens += sum(int(x) for x in input_lengths)
        per_batch.append(
            {
                "batch_size": len(batch_prompts),
                "elapsed_sec": elapsed,
                "prompt_tokens": [int(x) for x in input_lengths],
                "generated_tokens": generated_lengths,
                "tokens_per_sec": float(sum(generated_lengths) / elapsed) if elapsed > 0 else 0.0,
            }
        )

    total_elapsed = sum(batch["elapsed_sec"] for batch in per_batch)
    peak_memory_mb = None
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated(device=runtime_cfg.device) / (1024 ** 2)

    resident_registry = None
    if hasattr(model.engine, "get_resident_registry"):
        resident_registry = model.engine.get_resident_registry()

    payload = {
        "tag": runtime_cfg.tag,
        "model_path": runtime_cfg.model_path,
        "prompt_file": str(runtime_cfg.prompt_file),
        "offload_path": runtime_cfg.offload_path,
        "store_prefix": runtime_cfg.store_prefix,
        "resident_expert_ids_file": runtime_cfg.resident_expert_ids_file,
        "device_memory_ratio": float(runtime_cfg.device_memory_ratio),
        "prefetch_distance": int(runtime_cfg.prefetch_distance),
        "num_prompts": len(prompts),
        "batch_size": int(runtime_cfg.batch_size),
        "max_new_tokens": int(runtime_cfg.max_new_tokens),
        "total_elapsed_sec": float(total_elapsed),
        "total_prompt_tokens": int(total_prompt_tokens),
        "total_generated_tokens": int(total_generated_tokens),
        "generated_tokens_per_sec": float(total_generated_tokens / total_elapsed) if total_elapsed > 0 else 0.0,
        "end_to_end_tokens_per_sec": float((total_prompt_tokens + total_generated_tokens) / total_elapsed) if total_elapsed > 0 else 0.0,
        "peak_memory_mb": peak_memory_mb,
        "resident_count": (
            resident_registry["admitted_count"]
            if resident_registry is not None
            else len(getattr(model.engine, "resident_expert_ids", []))
        ),
        "resident_requested_count": resident_registry["requested_count"] if resident_registry is not None else 0,
        "resident_admitted_count": resident_registry["admitted_count"] if resident_registry is not None else 0,
        "resident_clipped": resident_registry["clipped"] if resident_registry is not None else False,
        "resident_registry": resident_registry,
        "per_batch": per_batch,
    }

    runtime_cfg.output.parent.mkdir(parents=True, exist_ok=True)
    runtime_cfg.output.write_text(json.dumps(payload, indent=2))

    if hasattr(model.engine, "archer_engine"):
        model.engine.archer_engine.clean_up_resources()

    return payload
