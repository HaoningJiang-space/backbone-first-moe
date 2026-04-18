import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEMO_ROOT.parent
for path in (DEMO_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import transformers

from finemoe import MoE
from configs.common.config_common import (
    device,
    state_path,
    offload_path,
    device_memory_ratio,
)
from configs.models.config_qwen import (
    model_path,
    prefetch_distance,
)
from configs.datasets.config_lmsys import (
    max_length,
    max_new_tokens,
    min_new_tokens,
)
from finemoe.utils import normalize_runtime_config, parse_expert_layout

warnings.filterwarnings("ignore")


def resolve_moe_name(resolved_model_path: str) -> str:
    return Path(resolved_model_path.rstrip("/")).name


def resolve_trace_prefetch_distance(model_config) -> int:
    normalized = normalize_runtime_config(model_config)
    if parse_expert_layout(normalized) == "packed":
        return 0
    return prefetch_distance


def load_prompts(prompt_file):
    with open(prompt_file, "r") as f:
        prompt_json = json.load(f)
    prompts = []
    for item in prompt_json:
        if isinstance(item, dict):
            prompts.append(item["prompt"])
        else:
            prompts.append(str(item))
    return prompts


def inference(
    model,
    tokenizer,
    generate_config,
    prompts,
    max_length,
    max_new_tokens,
    min_new_tokens,
):
    inputs = tokenizer(
        prompts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    output_ids = model.generate(
        inputs.input_ids.to(device),
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        attention_mask=inputs.attention_mask.to(device),
        do_sample=False,
        **generate_config,
    )

    return inputs.input_ids.detach().cpu(), output_ids.detach().cpu()


def main():
    parser = argparse.ArgumentParser(description="Prepare routed MoE state from a fixed prompt file.")
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    prompts = load_prompts(args.prompt_file)
    if args.sample_size is not None:
        prompts = prompts[:args.sample_size]
    sample_size = len(prompts)

    print("")
    print("**********")
    print("Preparing custom workload state")
    resolved_model_path = args.model_path or model_path
    print(f"model: {resolved_model_path}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"prompt_file: {args.prompt_file}")
    print(f"sample_size: {sample_size}")
    print("**********")
    print("")

    moe_name = resolve_moe_name(resolved_model_path)

    model_config = transformers.AutoConfig.from_pretrained(
        resolved_model_path,
        trust_remote_code=True,
    )
    trace_prefetch_distance = resolve_trace_prefetch_distance(model_config)

    model = MoE(
        resolved_model_path,
        {
            "offload_path": os.path.join(offload_path, moe_name),
            "device_memory_ratio": device_memory_ratio,
            "prefetch_distance": trace_prefetch_distance,
            "store_capacity": None,
            "device": device,
            "eval_batch_size": args.batch_size,
            "eval_max_length": max_length,
            "eval_mode": "online",
        }
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        resolved_model_path,
        device=device,
        clean_up_tokenization_spaces=True,
        trust_remote_code=True,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    generate_config = {"pad_token_id": tokenizer.pad_token_id}
    all_seq_ids = []
    input_ids_list = []
    output_ids_list = []
    seen_seq_count = 0

    for start in range(0, sample_size, args.batch_size):
        batch_prompts = prompts[start:start + args.batch_size]
        print(f"Tracing batch {start}..{start + len(batch_prompts) - 1}")
        input_ids, output_ids = inference(
            model=model,
            tokenizer=tokenizer,
            generate_config=generate_config,
            prompts=batch_prompts,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
        )
        input_ids_list.extend(input_ids)
        output_ids_list.extend(output_ids)

        current_seq_ids = list(model.engine.expert_tracer.trace.keys())
        new_seq_ids = current_seq_ids[seen_seq_count:]
        if len(new_seq_ids) != len(batch_prompts):
            raise RuntimeError(
                f"Expected {len(batch_prompts)} new traces, got {len(new_seq_ids)}"
            )
        all_seq_ids.extend(new_seq_ids)
        seen_seq_count = len(current_seq_ids)

    traj_dict = {}
    for seq_id, prompt, input_ids_row, output_ids_row in zip(
        all_seq_ids,
        prompts,
        input_ids_list,
        output_ids_list,
    ):
        traj_dict[seq_id] = {
            "matrix": model.engine.expert_tracer.trace[seq_id].matrix,
            "iters": model.engine.expert_tracer.trace[seq_id].iters,
        }

    output_path = f"{state_path}/{moe_name}~{args.dataset_name}~{sample_size}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(traj_dict, f)

    print(f"Saved routed state to {output_path}")


if __name__ == "__main__":
    main()
