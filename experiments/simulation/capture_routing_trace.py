"""Capture MoE routing trace from any HuggingFace MoE model.

Runs forward pass on prompts, hooks into MoE layers to capture
routing decisions (which experts are selected per token per layer),
and saves a state.pkl compatible with backbone_moe.simulator.

Works with any MoE model supported by transformers (Qwen, DeepSeek, Mixtral).
Does NOT require FineMoE runtime.

任何 HuggingFace MoE 模型的路由 trace 抓取。
不需要 FineMoE runtime。
"""

import argparse
import json
import pickle
import hashlib
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np


def get_moe_gates(model):
    """Find all MoE gate (Linear) modules in the model.

    Returns (gate_module, parent_moe_module, layer_index) tuples.
    Hooks on the gate directly to avoid re-running the MoE forward.
    """
    gates = []
    moe_idx = 0
    for name, module in model.named_modules():
        if hasattr(module, 'gate') and hasattr(module, 'experts'):
            gate = module.gate
            topk = getattr(module, 'num_experts_per_tok',
                   getattr(module, 'top_k', 4))
            gates.append((name, gate, topk, moe_idx))
            moe_idx += 1
    return gates


def capture_trace(model, tokenizer, prompts, max_length, device):
    """Run forward pass and capture routing decisions per token per layer."""
    model.eval()
    gates = get_moe_gates(model)
    num_layers = len(gates)
    num_experts = None

    # Storage for routing matrices
    traces = {}

    # Hook to capture routing decisions
    routing_log = []

    def make_gate_hook(layer_idx, topk):
        def hook_fn(module, input, output):
            # output of gate Linear = logits [num_tokens, num_experts]
            logits = output.detach()
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            routing_weights = torch.softmax(logits, dim=-1)
            _, selected = torch.topk(routing_weights, topk, dim=-1)
            routing_log.append({
                'layer': layer_idx,
                'routing_weights': routing_weights.cpu(),
                'selected_experts': selected.cpu(),
                'num_experts': routing_weights.shape[-1],
            })
        return hook_fn

    # Register hooks on gate modules (not MoE modules)
    hooks = []
    for name, gate, topk, idx in gates:
        hooks.append(gate.register_forward_hook(make_gate_hook(idx, topk)))

    for prompt_idx, prompt in enumerate(prompts):
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=max_length, padding=False
        ).to(device)

        seq_id = hashlib.md5(prompt.encode()).hexdigest()
        routing_log.clear()

        with torch.no_grad():
            outputs = model(**inputs)

        if not routing_log:
            print(f"  WARNING: no routing captured for prompt {prompt_idx}")
            continue

        num_experts = routing_log[0]['num_experts']
        num_tokens = inputs.input_ids.shape[1]

        # Build per-iter routing matrix [num_layers, num_experts]
        # Each cell = number of tokens routed to that expert
        matrix = torch.zeros(num_layers, num_experts)
        iters = []

        for token_idx in range(num_tokens):
            nodes = torch.zeros(num_layers, num_experts)
            probs = torch.zeros(num_layers, num_experts)
            for entry in routing_log:
                layer = entry['layer']
                if token_idx < entry['selected_experts'].shape[0]:
                    for eid in entry['selected_experts'][token_idx]:
                        nodes[layer, eid.item()] = 1.0
                if token_idx < entry['routing_weights'].shape[0]:
                    probs[layer] = entry['routing_weights'][token_idx]

            matrix += nodes
            # Create a dummy embed (zero) since we don't have the real embedding
            # backbone_moe.simulator only needs 'nodes' and 'probs'
            iters.append({
                'stage': 'decode',
                'embed': torch.zeros(model.config.hidden_size, dtype=torch.bfloat16),
                'nodes': nodes,
                'probs': probs,
                'preds': torch.zeros(num_layers, num_experts),
            })

        traces[seq_id] = {
            'matrix': matrix,
            'iters': iters,
        }
        print(f"  prompt {prompt_idx}: {num_tokens} tokens, {num_layers} layers, {num_experts} experts")

    # Remove hooks
    for h in hooks:
        h.remove()

    return traces


def main():
    parser = argparse.ArgumentParser(
        description="Capture MoE routing trace from HuggingFace model"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--prompt-file", type=str, required=True,
                        help="JSON file with list of prompts")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    print(f"Loading prompts from {args.prompt_file}...")
    with open(args.prompt_file) as f:
        prompt_data = json.load(f)
    prompts = []
    for item in prompt_data:
        if isinstance(item, dict):
            prompts.append(item.get("prompt", str(item)))
        else:
            prompts.append(str(item))
    prompts = prompts[:args.num_prompts]
    print(f"  {len(prompts)} prompts loaded")

    print("Capturing routing trace...")
    traces = capture_trace(model, tokenizer, prompts, args.max_length, args.device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(traces, f)
    print(f"Saved trace ({len(traces)} sequences) to {output_path}")


if __name__ == "__main__":
    main()
