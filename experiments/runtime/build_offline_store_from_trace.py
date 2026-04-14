import argparse
import pickle
from pathlib import Path

import torch

try:
    from finemoe.runtime.model_offload import ExpertMapStore
except ImportError:
    raise ImportError(
        "This script requires FineMoE runtime. "
        "See patches/README.md for installation instructions."
    )


def iter_trace_batches(state_dict):
    for entry in state_dict.values():
        embeds = []
        probs = []
        for it in entry.get("iters", []):
            embed = it.get("embed")
            prob = it.get("probs")
            if embed is None or prob is None:
                continue
            embeds.append(embed.detach().to(torch.float32).cpu())
            probs.append(prob.detach().to(torch.float32).cpu())
        if embeds and probs:
            yield torch.stack(embeds, dim=0), torch.stack(probs, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="Build runtime offline store files (~embed~/~traj~) from a traced state.pkl."
    )
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=1000)
    parser.add_argument("--prefetch-distance", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.state_file, "rb") as f:
        state_dict = pickle.load(f)

    if not state_dict:
        raise ValueError(f"Empty trace file: {args.state_file}")

    example = next(iter(state_dict.values()))
    example_iter = next(iter(example["iters"]))
    num_layers, num_experts = example_iter["probs"].shape
    embed_dim = example_iter["embed"].shape[-1]

    store = ExpertMapStore(
        capacity=args.capacity,
        num_layers=num_layers,
        num_experts=num_experts,
        embed_dim=embed_dim,
        prefetch_distance=args.prefetch_distance,
        device=args.device,
    )

    total_sequences = 0
    total_tokens = 0
    for embeds, probs in iter_trace_batches(state_dict):
        store.add(embeds=embeds, expert_maps=probs)
        total_sequences += 1
        total_tokens += int(embeds.shape[0])

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    store.export_store_data(str(args.output_prefix))
    print(
        f"Saved offline store to {args.output_prefix} "
        f"(capacity={args.capacity}, sequences={total_sequences}, tokens={total_tokens})"
    )


if __name__ == "__main__":
    main()
