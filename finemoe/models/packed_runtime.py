from __future__ import annotations

from typing import Iterable, List

import torch


def ensure_no_prefetch_runtime(module) -> None:
    matcher = getattr(module, "expert_map_matcher", None)
    prefetch_distance = getattr(matcher, "prefetch_distance", 0) if matcher else 0
    batch_prefetch_mode = getattr(module, "batch_prefetch_mode", False)
    if prefetch_distance > 0 or batch_prefetch_mode:
        raise NotImplementedError(
            "Packed-expert runtime currently supports demand/backbone execution only "
            "(prefetch_distance=0, no batch_prefetch_mode)."
        )
    if getattr(module, "expert_dispatcher", None) is None:
        raise RuntimeError("expert_dispatcher is not initialized for packed-expert block")
    if getattr(module, "layer_id", None) is None:
        raise RuntimeError("layer_id is not initialized for packed-expert block")


def dispatch_packed_experts(
    *,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
    layer_id: int,
    expert_dispatcher,
) -> torch.Tensor:
    tokens, hidden_dim = hidden_states.shape
    router_mask = torch.zeros(
        (tokens, num_experts),
        dtype=torch.bool,
        device=hidden_states.device,
    )
    router_mask.scatter_(1, top_k_index, True)

    token_weights = torch.zeros(
        (tokens, num_experts),
        dtype=top_k_weights.dtype,
        device=top_k_weights.device,
    )
    token_weights.scatter_add_(1, top_k_index, top_k_weights)

    active_experts = torch.nonzero(router_mask.any(dim=0), as_tuple=False).flatten().tolist()
    if not active_experts:
        return torch.zeros(
            (tokens, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    expert_dispatcher.set_inputs(hidden_states, router_mask)
    expert_dispatcher.set_expected_queue(len(active_experts))

    for expert_idx in active_experts:
        expert_dispatcher.enqueue_expert(layer_id, int(expert_idx))

    results = expert_dispatcher.wait_expert()
    final_hidden_states = torch.zeros(
        (tokens, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    for output_tensor, _, expert_idx, _ in results:
        expert_idx = int(expert_idx)
        token_idx = torch.where(router_mask[:, expert_idx])[0]
        weighted = output_tensor.to(
            device=final_hidden_states.device,
            dtype=final_hidden_states.dtype,
        ) * token_weights[token_idx, expert_idx].to(final_hidden_states.dtype).unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, weighted)

    return final_hidden_states
