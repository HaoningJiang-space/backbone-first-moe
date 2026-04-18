from __future__ import annotations

import torch


def dispatch_modulelist_experts(
    *,
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    experts,
) -> torch.Tensor:
    tokens, hidden_dim = hidden_states.shape
    top_k = selected_experts.shape[-1]
    final_hidden_states = torch.zeros(
        (tokens, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    if tokens == 0 or top_k == 0:
        return final_hidden_states

    token_indices = torch.arange(tokens, device=selected_experts.device).repeat_interleave(top_k)
    flat_experts = selected_experts.reshape(-1)
    flat_weights = routing_weights.reshape(-1)
    sort_order = torch.argsort(flat_experts)
    sorted_experts = flat_experts[sort_order]
    sorted_tokens = token_indices[sort_order]
    sorted_weights = flat_weights[sort_order]

    block_starts = torch.ones(sorted_experts.numel(), dtype=torch.bool, device=sorted_experts.device)
    block_starts[1:] = sorted_experts[1:] != sorted_experts[:-1]
    start_positions = torch.nonzero(block_starts, as_tuple=False).flatten()
    end_positions = torch.cat(
        (
            start_positions[1:],
            torch.tensor([sorted_experts.numel()], device=sorted_experts.device),
        )
    )

    for start, end in zip(start_positions.tolist(), end_positions.tolist()):
        expert_idx = int(sorted_experts[start])
        token_idx = sorted_tokens[start:end]
        current_state = hidden_states[token_idx]
        current_hidden_states = experts[expert_idx](current_state).to(
            device=final_hidden_states.device,
            dtype=final_hidden_states.dtype,
        )
        current_hidden_states = current_hidden_states * sorted_weights[start:end].to(
            device=final_hidden_states.device,
            dtype=final_hidden_states.dtype,
        ).unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden_states)

    return final_hidden_states
