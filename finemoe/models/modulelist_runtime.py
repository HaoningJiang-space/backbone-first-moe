from __future__ import annotations

import torch
import time


def dispatch_modulelist_experts(
    *,
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    experts,
    resident_expert_ids=None,
    runtime_profile=None,
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

    resident_expert_ids = set(resident_expert_ids or ())
    active_expert_blocks = int(start_positions.numel())
    token_assignments = int(sorted_experts.numel())
    resident_expert_blocks = 0
    demand_expert_blocks = 0
    resident_token_assignments = 0
    demand_token_assignments = 0
    compute_wall_time_sec = 0.0

    for start, end in zip(start_positions.tolist(), end_positions.tolist()):
        expert_idx = int(sorted_experts[start])
        block_tokens = int(end - start)
        if expert_idx in resident_expert_ids:
            resident_expert_blocks += 1
            resident_token_assignments += block_tokens
        else:
            demand_expert_blocks += 1
            demand_token_assignments += block_tokens
        token_idx = sorted_tokens[start:end]
        current_state = hidden_states[token_idx]
        t0 = time.perf_counter()
        current_hidden_states = experts[expert_idx](current_state).to(
            device=final_hidden_states.device,
            dtype=final_hidden_states.dtype,
        )
        compute_wall_time_sec += time.perf_counter() - t0
        current_hidden_states = current_hidden_states * sorted_weights[start:end].to(
            device=final_hidden_states.device,
            dtype=final_hidden_states.dtype,
        ).unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden_states)

    if runtime_profile is not None:
        runtime_profile.record_modulelist_dispatch(
            active_expert_blocks=active_expert_blocks,
            resident_expert_blocks=resident_expert_blocks,
            demand_expert_blocks=demand_expert_blocks,
            token_assignments=token_assignments,
            resident_token_assignments=resident_token_assignments,
            demand_token_assignments=demand_token_assignments,
            expert_compute_wall_time_sec=compute_wall_time_sec,
        )

    return final_hidden_states
