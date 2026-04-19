from __future__ import annotations

import torch
import time


def _build_assignment_blocks(selected_experts: torch.Tensor, routing_weights: torch.Tensor):
    tokens = selected_experts.shape[0]
    top_k = selected_experts.shape[-1]
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
    return sorted_experts, sorted_tokens, sorted_weights, start_positions.tolist(), end_positions.tolist()


def _run_modulelist_resident_lane(
    *,
    hidden_states: torch.Tensor,
    final_hidden_states: torch.Tensor,
    sorted_experts: torch.Tensor,
    sorted_tokens: torch.Tensor,
    sorted_weights: torch.Tensor,
    experts,
    blocks,
):
    compute_wall_time_sec = 0.0
    token_assignments = 0
    for start, end in blocks:
        expert_idx = int(sorted_experts[start])
        token_idx = sorted_tokens[start:end]
        block_tokens = int(end - start)
        token_assignments += block_tokens
        current_state = hidden_states[token_idx]
        expert_module = experts[expert_idx]
        t0 = time.perf_counter()
        current_hidden_states = expert_module(current_state)
        compute_wall_time_sec += time.perf_counter() - t0
        current_hidden_states = current_hidden_states.to(
            device=final_hidden_states.device,
            dtype=final_hidden_states.dtype,
        )
        current_hidden_states = current_hidden_states * sorted_weights[start:end].to(
            device=final_hidden_states.device,
            dtype=final_hidden_states.dtype,
        ).unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden_states)
    return token_assignments, compute_wall_time_sec


def _run_modulelist_demand_lane(
    *,
    hidden_states: torch.Tensor,
    final_hidden_states: torch.Tensor,
    sorted_experts: torch.Tensor,
    sorted_tokens: torch.Tensor,
    sorted_weights: torch.Tensor,
    experts,
    blocks,
):
    if not blocks:
        return 0, 0.0

    payloads = []
    offload_engine = None
    for start, end in blocks:
        expert_idx = int(sorted_experts[start])
        token_idx = sorted_tokens[start:end]
        current_state = hidden_states[token_idx]
        expert_module = experts[expert_idx]
        payloads.append((start, end, token_idx, expert_module, current_state))
        expert_engine = getattr(expert_module, "offload_engine", None)
        if offload_engine is None:
            offload_engine = expert_engine
        elif offload_engine is not expert_engine:
            offload_engine = False

    grouped_runner = None
    if offload_engine not in (None, False):
        grouped_runner = getattr(offload_engine, "run_module_demand_lane_group", None)

    t0 = time.perf_counter()
    if grouped_runner is not None:
        outputs = grouped_runner(
            [expert_module for _, _, _, expert_module, _ in payloads],
            [current_state for _, _, _, _, current_state in payloads],
        )
    else:
        outputs = []
        for _, _, _, expert_module, current_state in payloads:
            runner = getattr(getattr(expert_module, "offload_engine", None), "run_module_demand_lane", None)
            if runner is None:
                outputs.append(expert_module(current_state))
            else:
                outputs.append(runner(expert_module, current_state))
    compute_wall_time_sec = time.perf_counter() - t0

    token_assignments = 0
    for (start, end, token_idx, _, _), current_hidden_states in zip(payloads, outputs):
        token_assignments += int(end - start)
        current_hidden_states = current_hidden_states.to(
            device=final_hidden_states.device,
            dtype=final_hidden_states.dtype,
        )
        current_hidden_states = current_hidden_states * sorted_weights[start:end].to(
            device=final_hidden_states.device,
            dtype=final_hidden_states.dtype,
        ).unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden_states)
    return token_assignments, compute_wall_time_sec


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

    (
        sorted_experts,
        sorted_tokens,
        sorted_weights,
        start_positions,
        end_positions,
    ) = _build_assignment_blocks(selected_experts, routing_weights)

    resident_expert_ids = set(resident_expert_ids or ())
    resident_blocks = []
    demand_blocks = []
    for start, end in zip(start_positions, end_positions):
        if int(sorted_experts[start]) in resident_expert_ids:
            resident_blocks.append((start, end))
        else:
            demand_blocks.append((start, end))

    resident_token_assignments, resident_compute_wall_time_sec = _run_modulelist_resident_lane(
        hidden_states=hidden_states,
        final_hidden_states=final_hidden_states,
        sorted_experts=sorted_experts,
        sorted_tokens=sorted_tokens,
        sorted_weights=sorted_weights,
        experts=experts,
        blocks=resident_blocks,
    )
    demand_token_assignments, demand_compute_wall_time_sec = _run_modulelist_demand_lane(
        hidden_states=hidden_states,
        final_hidden_states=final_hidden_states,
        sorted_experts=sorted_experts,
        sorted_tokens=sorted_tokens,
        sorted_weights=sorted_weights,
        experts=experts,
        blocks=demand_blocks,
    )
    active_expert_blocks = len(start_positions)
    token_assignments = int(sorted_experts.numel())
    resident_expert_blocks = len(resident_blocks)
    demand_expert_blocks = len(demand_blocks)
    compute_wall_time_sec = resident_compute_wall_time_sec + demand_compute_wall_time_sec

    if runtime_profile is not None:
        runtime_profile.record_modulelist_dispatch(
            active_expert_blocks=active_expert_blocks,
            resident_expert_blocks=resident_expert_blocks,
            demand_expert_blocks=demand_expert_blocks,
            token_assignments=token_assignments,
            resident_token_assignments=resident_token_assignments,
            demand_token_assignments=demand_token_assignments,
            expert_compute_wall_time_sec=compute_wall_time_sec,
            resident_compute_wall_time_sec=resident_compute_wall_time_sec,
            demand_compute_wall_time_sec=demand_compute_wall_time_sec,
        )

    return final_hidden_states
