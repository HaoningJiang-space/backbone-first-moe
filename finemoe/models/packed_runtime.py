from __future__ import annotations

from typing import Dict, Tuple

import torch
from transformers.modeling_utils import PreTrainedModel


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


def install_runtime_device_property(model_cls) -> None:
    if getattr(model_cls, "_archer_runtime_device_installed", False):
        return

    original_device = getattr(model_cls, "device", None)

    def _device(self):
        runtime_device = getattr(self, "_device", None)
        if runtime_device is not None:
            return torch.device(runtime_device)
        if isinstance(original_device, property) and original_device.fget is not None:
            return original_device.fget(self)
        return PreTrainedModel.device.fget(self)

    model_cls.device = property(_device)
    model_cls._archer_runtime_device_installed = True


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
    router_mask, active_experts, assignment_map = _build_packed_expert_assignments(
        top_k_index=top_k_index,
        top_k_weights=top_k_weights,
        num_experts=num_experts,
        device=hidden_states.device,
    )
    if not active_experts:
        return torch.zeros(
            (tokens, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    expert_dispatcher.set_inputs(hidden_states, router_mask)
    expert_dispatcher.set_expected_queue(len(active_experts))
    gpu_id = hidden_states.device.index if hidden_states.is_cuda else -1

    for expert_idx in active_experts:
        expert_dispatcher.enqueue_expert(layer_id, int(expert_idx), gpu_id, False)

    results = expert_dispatcher.wait_expert()
    final_hidden_states = torch.zeros(
        (tokens, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    for output_tensor, _, expert_idx, _ in results:
        expert_idx = int(expert_idx)
        token_idx, weights = assignment_map[expert_idx]
        weighted = output_tensor.to(
            device=final_hidden_states.device,
            dtype=final_hidden_states.dtype,
        ) * weights.to(
            device=final_hidden_states.device,
            dtype=final_hidden_states.dtype,
        ).unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, weighted)

    return final_hidden_states


def _build_packed_expert_assignments(
    *,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
    device: torch.device,
) -> Tuple[torch.Tensor, list[int], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    tokens, top_k = top_k_index.shape
    router_mask = torch.zeros(
        (tokens, num_experts),
        dtype=torch.bool,
        device=device,
    )
    if tokens == 0 or top_k == 0:
        return router_mask, [], {}

    router_mask.scatter_(1, top_k_index, True)
    token_indices = torch.arange(tokens, device=top_k_index.device).repeat_interleave(top_k)
    flat_experts = top_k_index.reshape(-1)
    flat_weights = top_k_weights.reshape(-1)
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

    active_experts: list[int] = []
    assignment_map: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for start, end in zip(start_positions.tolist(), end_positions.tolist()):
        expert_idx = int(sorted_experts[start])
        active_experts.append(expert_idx)
        assignment_map[expert_idx] = (
            sorted_tokens[start:end],
            sorted_weights[start:end],
        )

    return router_mask, active_experts, assignment_map
