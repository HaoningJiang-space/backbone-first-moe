from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
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
    experts_module=None,
    resident_fastpath_expert_ids=None,
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

    final_hidden_states = torch.zeros(
        (tokens, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    resident_fastpath_expert_ids = set(resident_fastpath_expert_ids or ())
    if not _supports_packed_resident_fastpath(experts_module, hidden_states.device):
        resident_fastpath_expert_ids = set()
    resident_active: list[int] = []
    demand_active: list[int] = []
    for expert_idx in active_experts:
        if expert_idx in resident_fastpath_expert_ids:
            resident_active.append(expert_idx)
        else:
            demand_active.append(expert_idx)

    if len(resident_active) == 1:
        expert_idx = resident_active[0]
        token_idx, weights = assignment_map[expert_idx]
        output_tensor = _run_packed_resident_expert(experts_module, hidden_states[token_idx], expert_idx)
        weighted = output_tensor.to(
            device=final_hidden_states.device,
            dtype=final_hidden_states.dtype,
        ) * weights.to(
            device=final_hidden_states.device,
            dtype=final_hidden_states.dtype,
        ).unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, weighted)
    elif resident_active:
        token_idx, weighted = _run_grouped_packed_resident_experts(
            experts_module=experts_module,
            hidden_states=hidden_states,
            assignment_map=assignment_map,
            resident_active=resident_active,
            output_device=final_hidden_states.device,
            output_dtype=final_hidden_states.dtype,
        )
        final_hidden_states.index_add_(0, token_idx, weighted)

    if demand_active:
        expert_dispatcher.set_inputs(hidden_states, router_mask)
        expert_dispatcher.set_expected_queue(len(demand_active))
        gpu_id = hidden_states.device.index if hidden_states.is_cuda else -1

        for expert_idx in demand_active:
            expert_dispatcher.enqueue_expert(layer_id, int(expert_idx), gpu_id, False)

        results = expert_dispatcher.wait_expert()
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


def _run_packed_resident_expert(experts_module, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
    if experts_module is None:
        raise RuntimeError("experts_module is required for packed resident fast path")
    if hasattr(experts_module, "gate_up_proj") and hasattr(experts_module, "down_proj"):
        if not _supports_packed_resident_fastpath(experts_module, hidden_states.device):
            raise RuntimeError(
                "Packed resident fast path requires gate_up_proj/down_proj to be materialized on the "
                f"execution device {hidden_states.device}, but found "
                f"{experts_module.gate_up_proj.device} / {experts_module.down_proj.device}."
            )
        gate_up_proj = experts_module.gate_up_proj[expert_idx]
        down_proj = experts_module.down_proj[expert_idx]
        gate_up = F.linear(hidden_states, gate_up_proj)
        if getattr(experts_module, "has_gate", True):
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden_states = experts_module.act_fn(gate) * up
        else:
            current_hidden_states = experts_module.act_fn(gate_up)
        return F.linear(current_hidden_states, down_proj)
    if isinstance(experts_module, torch.nn.ModuleList):
        expert_module = experts_module[expert_idx]
        expert_device = _infer_module_device(expert_module)
        if expert_device is not None and hidden_states.device != expert_device:
            hidden_states = hidden_states.to(expert_device)
        return expert_module(hidden_states)

    raise RuntimeError(
        f"Unsupported packed resident expert container type: {type(experts_module)!r}"
    )


def _run_grouped_packed_resident_experts(
    *,
    experts_module,
    hidden_states: torch.Tensor,
    assignment_map: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    resident_active: list[int],
    output_device: torch.device,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    token_parts = []
    expert_parts = []
    weight_parts = []
    for expert_idx in resident_active:
        token_idx, weights = assignment_map[expert_idx]
        token_parts.append(token_idx)
        expert_parts.append(torch.full_like(token_idx, int(expert_idx)))
        weight_parts.append(weights)

    token_idx = torch.cat(token_parts, dim=0)
    expert_idx = torch.cat(expert_parts, dim=0)
    route_weights = torch.cat(weight_parts, dim=0)

    if hasattr(experts_module, "gate_up_proj") and hasattr(experts_module, "down_proj"):
        if not _supports_packed_resident_fastpath(experts_module, hidden_states.device):
            raise RuntimeError(
                "Packed resident fast path requires gate_up_proj/down_proj to be materialized on the "
                f"execution device {hidden_states.device}, but found "
                f"{experts_module.gate_up_proj.device} / {experts_module.down_proj.device}."
            )
        expert_device = experts_module.gate_up_proj.device
        current_states = hidden_states[token_idx]
        if current_states.device != expert_device:
            current_states = current_states.to(expert_device)
        expert_idx_on_device = expert_idx.to(expert_device)
        gate_up_proj = experts_module.gate_up_proj[expert_idx_on_device]
        gate_up = torch.bmm(gate_up_proj, current_states.unsqueeze(-1)).squeeze(-1)
        if getattr(experts_module, "has_gate", True):
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden_states = experts_module.act_fn(gate) * up
        else:
            current_hidden_states = experts_module.act_fn(gate_up)
        down_proj = experts_module.down_proj[expert_idx_on_device]
        output_tensor = torch.bmm(down_proj, current_hidden_states.unsqueeze(-1)).squeeze(-1)
        weighted = output_tensor.to(
            device=output_device,
            dtype=output_dtype,
        ) * route_weights.to(
            device=output_device,
            dtype=output_dtype,
        ).unsqueeze(-1)
        return token_idx.to(output_device), weighted

    outputs = []
    token_offsets = []
    for expert in resident_active:
        local_token_idx, local_weights = assignment_map[expert]
        output_tensor = _run_packed_resident_expert(experts_module, hidden_states[local_token_idx], expert)
        weighted = output_tensor.to(
            device=output_device,
            dtype=output_dtype,
        ) * local_weights.to(
            device=output_device,
            dtype=output_dtype,
        ).unsqueeze(-1)
        outputs.append(weighted)
        token_offsets.append(local_token_idx.to(output_device))
    return torch.cat(token_offsets, dim=0), torch.cat(outputs, dim=0)


def _infer_module_device(module) -> torch.device | None:
    devices = []
    for param in module.parameters(recurse=True):
        devices.append(param.device)
    for buf in module.buffers(recurse=True):
        devices.append(buf.device)
    if not devices:
        return None
    first = devices[0]
    if any(device != first for device in devices[1:]):
        return None
    return first


def _supports_packed_resident_fastpath(experts_module, execution_device: torch.device) -> bool:
    if experts_module is None:
        return False
    if not (hasattr(experts_module, "gate_up_proj") and hasattr(experts_module, "down_proj")):
        return False
    gate_device = getattr(experts_module.gate_up_proj, "device", None)
    down_device = getattr(experts_module.down_proj, "device", None)
    if gate_device is None or down_device is None:
        return False
    if gate_device != down_device:
        return False
    return gate_device == execution_device
