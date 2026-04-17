from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from transformers import PretrainedConfig

from .hf_config import parse_expert_layout, parse_moe_param, parse_packed_expert_tensor


@dataclass(frozen=True)
class SyntheticTensorEntry:
    name: str
    tensor: torch.Tensor
    layer_id: Optional[int]
    expert_idx: Optional[int]
    expert_group: Optional[str]
    tensor_role: Optional[str]
    source_name: str


def expand_tensor_for_offload(
    param_name: str, tensor: torch.Tensor, config: PretrainedConfig
) -> List[SyntheticTensorEntry]:
    if parse_expert_layout(config) != "packed":
        return [
            SyntheticTensorEntry(
                name=param_name,
                tensor=tensor,
                layer_id=None,
                expert_idx=None,
                expert_group=None,
                tensor_role=None,
                source_name=param_name,
            )
        ]

    layer_id, expert_group, tensor_role = parse_packed_expert_tensor(param_name, config)
    if layer_id is None:
        return [
            SyntheticTensorEntry(
                name=param_name,
                tensor=tensor,
                layer_id=None,
                expert_idx=None,
                expert_group=None,
                tensor_role=None,
                source_name=param_name,
            )
        ]

    if expert_group == "routed_experts":
        _, num_experts, _, _, _ = parse_moe_param(config)
        if tensor.shape[0] != num_experts:
            raise RuntimeError(
                f"Packed routed tensor {param_name!r} expected first dim {num_experts}, got {tuple(tensor.shape)}"
            )
        return [
            SyntheticTensorEntry(
                name=f"layers.{layer_id}.mlp.experts.{expert_idx}.{tensor_role}",
                tensor=tensor[expert_idx],
                layer_id=layer_id,
                expert_idx=expert_idx,
                expert_group=expert_group,
                tensor_role=tensor_role,
                source_name=param_name,
            )
            for expert_idx in range(num_experts)
        ]

    synthetic_name = param_name
    if expert_group == "router":
        synthetic_name = f"layers.{layer_id}.mlp.router.{tensor_role}.weight"
    elif expert_group == "shared_experts":
        synthetic_name = f"layers.{layer_id}.mlp.shared_experts.{tensor_role}.weight"

    return [
        SyntheticTensorEntry(
            name=synthetic_name,
            tensor=tensor,
            layer_id=layer_id,
            expert_idx=None,
            expert_group=expert_group,
            tensor_role=tensor_role,
            source_name=param_name,
        )
    ]


def expand_state_dict_for_offload(
    state_dict: dict[str, torch.Tensor], config: PretrainedConfig
) -> Iterable[SyntheticTensorEntry]:
    for param_name, tensor in state_dict.items():
        yield from expand_tensor_for_offload(param_name, tensor, config)
