from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from transformers import PretrainedConfig

from .hf_config import (
    parse_expert_layout,
    get_packed_expert_schema,
    parse_moe_param,
    parse_packed_expert_tensor,
)


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

    schema = get_packed_expert_schema(config)
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
        prefix = param_name.rsplit(".", 1)[0]
        entries: List[SyntheticTensorEntry] = []
        for expert_idx in range(num_experts):
            expert_tensor = tensor[expert_idx]
            if tensor_role == "gate_up_proj":
                if expert_tensor.shape[0] % 2 != 0:
                    raise RuntimeError(
                        f"Packed gate_up tensor {param_name!r} must split evenly into w1/w3, got shape {tuple(expert_tensor.shape)}"
                    )
                first_proj, second_proj = expert_tensor.chunk(2, dim=0)
                first_name = schema.gate_name
                second_name = schema.up_name
                entries.extend(
                    [
                        SyntheticTensorEntry(
                            name=f"{prefix}.{expert_idx}.{first_name}.weight",
                            tensor=first_proj,
                            layer_id=layer_id,
                            expert_idx=expert_idx,
                            expert_group=expert_group,
                            tensor_role=first_name,
                            source_name=param_name,
                        ),
                        SyntheticTensorEntry(
                            name=f"{prefix}.{expert_idx}.{second_name}.weight",
                            tensor=second_proj,
                            layer_id=layer_id,
                            expert_idx=expert_idx,
                            expert_group=expert_group,
                            tensor_role=second_name,
                            source_name=param_name,
                        ),
                    ]
                )
            elif tensor_role == "down_proj":
                down_name = schema.down_name
                entries.append(
                    SyntheticTensorEntry(
                        name=f"{prefix}.{expert_idx}.{down_name}.weight",
                        tensor=expert_tensor,
                        layer_id=layer_id,
                        expert_idx=expert_idx,
                        expert_group=expert_group,
                        tensor_role=down_name,
                        source_name=param_name,
                    )
                )
            else:
                raise RuntimeError(
                    f"Unsupported routed packed tensor role {tensor_role!r} for {param_name!r}"
                )
        return entries

    return [
        SyntheticTensorEntry(
            name=param_name,
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
