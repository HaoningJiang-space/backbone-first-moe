from typing import Optional, Tuple
import re

import torch
from transformers import PretrainedConfig


def _config_arch_string(config: PretrainedConfig) -> str:
    architecture = ""
    if getattr(config, "architectures", None):
        architecture = config.architectures[0] or ""
    if not architecture:
        architecture = getattr(config, "model_type", "") or ""
    if not architecture:
        architecture = config.__class__.__name__
    return architecture.lower()


def parse_moe_architecture(config: PretrainedConfig) -> str:
    arch = _config_arch_string(config)
    model_type = (getattr(config, "model_type", "") or "").lower()

    if "qwen" in arch or model_type == "qwen2_moe":
        return "qwen"
    if "olmoe" in arch or model_type == "olmoe":
        return "olmoe"
    if "mixtral" in arch or model_type == "mixtral":
        return "mixtral"
    if "deepseekv2" in arch or "deepseek_v2" in arch or model_type == "deepseek_v2":
        return "deepseek_v2"
    if "deepseekv3" in arch or "deepseek_v3" in arch or model_type == "deepseek_v3":
        return "deepseek_v3"

    raise RuntimeError(f"Unsupported architecture {_config_arch_string(config)}")


def parse_expert_layout(config: PretrainedConfig) -> str:
    arch = parse_moe_architecture(config)
    if arch in {"qwen", "olmoe"}:
        return "modulelist"
    if arch in {"mixtral", "deepseek_v2", "deepseek_v3"}:
        return "packed"
    raise RuntimeError(f"Unsupported architecture {arch}")


def parse_expert_dtype(config: PretrainedConfig) -> torch.dtype:
    dtype = config.torch_dtype
    if dtype is None:
        return torch.float32
    return dtype


def parse_expert_dtype_id(config: PretrainedConfig) -> int:
    dtype = parse_expert_dtype(config)
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    if dtype == torch.float16:
        return 2
    raise RuntimeError(f"Unsupported expert dtype {dtype}")


def parse_moe_param(config: PretrainedConfig) -> Tuple[int, int, int, int, int]:
    arch = parse_moe_architecture(config)
    num_encoder_layers = 0
    num_layers = config.num_hidden_layers
    embed_dim = config.hidden_size

    if arch in {"qwen", "olmoe"}:
        num_experts = config.num_experts
        top_k = config.num_experts_per_tok
    elif arch == "mixtral":
        num_experts = config.num_local_experts
        top_k = config.num_experts_per_tok
    elif arch in {"deepseek_v2", "deepseek_v3"}:
        num_experts = config.n_routed_experts
        top_k = config.num_experts_per_tok
    else:
        raise RuntimeError(f"Unsupported architecture {arch}")

    return num_layers, num_experts, num_encoder_layers, embed_dim, top_k


def parse_packed_expert_tensor(param_name: str, config: PretrainedConfig) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    arch = parse_moe_architecture(config)
    if parse_expert_layout(config) != "packed":
        return None, None, None

    routed = re.findall(r"layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$", param_name)
    if routed:
        layer_id, tensor_role = routed[0]
        return int(layer_id), "routed_experts", tensor_role

    shared = re.findall(r"layers\.(\d+)\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)\.weight$", param_name)
    if shared:
        layer_id, tensor_role = shared[0]
        return int(layer_id), "shared_experts", tensor_role

    router = re.findall(r"layers\.(\d+)\.mlp\.gate\.weight$", param_name)
    if router:
        layer_id = router[0]
        return int(layer_id), "router", "gate"

    return None, None, None


def parse_expert_id(param_name: str, config: PretrainedConfig) -> Tuple[Optional[int], Optional[int]]:
    arch = parse_moe_architecture(config)
    _, _, num_encoder_layers, _, _ = parse_moe_param(config)

    if arch in {"qwen", "olmoe"}:
        decoder_sparse_step = 1
        layer_type = "decoder"
        result = re.findall(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.", param_name)
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
        else:
            return None, None
    elif parse_expert_layout(config) == "packed":
        routed = re.findall(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_up_proj|down_proj)$", param_name)
        if routed:
            layer_id, expert_id, _ = routed[0]
            return int(layer_id), int(expert_id)
        layer_id, expert_group, tensor_role = parse_packed_expert_tensor(param_name, config)
        raise RuntimeError(
            f"Packed-expert architecture {arch} does not expose per-expert tensor ids via parameter names; "
            f"{param_name!r} maps to packed tensor ({expert_group}, {tensor_role}) at layer {layer_id}. "
            "It requires a slice-based runtime path instead of parse_expert_id()."
        )
    else:
        raise ValueError(f"{arch} not supported")

    if layer_type == "decoder":
        layer_id = layer_id // decoder_sparse_step + num_encoder_layers
    else:
        raise ValueError(f"Unsupported layer type {layer_type}")

    return layer_id, expert_id
