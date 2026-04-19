from dataclasses import dataclass
from typing import Optional, Tuple
import re

import torch
from transformers import AutoConfig, PretrainedConfig


_PACKED_MOE_PREFIX = r"(?:model\.)?layers\.(\d+)\.(?:mlp|block_sparse_moe)"


@dataclass(frozen=True)
class PackedExpertSchema:
    gate_name: str
    down_name: str
    up_name: str

    @property
    def expanded_names(self) -> Tuple[str, str, str]:
        return (self.gate_name, self.down_name, self.up_name)


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


def normalize_runtime_config(config: PretrainedConfig) -> PretrainedConfig:
    arch = parse_moe_architecture(config)

    if arch == "qwen":
        default_values = {
            "rope_theta": 10000.0,
            "use_sliding_window": False,
            "sliding_window": None,
            "max_window_layers": 28,
            "attention_dropout": 0.0,
            "rope_scaling": None,
        }
        for key, value in default_values.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        if getattr(config, "use_sliding_window", False) and getattr(config, "sliding_window", None) is None:
            setattr(config, "sliding_window", 4096)

    if arch in {"deepseek_v2", "deepseek_v3"}:
        default_values = {
            "attention_bias": False,
            "mlp_bias": False,
            "first_k_dense_replace": 0,
        }
        for key, value in default_values.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        if getattr(config, "head_dim", None) is None:
            head_dim = getattr(config, "qk_rope_head_dim", None)
            if head_dim is None:
                head_dim = getattr(config, "hidden_size", 0) // max(
                    1, getattr(config, "num_attention_heads", 1)
                )
            setattr(config, "head_dim", head_dim)
        if getattr(config, "num_key_value_heads", None) is None:
            setattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", None))

    return config


def parse_expert_layout(config: PretrainedConfig) -> str:
    arch = parse_moe_architecture(config)
    if arch in {"qwen", "olmoe"}:
        return "modulelist"
    if arch in {"mixtral", "deepseek_v2", "deepseek_v3"}:
        return "packed"
    raise RuntimeError(f"Unsupported architecture {arch}")


def get_packed_expert_schema(config: PretrainedConfig) -> PackedExpertSchema:
    arch = parse_moe_architecture(config)
    if arch == "mixtral":
        return PackedExpertSchema("w1", "w2", "w3")
    if arch in {"deepseek_v2", "deepseek_v3"}:
        return PackedExpertSchema("gate_proj", "down_proj", "up_proj")
    raise RuntimeError(f"Packed expert schema is undefined for architecture {arch}")


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


def dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype in {torch.float16, torch.bfloat16, torch.int16, torch.uint16}:
        return 2
    if dtype in {torch.float32, torch.int32, torch.uint32}:
        return 4
    if dtype in {torch.float64, torch.int64, torch.uint64}:
        return 8
    if dtype in {torch.int8, torch.uint8, torch.bool}:
        return 1
    raise RuntimeError(f"Unsupported dtype for byte-size inference: {dtype}")


def infer_expert_intermediate_size(config: PretrainedConfig) -> int:
    arch = parse_moe_architecture(config)
    if arch == "qwen":
        value = getattr(config, "moe_intermediate_size", None)
        if value is None:
            value = getattr(config, "intermediate_size", None)
    elif arch in {"olmoe", "mixtral"}:
        value = getattr(config, "intermediate_size", None)
    elif arch in {"deepseek_v2", "deepseek_v3"}:
        value = getattr(config, "moe_intermediate_size", None)
        if value is None:
            value = getattr(config, "intermediate_size", None)
    else:
        raise RuntimeError(f"Unsupported architecture {arch}")

    if value is None or int(value) <= 0:
        raise RuntimeError(
            f"Could not infer expert intermediate size for architecture {arch}: "
            f"got {value!r} from config."
        )
    return int(value)


def infer_routed_expert_size_bytes(config: PretrainedConfig) -> int:
    """
    Estimate the routed-expert parameter footprint for one expert.

    This models the routed MLP weights only:
      - gate / w1
      - up / w3
      - down / w2

    Biases are ignored because the dominant error mode we need to fix is the
    multi-hundred-MB underestimate on packed MoE checkpoints.
    """
    config = normalize_runtime_config(config)
    hidden_size = int(getattr(config, "hidden_size"))
    intermediate_size = infer_expert_intermediate_size(config)
    bytes_per_param = dtype_nbytes(parse_expert_dtype(config))
    return 3 * hidden_size * intermediate_size * bytes_per_param


def infer_routed_expert_size_mb(config: PretrainedConfig) -> float:
    return float(infer_routed_expert_size_bytes(config) / (1024 ** 2))


def load_config_and_infer_expert_size_mb(model_path: str) -> float:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return infer_routed_expert_size_mb(config)


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

    routed = re.findall(
        rf"{_PACKED_MOE_PREFIX}\.experts\.(gate_up_proj|down_proj)$",
        param_name,
    )
    if routed:
        layer_id, tensor_role = routed[0]
        return int(layer_id), "routed_experts", tensor_role

    shared = re.findall(
        rf"{_PACKED_MOE_PREFIX}\.shared_experts\.(gate_proj|up_proj|down_proj)\.weight$",
        param_name,
    )
    if shared:
        layer_id, tensor_role = shared[0]
        return int(layer_id), "shared_experts", tensor_role

    router = re.findall(rf"{_PACKED_MOE_PREFIX}\.gate\.weight$", param_name)
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
        result = re.findall(r"(?:model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\.", param_name)
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
        else:
            return None, None
    elif parse_expert_layout(config) == "packed":
        schema = get_packed_expert_schema(config)
        expanded_names = "|".join(re.escape(name) for name in schema.expanded_names)
        routed = re.findall(
            rf"{_PACKED_MOE_PREFIX}\.experts\.(\d+)\.({expanded_names})\.weight$",
            param_name,
        )
        if routed:
            layer_id, expert_id, _ = routed[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
            if arch in {"deepseek_v2", "deepseek_v3"}:
                dense_prefix = int(getattr(config, "first_k_dense_replace", 0) or 0)
                if layer_id < dense_prefix:
                    raise RuntimeError(
                        f"Packed expert tensor {param_name!r} resolved to dense-only layer {layer_id} "
                        f"with first_k_dense_replace={dense_prefix}."
                    )
                layer_id -= dense_prefix
            return layer_id, expert_id
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
