from transformers import (
    PretrainedConfig,
    MixtralForCausalLM,
    DeepseekV2ForCausalLM,
    DeepseekV3ForCausalLM,
)
from ..models.modeling_qwen.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ..models.modeling_olmoe import OlmoeForCausalLM
from ..utils import parse_moe_architecture

MODEL_MAPPING_NAMES = {
    "qwen": Qwen2MoeForCausalLM,
    "olmoe": OlmoeForCausalLM,
    "mixtral": MixtralForCausalLM,
    "deepseek_v2": DeepseekV2ForCausalLM,
    "deepseek_v3": DeepseekV3ForCausalLM,
}

MODEL_MAPPING_TYPES = {
    "qwen": 4,
    "olmoe": 4,
    "mixtral": 4,
    "deepseek_v2": 4,
    "deepseek_v3": 4,
}


def parse_expert_type(config: PretrainedConfig) -> int:
    arch = parse_moe_architecture(config)
    if arch not in MODEL_MAPPING_TYPES:
        raise RuntimeError(
            f"The `load_checkpoint_and_dispatch` function does not support the architecture {arch}. "
            f"Supported architectures are {list(MODEL_MAPPING_NAMES.keys())}."
        )
    return MODEL_MAPPING_TYPES[arch]
