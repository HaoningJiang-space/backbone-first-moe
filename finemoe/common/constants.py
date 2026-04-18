from transformers import AutoConfig, PretrainedConfig

from ..models.modeling_qwen.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ..models.modeling_olmoe import OlmoeForCausalLM
from ..utils import parse_moe_architecture

MODEL_MAPPING_NAMES = {
    "qwen": Qwen2MoeForCausalLM,
    "olmoe": OlmoeForCausalLM,
}

try:
    from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

    MODEL_MAPPING_NAMES["mixtral"] = MixtralForCausalLM
except ImportError:
    pass

try:
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2ForCausalLM

    MODEL_MAPPING_NAMES["deepseek_v2"] = DeepseekV2ForCausalLM
except ImportError:
    pass

try:
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM

    MODEL_MAPPING_NAMES["deepseek_v3"] = DeepseekV3ForCausalLM
except ImportError:
    pass

MODEL_MAPPING_TYPES = {arch: 4 for arch in MODEL_MAPPING_NAMES}


def _register_local_auto_configs():
    registrations = []
    try:
        from ..models.modeling_deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config

        registrations.append(("deepseek_v2", DeepseekV2Config))
    except ImportError:
        pass

    try:
        from ..models.modeling_deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

        registrations.append(("deepseek_v3", DeepseekV3Config))
    except ImportError:
        pass

    for model_type, config_cls in registrations:
        try:
            AutoConfig.register(model_type, config_cls, exist_ok=True)
        except TypeError:
            try:
                AutoConfig.register(model_type, config_cls)
            except ValueError:
                pass
        except ValueError:
            pass


_register_local_auto_configs()


def parse_expert_type(config: PretrainedConfig) -> int:
    arch = parse_moe_architecture(config)
    if arch not in MODEL_MAPPING_TYPES:
        raise RuntimeError(
            f"The `load_checkpoint_and_dispatch` function does not support the architecture {arch}. "
            f"Supported architectures are {list(MODEL_MAPPING_NAMES.keys())}."
        )
    return MODEL_MAPPING_TYPES[arch]
