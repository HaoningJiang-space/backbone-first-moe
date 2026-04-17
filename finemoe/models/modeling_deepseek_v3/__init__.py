from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

_import_structure = {
    "configuration_deepseek_v3": ["DeepseekV3Config"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_deepseek_v3"] = [
        "DeepseekV3ForCausalLM",
        "DeepseekV3MLP",
        "DeepseekV3Model",
        "DeepseekV3MoE",
        "DeepseekV3PreTrainedModel",
        "SyncDeepseekV3MoE",
    ]

if TYPE_CHECKING:
    from .configuration_deepseek_v3 import DeepseekV3Config
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_deepseek_v3 import (
            DeepseekV3ForCausalLM,
            DeepseekV3MLP,
            DeepseekV3Model,
            DeepseekV3MoE,
            DeepseekV3PreTrainedModel,
            SyncDeepseekV3MoE,
        )
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
