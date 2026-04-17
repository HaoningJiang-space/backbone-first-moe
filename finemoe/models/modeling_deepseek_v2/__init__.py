from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

_import_structure = {
    "configuration_deepseek_v2": ["DeepseekV2Config"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_deepseek_v2"] = [
        "DeepseekV2Experts",
        "DeepseekV2ForCausalLM",
        "DeepseekV2MLP",
        "DeepseekV2Model",
        "DeepseekV2Moe",
        "DeepseekV2PreTrainedModel",
        "SyncDeepseekV2Moe",
    ]

if TYPE_CHECKING:
    from .configuration_deepseek_v2 import DeepseekV2Config
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_deepseek_v2 import (
            DeepseekV2Experts,
            DeepseekV2ForCausalLM,
            DeepseekV2MLP,
            DeepseekV2Model,
            DeepseekV2Moe,
            DeepseekV2PreTrainedModel,
            SyncDeepseekV2Moe,
        )
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
