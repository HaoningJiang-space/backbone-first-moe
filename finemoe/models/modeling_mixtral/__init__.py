from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

_import_structure = {
    "configuration_mixtral": ["MixtralConfig"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_mixtral"] = [
        "MixtralExperts",
        "MixtralForCausalLM",
        "MixtralModel",
        "MixtralPreTrainedModel",
        "MixtralSparseMoeBlock",
        "SyncMixtralSparseMoeBlock",
    ]

if TYPE_CHECKING:
    from .configuration_mixtral import MixtralConfig
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_mixtral import (
            MixtralExperts,
            MixtralForCausalLM,
            MixtralModel,
            MixtralPreTrainedModel,
            MixtralSparseMoeBlock,
            SyncMixtralSparseMoeBlock,
        )
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
