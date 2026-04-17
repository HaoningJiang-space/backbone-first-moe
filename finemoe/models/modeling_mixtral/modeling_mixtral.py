from transformers.models.mixtral.modeling_mixtral import (
    MixtralExperts,
    MixtralForCausalLM,
    MixtralModel,
    MixtralPreTrainedModel,
    MixtralSparseMoeBlock,
)


class SyncMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    """Placeholder sync block for future packed-expert runtime support.

    The current runtime still gates packed MoE architectures before model loading.
    This class exists so the codebase has a stable local symbol to patch once the
    slice-based runtime path lands.
    """

    pass


__all__ = [
    "MixtralExperts",
    "MixtralForCausalLM",
    "MixtralModel",
    "MixtralPreTrainedModel",
    "MixtralSparseMoeBlock",
    "SyncMixtralSparseMoeBlock",
]
