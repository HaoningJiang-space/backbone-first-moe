from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
    DeepseekV3PreTrainedModel,
)


class SyncDeepseekV3MoE(DeepseekV3MoE):
    """Placeholder sync block for future packed-expert runtime support."""

    pass


__all__ = [
    "DeepseekV3ForCausalLM",
    "DeepseekV3MLP",
    "DeepseekV3Model",
    "DeepseekV3MoE",
    "DeepseekV3PreTrainedModel",
    "SyncDeepseekV3MoE",
]
