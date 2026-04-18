import torch

from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Experts,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2Moe,
    DeepseekV2PreTrainedModel,
)

from ..packed_runtime import (
    dispatch_packed_experts,
    ensure_no_prefetch_runtime,
    install_runtime_device_property,
)


install_runtime_device_property(DeepseekV2PreTrainedModel)


class SyncDeepseekV2Moe(DeepseekV2Moe):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ensure_no_prefetch_runtime(self)

        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states.to(self.gate.weight.dtype))
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        flat_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = dispatch_packed_experts(
            hidden_states=flat_states,
            top_k_index=topk_indices,
            top_k_weights=topk_weights,
            num_experts=self.config.n_routed_experts,
            layer_id=self.layer_id,
            expert_dispatcher=self.expert_dispatcher,
        ).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


__all__ = [
    "DeepseekV2Experts",
    "DeepseekV2ForCausalLM",
    "DeepseekV2MLP",
    "DeepseekV2Model",
    "DeepseekV2Moe",
    "DeepseekV2PreTrainedModel",
    "SyncDeepseekV2Moe",
]
