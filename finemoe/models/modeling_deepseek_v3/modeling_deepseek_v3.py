import torch

from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
    DeepseekV3NaiveMoe,
    DeepseekV3PreTrainedModel,
)

from ..packed_runtime import dispatch_packed_experts, ensure_no_prefetch_runtime


class SyncDeepseekV3MoE(DeepseekV3MoE):
    def forward(self, hidden_states):
        ensure_no_prefetch_runtime(self)

        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        flat_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = dispatch_packed_experts(
            hidden_states=flat_states,
            top_k_index=topk_indices,
            top_k_weights=topk_weights,
            num_experts=self.n_routed_experts,
            layer_id=self.layer_id,
            expert_dispatcher=self.expert_dispatcher,
        ).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


__all__ = [
    "DeepseekV3ForCausalLM",
    "DeepseekV3MLP",
    "DeepseekV3Model",
    "DeepseekV3MoE",
    "DeepseekV3NaiveMoe",
    "DeepseekV3PreTrainedModel",
    "SyncDeepseekV3MoE",
]
