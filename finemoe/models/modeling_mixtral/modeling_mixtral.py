import torch

from transformers.models.mixtral.modeling_mixtral import (
    MixtralExperts,
    MixtralForCausalLM,
    MixtralModel,
    MixtralPreTrainedModel,
    MixtralSparseMoeBlock,
)

from ..packed_runtime import dispatch_packed_experts, ensure_no_prefetch_runtime


class SyncMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        ensure_no_prefetch_runtime(self)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        flat_states = hidden_states.view(-1, hidden_states.shape[-1])
        _, top_k_weights, top_k_index = self.gate(flat_states)
        final_hidden_states = dispatch_packed_experts(
            hidden_states=flat_states,
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
            num_experts=self.experts.num_experts,
            layer_id=self.layer_id,
            expert_dispatcher=self.expert_dispatcher,
        )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states


__all__ = [
    "MixtralExperts",
    "MixtralForCausalLM",
    "MixtralModel",
    "MixtralPreTrainedModel",
    "MixtralSparseMoeBlock",
    "SyncMixtralSparseMoeBlock",
]
