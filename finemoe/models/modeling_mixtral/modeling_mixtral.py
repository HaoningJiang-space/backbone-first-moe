import torch
import torch.nn.functional as F

from transformers.models.mixtral.modeling_mixtral import (
    MixtralForCausalLM,
    MixtralModel,
    MixtralPreTrainedModel,
    MixtralSparseMoeBlock,
)

from ..packed_runtime import (
    dispatch_packed_experts,
    ensure_no_prefetch_runtime,
    install_runtime_device_property,
)


install_runtime_device_property(MixtralPreTrainedModel)


class SyncMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        ensure_no_prefetch_runtime(self)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        flat_states = hidden_states.view(-1, hidden_states.shape[-1])
        gate_output = self.gate(flat_states)
        return_router_logits = not isinstance(gate_output, tuple)
        if isinstance(gate_output, tuple):
            if len(gate_output) == 3:
                _, top_k_weights, top_k_index = gate_output
            elif len(gate_output) == 2:
                top_k_weights, top_k_index = gate_output
            else:
                raise RuntimeError(
                    f"Unexpected Mixtral gate output tuple length: {len(gate_output)}"
                )
            router_logits = None
        else:
            router_logits = gate_output
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            top_k_weights, top_k_index = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
            top_k_weights = top_k_weights.to(flat_states.dtype)

        num_experts = getattr(self, "num_experts", None)
        if num_experts is None:
            num_experts = getattr(self.experts, "num_experts", None)
        if num_experts is None:
            num_experts = len(self.experts)
        final_hidden_states = dispatch_packed_experts(
            hidden_states=flat_states,
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
            num_experts=num_experts,
            layer_id=self.layer_id,
            expert_dispatcher=self.expert_dispatcher,
            experts_module=self.experts,
            resident_fastpath_expert_ids=getattr(self, "resident_fastpath_local_expert_ids", ()),
        )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        if return_router_logits:
            return final_hidden_states, router_logits
        return final_hidden_states


__all__ = [
    "MixtralForCausalLM",
    "MixtralModel",
    "MixtralPreTrainedModel",
    "MixtralSparseMoeBlock",
    "SyncMixtralSparseMoeBlock",
]
