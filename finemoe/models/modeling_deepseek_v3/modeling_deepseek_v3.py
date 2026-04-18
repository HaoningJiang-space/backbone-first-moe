import torch

from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
    DeepseekV3NaiveMoe,
    DeepseekV3PreTrainedModel,
)

from ..packed_runtime import (
    dispatch_packed_experts,
    ensure_no_prefetch_runtime,
    install_runtime_device_property,
)


install_runtime_device_property(DeepseekV3PreTrainedModel)


class SyncDeepseekV3MoE(DeepseekV3MoE):
    def forward(self, hidden_states):
        ensure_no_prefetch_runtime(self)

        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        router_scores = router_logits.sigmoid()
        correction_bias = self.gate.e_score_correction_bias.to(
            device=router_scores.device,
            dtype=router_scores.dtype,
        )
        router_logits_for_choice = router_scores + correction_bias
        group_scores = (
            router_logits_for_choice.view(
                -1, self.n_group, self.n_routed_experts // self.n_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        hidden_states = dispatch_packed_experts(
            hidden_states=hidden_states.view(-1, hidden_states.shape[-1]),
            top_k_index=topk_indices,
            top_k_weights=topk_weights,
            num_experts=self.n_routed_experts,
            layer_id=self.layer_id,
            expert_dispatcher=self.expert_dispatcher,
            experts_module=self.experts,
            resident_expert_ids=getattr(self, "resident_local_expert_ids", ()),
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
