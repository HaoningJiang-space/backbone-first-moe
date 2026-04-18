import unittest
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.models.mixtral.modeling_mixtral import MixtralConfig, MixtralSparseMoeBlock
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Config, DeepseekV2Moe
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Config, DeepseekV3MoE

from finemoe.models.modeling_mixtral.modeling_mixtral import SyncMixtralSparseMoeBlock
from finemoe.models.modeling_deepseek_v2.modeling_deepseek_v2 import SyncDeepseekV2Moe
from finemoe.models.modeling_deepseek_v3.modeling_deepseek_v3 import SyncDeepseekV3MoE

PACKED_RUNTIME_PATH = Path(__file__).resolve().parents[1] / "finemoe" / "models" / "packed_runtime.py"
PACKED_SPEC = importlib.util.spec_from_file_location("packed_runtime_local", PACKED_RUNTIME_PATH)
PACKED_RUNTIME = importlib.util.module_from_spec(PACKED_SPEC)
assert PACKED_SPEC.loader is not None
PACKED_SPEC.loader.exec_module(PACKED_RUNTIME)

_build_packed_expert_assignments = PACKED_RUNTIME._build_packed_expert_assignments


class FakePackedDispatcher:
    def __init__(self, experts, act_fn):
        self.experts = experts
        self.act_fn = act_fn
        self.hidden_states = None
        self.router_mask = None
        self.queue = []

    def set_inputs(self, hidden_states, router_mask):
        self.hidden_states = hidden_states
        self.router_mask = router_mask

    def set_expected_queue(self, n):
        return None

    def enqueue_expert(self, layer_id, expert_idx, gpu_id=-1, remote=False):
        self.queue.append((layer_id, expert_idx))

    def wait_expert(self):
        results = []
        for layer_id, expert_idx in self.queue:
            token_idx = torch.where(self.router_mask[:, expert_idx])[0]
            current_state = self.hidden_states[token_idx]
            gate, up = F.linear(
                current_state, self.experts.gate_up_proj[expert_idx]
            ).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(
                current_hidden_states, self.experts.down_proj[expert_idx]
            )
            results.append((current_hidden_states, layer_id, expert_idx, 0))
        self.queue = []
        return results


class PackedRuntimeForwardTest(unittest.TestCase):
    @staticmethod
    def _reinitialize_parameters(module):
        with torch.no_grad():
            for param in module.parameters():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

    def _assert_runtime_equivalent(self, ref_block, sync_block, hidden_states, atol=1e-5):
        self._reinitialize_parameters(ref_block)
        sync_block.load_state_dict(ref_block.state_dict())
        sync_block.eval()
        ref_block.eval()
        sync_block.layer_id = 0
        sync_block.batch_prefetch_mode = False
        sync_block.expert_map_matcher = None
        sync_block.expert_dispatcher = FakePackedDispatcher(
            sync_block.experts, sync_block.experts.act_fn
        )
        with torch.no_grad():
            ref_out = ref_block(hidden_states)
            sync_out = sync_block(hidden_states)
        self.assertTrue(torch.allclose(sync_out, ref_out, atol=atol, rtol=1e-4))

    def test_mixtral_sync_block_matches_reference(self):
        torch.manual_seed(0)
        cfg = MixtralConfig(
            num_hidden_layers=1,
            hidden_size=32,
            intermediate_size=16,
            num_local_experts=4,
            num_experts_per_tok=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        ref_block = MixtralSparseMoeBlock(cfg)
        sync_block = SyncMixtralSparseMoeBlock(cfg)
        hidden_states = torch.randn(2, 3, cfg.hidden_size)
        self._assert_runtime_equivalent(ref_block, sync_block, hidden_states)

    def test_deepseek_v2_sync_block_matches_reference(self):
        torch.manual_seed(0)
        cfg = DeepseekV2Config(
            num_hidden_layers=1,
            hidden_size=32,
            intermediate_size=16,
            moe_intermediate_size=8,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            q_lora_rank=1536,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=0,
            v_head_dim=8,
        )
        ref_block = DeepseekV2Moe(cfg)
        sync_block = SyncDeepseekV2Moe(cfg)
        hidden_states = torch.randn(2, 3, cfg.hidden_size)
        self._assert_runtime_equivalent(ref_block, sync_block, hidden_states)

    def test_deepseek_v3_sync_block_matches_reference(self):
        torch.manual_seed(0)
        cfg = DeepseekV3Config(
            num_hidden_layers=1,
            hidden_size=32,
            intermediate_size=16,
            moe_intermediate_size=8,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            q_lora_rank=1536,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=0,
            v_head_dim=8,
            n_group=1,
            topk_group=1,
        )
        ref_block = DeepseekV3MoE(cfg)
        sync_block = SyncDeepseekV3MoE(cfg)
        hidden_states = torch.randn(2, 3, cfg.hidden_size)
        self._assert_runtime_equivalent(ref_block, sync_block, hidden_states)

    def test_build_packed_expert_assignments_groups_tokens(self):
        top_k_index = torch.tensor([[2, 0], [1, 2], [2, 1]], dtype=torch.long)
        top_k_weights = torch.tensor(
            [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]],
            dtype=torch.float32,
        )
        router_mask, active_experts, assignment_map = _build_packed_expert_assignments(
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
            num_experts=4,
            device=torch.device("cpu"),
        )

        self.assertEqual(active_experts, [0, 1, 2])
        self.assertTrue(router_mask[0, 0])
        self.assertTrue(router_mask[0, 2])
        token_idx, weights = assignment_map[2]
        self.assertTrue(torch.equal(token_idx, torch.tensor([0, 1, 2])))
        self.assertTrue(torch.allclose(weights, torch.tensor([0.7, 0.4, 0.8])))


if __name__ == "__main__":
    unittest.main()
