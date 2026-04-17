import unittest

import torch
from transformers import MixtralConfig, DeepseekV2Config

from finemoe.models.modeling_olmoe import OlmoeConfig
from finemoe.models.modeling_qwen.configuration_qwen2_moe import Qwen2MoeConfig
from finemoe.common.constants import MODEL_MAPPING_NAMES, parse_expert_type
from finemoe.utils import (
    expand_tensor_for_offload,
    parse_expert_id,
    parse_expert_layout,
    parse_moe_architecture,
    parse_moe_param,
    parse_packed_expert_tensor,
)


class HFConfigParsingTest(unittest.TestCase):
    def test_qwen_modulelist_layout(self):
        cfg = Qwen2MoeConfig(num_hidden_layers=24, num_experts=60, num_experts_per_tok=4, hidden_size=2048)
        self.assertEqual(parse_moe_architecture(cfg), "qwen")
        self.assertEqual(parse_expert_layout(cfg), "modulelist")
        self.assertEqual(parse_moe_param(cfg), (24, 60, 0, 2048, 4))
        self.assertEqual(
            parse_expert_id("layers.3.mlp.experts.17.gate_proj.weight", cfg),
            (3, 17),
        )

    def test_olmoe_modulelist_layout(self):
        cfg = OlmoeConfig(num_hidden_layers=16, num_experts=64, num_experts_per_tok=8, hidden_size=2048)
        self.assertEqual(parse_moe_architecture(cfg), "olmoe")
        self.assertEqual(parse_expert_layout(cfg), "modulelist")
        self.assertEqual(parse_moe_param(cfg), (16, 64, 0, 2048, 8))

    def test_mixtral_packed_layout(self):
        cfg = MixtralConfig(num_hidden_layers=32, num_local_experts=8, num_experts_per_tok=2, hidden_size=4096)
        self.assertEqual(parse_moe_architecture(cfg), "mixtral")
        self.assertEqual(parse_expert_layout(cfg), "packed")
        self.assertEqual(parse_moe_param(cfg), (32, 8, 0, 4096, 2))
        self.assertEqual(
            parse_packed_expert_tensor("layers.3.mlp.experts.gate_up_proj", cfg),
            (3, "routed_experts", "gate_up_proj"),
        )
        self.assertEqual(
            parse_packed_expert_tensor("layers.3.mlp.gate.weight", cfg),
            (3, "router", "gate"),
        )
        with self.assertRaisesRegex(RuntimeError, "slice-based runtime path"):
            parse_expert_id("layers.3.mlp.experts.gate_up_proj", cfg)

    def test_deepseek_v2_packed_layout(self):
        cfg = DeepseekV2Config(
            num_hidden_layers=27,
            hidden_size=2048,
            n_routed_experts=64,
            num_experts_per_tok=6,
            n_shared_experts=2,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            v_head_dim=128,
        )
        self.assertEqual(parse_moe_architecture(cfg), "deepseek_v2")
        self.assertEqual(parse_expert_layout(cfg), "packed")
        self.assertEqual(parse_moe_param(cfg), (27, 64, 0, 2048, 6))
        self.assertEqual(
            parse_packed_expert_tensor("layers.5.mlp.experts.down_proj", cfg),
            (5, "routed_experts", "down_proj"),
        )
        self.assertEqual(
            parse_packed_expert_tensor("layers.5.mlp.shared_experts.up_proj.weight", cfg),
            (5, "shared_experts", "up_proj"),
        )
        with self.assertRaisesRegex(RuntimeError, "slice-based runtime path"):
            parse_expert_id("layers.5.mlp.experts.gate_up_proj", cfg)

    def test_model_registry_includes_packed_architectures(self):
        self.assertIn("mixtral", MODEL_MAPPING_NAMES)
        self.assertIn("deepseek_v2", MODEL_MAPPING_NAMES)
        self.assertIn("deepseek_v3", MODEL_MAPPING_NAMES)

        mixtral_cfg = MixtralConfig(num_hidden_layers=2, num_local_experts=8, num_experts_per_tok=2, hidden_size=64)
        deepseek_cfg = DeepseekV2Config(
            num_hidden_layers=2, hidden_size=64, n_routed_experts=8, num_experts_per_tok=2,
            n_shared_experts=2, q_lora_rank=1536, kv_lora_rank=16, qk_rope_head_dim=8, v_head_dim=8, qk_nope_head_dim=0,
        )
        self.assertEqual(parse_expert_type(mixtral_cfg), 4)
        self.assertEqual(parse_expert_type(deepseek_cfg), 4)

    def test_mixtral_packed_tensor_expands_to_synthetic_expert_entries(self):
        cfg = MixtralConfig(num_hidden_layers=2, num_local_experts=4, num_experts_per_tok=2, hidden_size=16, intermediate_size=8)
        tensor = torch.arange(4 * 10 * 16, dtype=torch.float32).reshape(4, 10, 16)
        entries = expand_tensor_for_offload("layers.1.mlp.experts.gate_up_proj", tensor, cfg)
        self.assertEqual(len(entries), 4)
        self.assertEqual(entries[0].name, "layers.1.mlp.experts.0.gate_up_proj")
        self.assertEqual(entries[3].name, "layers.1.mlp.experts.3.gate_up_proj")
        self.assertTrue(torch.equal(entries[2].tensor, tensor[2]))
        self.assertEqual(parse_expert_id(entries[1].name, cfg), (1, 1))

    def test_deepseek_shared_tensor_is_kept_as_single_entry(self):
        cfg = DeepseekV2Config(
            num_hidden_layers=2, hidden_size=64, num_attention_heads=8, num_key_value_heads=8,
            n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=2,
            q_lora_rank=1536, kv_lora_rank=16, qk_rope_head_dim=8, v_head_dim=8, qk_nope_head_dim=0,
        )
        tensor = torch.randn(32, 64)
        entries = expand_tensor_for_offload("layers.0.mlp.shared_experts.up_proj.weight", tensor, cfg)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].name, "layers.0.mlp.shared_experts.up_proj.weight")
        self.assertIsNone(entries[0].expert_idx)


if __name__ == "__main__":
    unittest.main()
