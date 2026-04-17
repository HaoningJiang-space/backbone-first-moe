import unittest

from transformers import MixtralConfig, DeepseekV2Config

from finemoe.models.modeling_olmoe import OlmoeConfig
from finemoe.models.modeling_qwen.configuration_qwen2_moe import Qwen2MoeConfig
from finemoe.utils import (
    parse_expert_id,
    parse_expert_layout,
    parse_moe_architecture,
    parse_moe_param,
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
        with self.assertRaisesRegex(RuntimeError, "slice-based runtime path"):
            parse_expert_id("model.layers.0.mlp.experts.gate_up_proj", cfg)

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
        with self.assertRaisesRegex(RuntimeError, "slice-based runtime path"):
            parse_expert_id("model.layers.0.mlp.experts.gate_up_proj", cfg)


if __name__ == "__main__":
    unittest.main()
