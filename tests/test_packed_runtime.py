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
install_runtime_device_property = PACKED_RUNTIME.install_runtime_device_property
dispatch_packed_experts = PACKED_RUNTIME.dispatch_packed_experts
_run_packed_resident_expert = PACKED_RUNTIME._run_packed_resident_expert
_infer_module_device = PACKED_RUNTIME._infer_module_device


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


class TrackingPackedDispatcher(FakePackedDispatcher):
    def __init__(self, experts, act_fn):
        super().__init__(experts, act_fn)
        self.enqueued = []
        self.wait_calls = 0

    def enqueue_expert(self, layer_id, expert_idx, gpu_id=-1, remote=False):
        self.enqueued.append((layer_id, expert_idx))
        super().enqueue_expert(layer_id, expert_idx, gpu_id, remote)

    def wait_expert(self):
        self.wait_calls += 1
        return super().wait_expert()


class _AffineExpert(torch.nn.Module):
    def __init__(self, scale: float, bias: float):
        super().__init__()
        self.scale = scale
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias


class _MixedDeviceExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.register_buffer("other_device_buffer", torch.ones(2, device="meta"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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

    def test_install_runtime_device_property_prefers_runtime_device(self):
        class _DummyBase:
            @property
            def device(self):
                return torch.device("cpu")

        class _DummyModel(_DummyBase):
            pass

        install_runtime_device_property(_DummyModel)
        model = _DummyModel()
        self.assertEqual(model.device, torch.device("cpu"))
        model._device = "cuda:0"
        self.assertEqual(model.device, torch.device("cuda:0"))

    def test_run_packed_resident_expert_supports_modulelist_container(self):
        experts = torch.nn.ModuleList(
            [
                _AffineExpert(scale=2.0, bias=1.0),
                _AffineExpert(scale=-1.0, bias=0.5),
            ]
        )
        hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        actual = _run_packed_resident_expert(experts, hidden_states, 1)
        expected = experts[1](hidden_states)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_infer_module_device_returns_none_for_mixed_devices(self):
        expert = _MixedDeviceExpert()
        self.assertIsNone(_infer_module_device(expert))

    def test_modulelist_containers_do_not_use_packed_resident_fast_path(self):
        experts = torch.nn.ModuleList(
            [
                _AffineExpert(scale=2.0, bias=1.0),
                _AffineExpert(scale=-1.0, bias=0.5),
            ]
        )
        self.assertFalse(PACKED_RUNTIME._supports_packed_resident_fast_path(experts, 0))

    def test_dispatch_packed_experts_supports_resident_fast_path(self):
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
        block = SyncMixtralSparseMoeBlock(cfg)
        self._reinitialize_parameters(block)
        hidden_states = torch.randn(5, cfg.hidden_size)
        gate_output = block.gate(hidden_states)
        if isinstance(gate_output, tuple):
            if len(gate_output) == 3:
                _, top_k_weights, top_k_index = gate_output
            else:
                top_k_weights, top_k_index = gate_output
        else:
            routing_weights = F.softmax(gate_output, dim=1, dtype=torch.float)
            top_k_weights, top_k_index = torch.topk(routing_weights, block.top_k, dim=-1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
            top_k_weights = top_k_weights.to(hidden_states.dtype)
        dispatcher = FakePackedDispatcher(block.experts, block.experts.act_fn)

        baseline = dispatch_packed_experts(
            hidden_states=hidden_states,
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
            num_experts=block.experts.num_experts,
            layer_id=0,
            expert_dispatcher=dispatcher,
            experts_module=block.experts,
            resident_expert_ids=(),
        )
        resident = dispatch_packed_experts(
            hidden_states=hidden_states,
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
            num_experts=block.experts.num_experts,
            layer_id=0,
            expert_dispatcher=FakePackedDispatcher(block.experts, block.experts.act_fn),
            experts_module=block.experts,
            resident_expert_ids={0, 1, 2, 3},
        )
        self.assertTrue(
            torch.allclose(
                torch.nan_to_num(baseline),
                torch.nan_to_num(resident),
                atol=1e-5,
                rtol=1e-4,
            )
        )

    def test_dispatch_packed_experts_only_dispatches_demand_experts(self):
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
        block = SyncMixtralSparseMoeBlock(cfg)
        self._reinitialize_parameters(block)
        hidden_states = torch.randn(5, cfg.hidden_size)
        gate_output = block.gate(hidden_states)
        if isinstance(gate_output, tuple):
            if len(gate_output) == 3:
                _, top_k_weights, top_k_index = gate_output
            else:
                top_k_weights, top_k_index = gate_output
        else:
            routing_weights = F.softmax(gate_output, dim=1, dtype=torch.float)
            top_k_weights, top_k_index = torch.topk(routing_weights, block.top_k, dim=-1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
            top_k_weights = top_k_weights.to(hidden_states.dtype)

        _, active_experts, _ = _build_packed_expert_assignments(
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
            num_experts=block.experts.num_experts,
            device=hidden_states.device,
        )
        resident_ids = {active_experts[0]}
        dispatcher = TrackingPackedDispatcher(block.experts, block.experts.act_fn)

        dispatch_packed_experts(
            hidden_states=hidden_states,
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
            num_experts=block.experts.num_experts,
            layer_id=0,
            expert_dispatcher=dispatcher,
            experts_module=block.experts,
            resident_expert_ids=resident_ids,
        )

        dispatched = {expert_idx for _, expert_idx in dispatcher.enqueued}
        self.assertNotIn(next(iter(resident_ids)), dispatched)
        self.assertEqual(dispatched, set(active_experts) - resident_ids)
        self.assertEqual(dispatcher.wait_calls, 1)


if __name__ == "__main__":
    unittest.main()
