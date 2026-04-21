import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "finemoe" / "models" / "modulelist_runtime.py"
SPEC = importlib.util.spec_from_file_location("modulelist_runtime", MODULE_PATH)
MODULELIST_RUNTIME = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULELIST_RUNTIME)

dispatch_modulelist_experts = MODULELIST_RUNTIME.dispatch_modulelist_experts


class _AffineExpert(torch.nn.Module):
    def __init__(self, scale: float, bias: float):
        super().__init__()
        self.scale = scale
        self.bias = bias
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return x * self.scale + self.bias


class _FakeOffloadEngine:
    def __init__(self):
        self.calls = []
        self.group_calls = []
        self.group_begins = []
        self.group_ends = []

    def run_module_demand_lane(self, module, x):
        self.calls.append(module)
        return module(x)

    def run_module_demand_lane_group(self, modules, inputs):
        self.group_calls.append(tuple(modules))
        return [module(x) for module, x in zip(modules, inputs)]

    def begin_module_group(self, modules, *, expert_blocks=0, token_assignments=0):
        self.group_begins.append((tuple(modules), expert_blocks, token_assignments))
        return {"modules": tuple(modules)}

    def run_module_group(self, service_ctx, inputs, kwargs_list=None):
        self.group_calls.append(service_ctx["modules"])
        return [module(x) for module, x in zip(service_ctx["modules"], inputs)]

    def end_module_group(self, service_ctx):
        self.group_ends.append(service_ctx["modules"])


def _reference_dispatch(hidden_states, selected_experts, routing_weights, experts):
    tokens, hidden_dim = hidden_states.shape
    num_experts = len(experts)
    final_hidden_states = torch.zeros(
        (tokens, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    expert_mask = torch.nn.functional.one_hot(
        selected_experts,
        num_classes=num_experts,
    ).permute(2, 1, 0)

    for expert_idx in range(num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        if top_x.numel() == 0:
            continue
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = experts[expert_idx](current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    return final_hidden_states


class ModulelistRuntimeTest(unittest.TestCase):
    def test_dispatch_matches_reference(self):
        hidden_states = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
            dtype=torch.float32,
        )
        selected_experts = torch.tensor(
            [
                [2, 0],
                [1, 2],
                [0, 1],
                [2, 1],
            ],
            dtype=torch.long,
        )
        routing_weights = torch.tensor(
            [
                [0.75, 0.25],
                [0.60, 0.40],
                [0.55, 0.45],
                [0.90, 0.10],
            ],
            dtype=torch.float32,
        )
        experts = torch.nn.ModuleList(
            [
                _AffineExpert(scale=1.0, bias=0.5),
                _AffineExpert(scale=-0.5, bias=1.0),
                _AffineExpert(scale=2.0, bias=-1.5),
            ]
        )

        actual = dispatch_modulelist_experts(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=experts,
        )
        expected = _reference_dispatch(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=experts,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_dispatch_handles_empty_tokens(self):
        experts = torch.nn.ModuleList([_AffineExpert(scale=1.0, bias=0.0)])
        actual = dispatch_modulelist_experts(
            hidden_states=torch.zeros((0, 4), dtype=torch.float32),
            selected_experts=torch.zeros((0, 0), dtype=torch.long),
            routing_weights=torch.zeros((0, 0), dtype=torch.float32),
            experts=experts,
        )
        self.assertEqual(tuple(actual.shape), (0, 4))

    def test_dispatch_splits_resident_and_demand_lanes(self):
        hidden_states = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=torch.float32,
        )
        selected_experts = torch.tensor(
            [
                [0, 1],
                [1, 0],
                [1, 0],
            ],
            dtype=torch.long,
        )
        routing_weights = torch.tensor(
            [
                [0.7, 0.3],
                [0.6, 0.4],
                [0.8, 0.2],
            ],
            dtype=torch.float32,
        )
        experts = torch.nn.ModuleList(
            [
                _AffineExpert(scale=2.0, bias=0.0),
                _AffineExpert(scale=1.0, bias=1.0),
            ]
        )
        reference_experts = torch.nn.ModuleList(
            [
                _AffineExpert(scale=2.0, bias=0.0),
                _AffineExpert(scale=1.0, bias=1.0),
            ]
        )
        fake_engine = _FakeOffloadEngine()
        experts[0].offload_engine = fake_engine
        experts[1].offload_engine = fake_engine

        actual = dispatch_modulelist_experts(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=experts,
            resident_expert_ids={1},
        )
        expected = _reference_dispatch(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=reference_experts,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertEqual(experts[0].forward_calls, 1)
        self.assertEqual(experts[1].forward_calls, 1)
        self.assertEqual(fake_engine.calls, [])
        self.assertEqual(fake_engine.group_calls, [(experts[0],)])
        self.assertEqual(fake_engine.group_begins, [((experts[0],), 1, 3)])
        self.assertEqual(fake_engine.group_ends, [(experts[0],)])

    def test_dispatch_groups_multiple_demand_experts(self):
        hidden_states = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=torch.float32,
        )
        selected_experts = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 0],
            ],
            dtype=torch.long,
        )
        routing_weights = torch.tensor(
            [
                [0.7, 0.3],
                [0.6, 0.4],
                [0.8, 0.2],
            ],
            dtype=torch.float32,
        )
        experts = torch.nn.ModuleList(
            [
                _AffineExpert(scale=2.0, bias=0.0),
                _AffineExpert(scale=1.0, bias=1.0),
                _AffineExpert(scale=-1.0, bias=0.5),
            ]
        )
        reference_experts = torch.nn.ModuleList(
            [
                _AffineExpert(scale=2.0, bias=0.0),
                _AffineExpert(scale=1.0, bias=1.0),
                _AffineExpert(scale=-1.0, bias=0.5),
            ]
        )
        fake_engine = _FakeOffloadEngine()
        for expert in experts:
            expert.offload_engine = fake_engine

        actual = dispatch_modulelist_experts(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=experts,
            resident_expert_ids={2},
        )
        expected = _reference_dispatch(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=reference_experts,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertEqual(fake_engine.calls, [])
        self.assertEqual(fake_engine.group_calls, [(experts[0], experts[1])])
        self.assertEqual(fake_engine.group_begins, [((experts[0], experts[1]), 2, 4)])
        self.assertEqual(fake_engine.group_ends, [(experts[0], experts[1])])

    def test_dispatch_can_disable_backbone_lane_split_while_keeping_resident_ids(self):
        hidden_states = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=torch.float32,
        )
        selected_experts = torch.tensor(
            [
                [0, 1],
                [1, 0],
                [1, 0],
            ],
            dtype=torch.long,
        )
        routing_weights = torch.tensor(
            [
                [0.7, 0.3],
                [0.6, 0.4],
                [0.8, 0.2],
            ],
            dtype=torch.float32,
        )
        experts = torch.nn.ModuleList(
            [
                _AffineExpert(scale=2.0, bias=0.0),
                _AffineExpert(scale=1.0, bias=1.0),
            ]
        )
        reference_experts = torch.nn.ModuleList(
            [
                _AffineExpert(scale=2.0, bias=0.0),
                _AffineExpert(scale=1.0, bias=1.0),
            ]
        )
        fake_engine = _FakeOffloadEngine()
        for expert in experts:
            expert.offload_engine = fake_engine

        actual = dispatch_modulelist_experts(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=experts,
            resident_expert_ids={1},
            enable_backbone_lane_split=False,
        )
        expected = _reference_dispatch(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=reference_experts,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertEqual(experts[0].forward_calls, 1)
        self.assertEqual(experts[1].forward_calls, 1)
        self.assertEqual(fake_engine.calls, [])
        self.assertEqual(fake_engine.group_calls, [(experts[0], experts[1])])
        self.assertEqual(fake_engine.group_begins, [((experts[0], experts[1]), 2, 6)])
        self.assertEqual(fake_engine.group_ends, [(experts[0], experts[1])])

    def test_dispatch_records_assignment_gather_and_merge_breakdown(self):
        hidden_states = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=torch.float32,
        )
        selected_experts = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 0],
            ],
            dtype=torch.long,
        )
        routing_weights = torch.tensor(
            [
                [0.7, 0.3],
                [0.6, 0.4],
                [0.8, 0.2],
            ],
            dtype=torch.float32,
        )
        experts = torch.nn.ModuleList(
            [
                _AffineExpert(scale=2.0, bias=0.0),
                _AffineExpert(scale=1.0, bias=1.0),
                _AffineExpert(scale=-1.0, bias=0.5),
            ]
        )
        fake_engine = _FakeOffloadEngine()
        for expert in experts:
            expert.offload_engine = fake_engine

        calls = []
        runtime_profile = SimpleNamespace(
            record_modulelist_dispatch=lambda **kwargs: calls.append(kwargs)
        )

        dispatch_modulelist_experts(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=experts,
            resident_expert_ids={2},
            runtime_profile=runtime_profile,
        )

        self.assertEqual(len(calls), 1)
        payload = calls[0]
        self.assertGreaterEqual(payload["assignment_build_wall_time_sec"], 0.0)
        self.assertGreaterEqual(payload["resident_gather_wall_time_sec"], 0.0)
        self.assertGreaterEqual(payload["demand_gather_wall_time_sec"], 0.0)
        self.assertGreaterEqual(payload["resident_merge_wall_time_sec"], 0.0)
        self.assertGreaterEqual(payload["demand_merge_wall_time_sec"], 0.0)
        self.assertEqual(payload["resident_expert_blocks"], 1)
        self.assertEqual(payload["demand_expert_blocks"], 2)

    def test_dispatch_groups_multiple_resident_experts(self):
        hidden_states = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=torch.float32,
        )
        selected_experts = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 0],
            ],
            dtype=torch.long,
        )
        routing_weights = torch.tensor(
            [
                [0.7, 0.3],
                [0.6, 0.4],
                [0.8, 0.2],
            ],
            dtype=torch.float32,
        )
        experts = torch.nn.ModuleList(
            [
                _AffineExpert(scale=2.0, bias=0.0),
                _AffineExpert(scale=1.0, bias=1.0),
                _AffineExpert(scale=-1.0, bias=0.5),
            ]
        )
        reference_experts = torch.nn.ModuleList(
            [
                _AffineExpert(scale=2.0, bias=0.0),
                _AffineExpert(scale=1.0, bias=1.0),
                _AffineExpert(scale=-1.0, bias=0.5),
            ]
        )
        fake_engine = _FakeOffloadEngine()
        for expert in experts:
            expert.offload_engine = fake_engine

        actual = dispatch_modulelist_experts(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=experts,
            resident_expert_ids={1, 2},
        )
        expected = _reference_dispatch(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=reference_experts,
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertEqual(fake_engine.group_calls, [(experts[0],)])
        self.assertEqual(fake_engine.group_begins, [((experts[0],), 1, 2)])
        self.assertEqual(fake_engine.group_ends, [(experts[0],)])
        self.assertEqual(experts[1].forward_calls, 1)
        self.assertEqual(experts[2].forward_calls, 1)

    def test_dispatch_reuses_output_buffer_with_runtime_cache(self):
        hidden_states = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=torch.float32,
        )
        selected_experts = torch.tensor(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=torch.long,
        )
        routing_weights = torch.tensor(
            [
                [0.6, 0.4],
                [0.7, 0.3],
            ],
            dtype=torch.float32,
        )
        experts = torch.nn.ModuleList(
            [
                _AffineExpert(scale=2.0, bias=0.0),
                _AffineExpert(scale=1.0, bias=1.0),
            ]
        )
        runtime_cache = {}
        calls = []
        runtime_profile = SimpleNamespace(
            record_modulelist_dispatch=lambda **kwargs: calls.append(kwargs)
        )

        first = dispatch_modulelist_experts(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=experts,
            runtime_profile=runtime_profile,
            runtime_cache=runtime_cache,
        )
        first_ptr = first.data_ptr()
        second = dispatch_modulelist_experts(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=experts,
            runtime_profile=runtime_profile,
            runtime_cache=runtime_cache,
        )

        self.assertEqual(first_ptr, second.data_ptr())
        self.assertEqual(calls[0]["output_buffer_cache_hit"], 0)
        self.assertEqual(calls[0]["output_buffer_cache_miss"], 1)
        self.assertEqual(calls[1]["output_buffer_cache_hit"], 1)
        self.assertEqual(calls[1]["output_buffer_cache_miss"], 0)
        self.assertGreaterEqual(calls[0]["output_buffer_prepare_wall_time_sec"], 0.0)
        self.assertGreaterEqual(calls[1]["output_buffer_prepare_wall_time_sec"], 0.0)

    def test_dispatch_backbone_grouped_resident_mode_reuses_workspace(self):
        hidden_states = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=torch.float32,
        )
        selected_experts = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 0],
            ],
            dtype=torch.long,
        )
        routing_weights = torch.tensor(
            [
                [0.7, 0.3],
                [0.6, 0.4],
                [0.8, 0.2],
            ],
            dtype=torch.float32,
        )
        experts = torch.nn.ModuleList(
            [
                _AffineExpert(scale=2.0, bias=0.0),
                _AffineExpert(scale=1.0, bias=1.0),
                _AffineExpert(scale=-1.0, bias=0.5),
            ]
        )
        reference_experts = torch.nn.ModuleList(
            [
                _AffineExpert(scale=2.0, bias=0.0),
                _AffineExpert(scale=1.0, bias=1.0),
                _AffineExpert(scale=-1.0, bias=0.5),
            ]
        )
        fake_engine = _FakeOffloadEngine()
        for expert in experts:
            expert.offload_engine = fake_engine

        runtime_cache = {}
        calls = []
        runtime_profile = SimpleNamespace(
            record_modulelist_dispatch=lambda **kwargs: calls.append(kwargs)
        )

        first = dispatch_modulelist_experts(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=experts,
            resident_expert_ids={1, 2},
            enable_backbone_lane_split=False,
            backbone_grouped_resident_mode=True,
            runtime_profile=runtime_profile,
            runtime_cache=runtime_cache,
        )
        second = dispatch_modulelist_experts(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=experts,
            resident_expert_ids={1, 2},
            enable_backbone_lane_split=False,
            backbone_grouped_resident_mode=True,
            runtime_profile=runtime_profile,
            runtime_cache=runtime_cache,
        )
        expected = _reference_dispatch(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            experts=reference_experts,
        )

        self.assertTrue(torch.allclose(first, expected, atol=1e-6))
        self.assertTrue(torch.allclose(second, expected, atol=1e-6))
        self.assertEqual(fake_engine.group_calls, [(experts[0],), (experts[0],)])
        self.assertEqual(calls[0]["resident_workspace_cache_hit"], 0)
        self.assertEqual(calls[0]["resident_workspace_cache_miss"], 1)
        self.assertEqual(calls[1]["resident_workspace_cache_hit"], 1)
        self.assertEqual(calls[1]["resident_workspace_cache_miss"], 0)
        self.assertGreaterEqual(calls[0]["resident_workspace_prepare_wall_time_sec"], 0.0)
        self.assertGreaterEqual(calls[1]["resident_workspace_prepare_wall_time_sec"], 0.0)


if __name__ == "__main__":
    unittest.main()
