import unittest
import importlib.util
from pathlib import Path

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

    def run_module_demand_lane(self, module, x):
        self.calls.append(module)
        return module(x)

    def run_module_demand_lane_group(self, modules, inputs):
        self.group_calls.append(tuple(modules))
        return [module(x) for module, x in zip(modules, inputs)]


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


if __name__ == "__main__":
    unittest.main()
