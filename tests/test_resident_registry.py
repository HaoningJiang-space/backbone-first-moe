import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from finemoe.models.modeling_qwen.configuration_qwen2_moe import Qwen2MoeConfig

MODULE_PATH = Path(__file__).resolve().parents[1] / "finemoe" / "runtime" / "model_offload.py"
SPEC = importlib.util.spec_from_file_location("test_model_offload", MODULE_PATH)
MODEL_OFFLOAD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODEL_OFFLOAD)

OffloadEngine = MODEL_OFFLOAD.OffloadEngine


class ResidentRegistryTest(unittest.TestCase):
    def _build_engine_stub(self):
        engine = object.__new__(OffloadEngine)
        engine.config = Qwen2MoeConfig(
            num_hidden_layers=4,
            num_experts=8,
            num_experts_per_tok=2,
            hidden_size=256,
        )
        engine.archer_config = SimpleNamespace(resident_expert_ids_file="")
        engine.resident_expert_ids = []
        engine.resident_expert_ids_set = set()
        OffloadEngine._reset_resident_registry(engine)
        return engine

    @staticmethod
    def _fake_archer_engine(node_map, sparse_cache_limit=0):
        class _FakeArcherEngine:
            def __init__(self, mapping, sparse_cache_limit=0):
                self.mapping = mapping
                self.sparse_cache_limit = sparse_cache_limit

            def get_node_id(self, tensor_ids):
                return self.mapping[int(tensor_ids[0])][0]

            def get_node_byte_size(self, tensor_ids):
                return self.mapping[int(tensor_ids[0])][1]

            def get_sparse_cache_limit(self, device):
                return self.sparse_cache_limit

        return _FakeArcherEngine(node_map, sparse_cache_limit=sparse_cache_limit)

    def test_load_resident_ids_preserves_order_and_records_registry(self):
        engine = self._build_engine_stub()
        with tempfile.TemporaryDirectory() as tmpdir:
            resident_file = Path(tmpdir) / "resident.json"
            resident_file.write_text(
                json.dumps(
                    {
                        "selection_rule": "frontier_prefix",
                        "selection_budget_bytes": 1024,
                        "selection_budget_source": "runtime_sparse_budget_bytes",
                        "resident_set": [
                            [0, 2],
                            {"layer": 1, "expert": 3},
                            [0, 2],
                            [2, 1],
                        ],
                    }
                )
            )
            engine.archer_config.resident_expert_ids_file = str(resident_file)
            resident_ids = OffloadEngine._load_resident_expert_ids(engine)

        self.assertEqual(resident_ids, [(0, 2), (1, 3), (2, 1)])
        registry = OffloadEngine.get_resident_registry(engine)
        self.assertTrue(registry["enabled"])
        self.assertEqual(registry["selection_rule"], "frontier_prefix")
        self.assertEqual(registry["requested_count"], 3)
        self.assertEqual(registry["admitted_count"], 0)
        self.assertFalse(registry["clipped"])
        self.assertEqual(engine._resident_budget_override_bytes, 1024)
        self.assertEqual(engine._resident_budget_override_source, "runtime_sparse_budget_bytes")

    def test_activate_registry_tracks_requested_vs_admitted_counts(self):
        engine = self._build_engine_stub()
        engine.archer_engine = self._fake_archer_engine(
            {
                10: (100, 64),
                11: (101, 128),
                12: (101, 128),
                13: (102, 256),
            }
        )
        OffloadEngine._record_requested_residents(
            engine,
            resident_file="/tmp/resident.json",
            resident_expert_ids=[(0, 1), (0, 2), (1, 0)],
            selection_rule="frontier_prefix",
        )
        OffloadEngine._activate_resident_registry(
            engine,
            resident_expert_ids=[(0, 1), (1, 0)],
            node_ids=[10, 11, 12, 13],
        )

        registry = OffloadEngine.get_resident_registry(engine)
        self.assertEqual(registry["requested_count"], 3)
        self.assertEqual(registry["admitted_count"], 2)
        self.assertEqual(registry["requested_node_count"], 3)
        self.assertEqual(registry["admitted_node_count"], 3)
        self.assertEqual(registry["admitted_tensor_count"], 4)
        self.assertEqual(registry["requested_bytes"], 448)
        self.assertEqual(registry["admitted_bytes"], 448)
        self.assertEqual(registry["budget_bytes"], 0)
        self.assertEqual(registry["budget_source"], "")
        self.assertTrue(registry["clipped"])

    def test_collect_unique_node_stats_deduplicates_packed_tensor_ids(self):
        engine = self._build_engine_stub()
        engine.archer_engine = self._fake_archer_engine(
            {
                20: (200, 512),
                21: (200, 512),
                22: (201, 256),
            }
        )

        node_ids, total_bytes = OffloadEngine._collect_unique_node_stats(engine, [20, 21, 22])
        self.assertEqual(node_ids, [200, 201])
        self.assertEqual(total_bytes, 768)

    def test_mark_module_resident_fastpath_marks_subtree(self):
        engine = self._build_engine_stub()
        module = nn.Sequential(nn.Linear(4, 4), nn.Sequential(nn.Linear(4, 4)))
        marked = OffloadEngine._mark_module_resident_fastpath(engine, module)
        self.assertEqual(marked, 4)
        self.assertTrue(getattr(module, "_archer_resident_fastpath"))
        self.assertTrue(getattr(module[0], "_archer_resident_fastpath"))
        self.assertTrue(getattr(module[1], "_archer_resident_fastpath"))
        self.assertTrue(getattr(module[1][0], "_archer_resident_fastpath"))

    def test_run_module_demand_lane_cleans_up_partial_begin_failures(self):
        engine = self._build_engine_stub()
        engine.runtime_profile = MODEL_OFFLOAD.RuntimeProfile()
        engine.device = "cpu"
        engine.request_id = 7
        engine.offload_set = set()

        class _RaisingArcherEngine:
            def __init__(self):
                self.begin_calls = 0
                self.end_calls = 0

            def begin(self, request_id, tensor):
                self.begin_calls += 1
                if self.begin_calls == 2:
                    raise RuntimeError("boom")

            def end(self, request_id, tensor):
                self.end_calls += 1

        engine.archer_engine = _RaisingArcherEngine()
        module = nn.Linear(4, 4)
        engine.offload_set = {param.data_ptr() for param in module.parameters()}

        with self.assertRaisesRegex(RuntimeError, "boom"):
            OffloadEngine.run_module_demand_lane(engine, module, torch.zeros(1, 4))

        self.assertEqual(engine.archer_engine.begin_calls, 2)
        self.assertEqual(engine.archer_engine.end_calls, 1)
        self.assertEqual(
            engine.offload_set,
            {param.data_ptr() for param in module.parameters()},
        )
        self.assertFalse(getattr(module, "_archer_manual_service_active", True))

    def test_begin_module_subtree_moves_non_offloaded_buffers(self):
        engine = self._build_engine_stub()
        engine.runtime_profile = MODEL_OFFLOAD.RuntimeProfile()
        engine.device = "cpu"
        engine.request_id = 1
        engine.offload_set = set()

        class _BufferOnlyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.ones(2, dtype=torch.float32))

        module = _BufferOnlyModule()
        with mock.patch.object(
            OffloadEngine,
            "_move_tensor_to_service_device",
            autospec=True,
        ) as move_mock:
            begun = OffloadEngine._begin_module_subtree(engine, module)

        self.assertEqual(begun, ())
        move_mock.assert_called_once_with(engine, module.cache)

    def test_activate_registry_assigns_explicit_packed_fastpath_ids(self):
        engine = self._build_engine_stub()
        engine.config.model_type = "mixtral"
        engine.device = "cpu"

        class _PackedExperts:
            def __init__(self):
                self.gate_up_proj = torch.nn.Parameter(torch.randn(2, 4, 4))
                self.down_proj = torch.nn.Parameter(torch.randn(2, 2, 2))

        packed_module = SimpleNamespace(layer_id=0, experts=_PackedExperts())
        modulelist_module = SimpleNamespace(layer_id=1, experts=nn.ModuleList([nn.Linear(2, 2)]))
        engine.expert_layer_modules = [packed_module, modulelist_module]

        OffloadEngine._record_requested_residents(
            engine,
            resident_file="/tmp/resident.json",
            resident_expert_ids=[(0, 2), (0, 3), (1, 0)],
            selection_rule="frontier_prefix",
        )
        OffloadEngine._activate_resident_registry(
            engine,
            resident_expert_ids=[(0, 2), (0, 3), (1, 0)],
            node_ids=[10, 11, 12],
        )

        self.assertEqual(packed_module.resident_local_expert_ids, {2, 3})
        self.assertEqual(packed_module.resident_fastpath_local_expert_ids, {2, 3})
        self.assertEqual(modulelist_module.resident_local_expert_ids, {0})
        self.assertEqual(modulelist_module.resident_fastpath_local_expert_ids, set())
        registry = OffloadEngine.get_resident_registry(engine)
        self.assertEqual(registry["fast_path_expert_count"], 2)

    def test_activate_registry_disables_packed_fastpath_when_weights_stay_on_cpu(self):
        engine = self._build_engine_stub()
        engine.config.model_type = "mixtral"
        engine.device = "cuda:0"

        class _PackedExperts:
            def __init__(self):
                self.gate_up_proj = torch.nn.Parameter(torch.randn(2, 4, 4))
                self.down_proj = torch.nn.Parameter(torch.randn(2, 2, 2))

        packed_module = SimpleNamespace(layer_id=0, experts=_PackedExperts())
        engine.expert_layer_modules = [packed_module]

        OffloadEngine._record_requested_residents(
            engine,
            resident_file="/tmp/resident.json",
            resident_expert_ids=[(0, 0), (0, 1)],
            selection_rule="frontier_prefix",
        )
        OffloadEngine._activate_resident_registry(
            engine,
            resident_expert_ids=[(0, 0), (0, 1)],
            node_ids=[10, 11],
        )

        self.assertEqual(packed_module.resident_local_expert_ids, {0, 1})
        self.assertEqual(packed_module.resident_fastpath_local_expert_ids, set())
        registry = OffloadEngine.get_resident_registry(engine)
        self.assertEqual(registry["fast_path_expert_count"], 0)

    def test_clip_resident_prefix_respects_sparse_budget(self):
        engine = self._build_engine_stub()
        engine.device = "cuda:0"
        engine.archer_engine = self._fake_archer_engine(
            {
                10: (100, 128),
                11: (101, 128),
                12: (102, 128),
            },
            sparse_cache_limit=256,
        )

        admitted_experts, admitted_tensors = OffloadEngine._clip_resident_prefix_to_sparse_budget(
            engine,
            resident_expert_ids=[(0, 0), (0, 1), (0, 2)],
            expert_tensor_ids=[[10], [11], [12]],
        )

        self.assertEqual(admitted_experts, [(0, 0), (0, 1)])
        self.assertEqual(admitted_tensors, [10, 11])
        self.assertEqual(engine.resident_registry.budget_bytes, 256)
        self.assertEqual(engine.resident_registry.budget_source, "free_device_memory_ratio")

    def test_sparse_budget_info_is_available_without_resident_admission(self):
        engine = self._build_engine_stub()
        engine.device = "cuda:0"
        engine.archer_engine = self._fake_archer_engine({}, sparse_cache_limit=4096)

        budget = OffloadEngine.get_sparse_budget_info(engine)

        self.assertEqual(budget["budget_bytes"], 4096)
        self.assertEqual(budget["budget_source"], "free_device_memory_ratio")

    def test_sparse_budget_info_respects_resident_budget_override(self):
        engine = self._build_engine_stub()
        engine.device = "cuda:0"
        engine.archer_engine = self._fake_archer_engine({}, sparse_cache_limit=4096)
        engine._resident_budget_override_bytes = 1024
        engine._resident_budget_override_source = "runtime_sparse_budget_bytes"

        budget = OffloadEngine.get_sparse_budget_info(engine)

        self.assertEqual(budget["budget_bytes"], 1024)
        self.assertEqual(
            budget["budget_source"],
            "min(free_device_memory_ratio,runtime_sparse_budget_bytes)",
        )

    def test_runtime_profile_exposes_recorded_counters(self):
        engine = self._build_engine_stub()
        engine.runtime_profile = MODEL_OFFLOAD.RuntimeProfile()
        engine.runtime_profile.record_module_io(
            begin_calls=2,
            param_begin_calls=3,
            begin_wall_time_sec=0.25,
            skipped_fastpath=True,
        )
        engine.runtime_profile.record_module_io(skipped_manual_service=True)
        engine.runtime_profile.record_manual_subtree_service(
            begin_calls=1,
            end_calls=1,
            begin_wall_time_sec=0.1,
            end_wall_time_sec=0.2,
        )
        engine.runtime_profile.record_modulelist_dispatch(
            active_expert_blocks=4,
            resident_expert_blocks=1,
            demand_expert_blocks=3,
            token_assignments=12,
            resident_token_assignments=2,
            demand_token_assignments=10,
            expert_compute_wall_time_sec=0.5,
            resident_compute_wall_time_sec=0.2,
            demand_compute_wall_time_sec=0.3,
        )
        engine.runtime_profile.record_packed_dispatch(
            resident_expert_blocks=2,
            demand_expert_blocks=5,
            resident_token_assignments=6,
            demand_token_assignments=14,
            resident_compute_wall_time_sec=0.75,
            dispatch_wait_calls=1,
            dispatch_wait_wall_time_sec=1.25,
        )

        payload = OffloadEngine.get_runtime_profile(engine)
        self.assertEqual(payload["module_begin_calls"], 2)
        self.assertEqual(payload["param_begin_calls"], 3)
        self.assertEqual(payload["resident_fastpath_module_skips"], 1)
        self.assertEqual(payload["manual_service_module_skips"], 1)
        self.assertEqual(payload["manual_subtree_begin_calls"], 1)
        self.assertEqual(payload["modulelist_active_expert_blocks"], 4)
        self.assertEqual(payload["modulelist_resident_compute_wall_time_sec"], 0.2)
        self.assertEqual(payload["packed_demand_expert_blocks"], 5)
        self.assertEqual(payload["packed_dispatch_wait_calls"], 1)


if __name__ == "__main__":
    unittest.main()
