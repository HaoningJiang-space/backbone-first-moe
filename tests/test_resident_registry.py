import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
    def _fake_archer_engine(node_map):
        class _FakeArcherEngine:
            def __init__(self, mapping):
                self.mapping = mapping

            def get_node_id(self, tensor_ids):
                return self.mapping[int(tensor_ids[0])][0]

            def get_node_byte_size(self, tensor_ids):
                return self.mapping[int(tensor_ids[0])][1]

        return _FakeArcherEngine(node_map)

    def test_load_resident_ids_preserves_order_and_records_registry(self):
        engine = self._build_engine_stub()
        with tempfile.TemporaryDirectory() as tmpdir:
            resident_file = Path(tmpdir) / "resident.json"
            resident_file.write_text(
                json.dumps(
                    {
                        "selection_rule": "frontier_prefix",
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

    def test_activate_registry_assigns_explicit_packed_fastpath_ids(self):
        engine = self._build_engine_stub()
        engine.config.model_type = "mixtral"

        class _PackedExperts:
            def __init__(self):
                self.gate_up_proj = object()
                self.down_proj = object()

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


if __name__ == "__main__":
    unittest.main()
