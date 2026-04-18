import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
        self.assertEqual(registry["admitted_tensor_count"], 4)
        self.assertTrue(registry["clipped"])


if __name__ == "__main__":
    unittest.main()
