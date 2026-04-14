import tempfile
import unittest
from pathlib import Path
from backbone_moe import workload




class BackboneWorkloadTest(unittest.TestCase):
    def test_split_sequence_keys(self):
        train, test = workload.split_sequence_keys(["a", "b", "c", "d"], 0.5)
        self.assertEqual(train, ["a", "b"])
        self.assertEqual(test, ["c", "d"])

    def test_build_kfold_splits(self):
        folds = workload.build_kfold_splits(["a", "b", "c", "d", "e"], 2)
        self.assertEqual(len(folds), 2)
        self.assertEqual(folds[0]["test_sequences"], ["a", "b", "c"])
        self.assertEqual(folds[1]["test_sequences"], ["d", "e"])

    def test_save_subset_state_roundtrip(self):
        full_state = {"a": {"x": 1}, "b": {"x": 2}, "c": {"x": 3}}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "subset.pkl"
            workload.save_subset_state(out, full_state, ["a", "c"])
            loaded = workload.load_state_dict(out)
            self.assertEqual(sorted(loaded.keys()), ["a", "c"])
            self.assertEqual(loaded["a"]["x"], 1)
            self.assertEqual(loaded["c"]["x"], 3)


if __name__ == "__main__":
    unittest.main()
