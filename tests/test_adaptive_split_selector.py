import unittest

from backbone_moe.evaluation import compute_capacity_knee


class AdaptiveSplitSelectorTest(unittest.TestCase):
    def test_compute_capacity_knee_returns_internal_cutoff(self):
        ranked = [((0, idx), float(100 - idx)) for idx in range(100)]
        knee = compute_capacity_knee(ranked, cache_capacity=100)
        self.assertGreater(knee, 0)
        self.assertLess(knee, 100)

    def test_compute_capacity_knee_handles_flat_scores(self):
        ranked = [((0, idx), 0.0) for idx in range(32)]
        knee = compute_capacity_knee(ranked, cache_capacity=32)
        self.assertEqual(knee, 32)


if __name__ == "__main__":
    unittest.main()
