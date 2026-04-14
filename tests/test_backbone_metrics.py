import unittest
from pathlib import Path
from backbone_moe import metrics




class BackboneMetricsTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(
            metrics.mean_and_ci95([]),
            {"mean": None, "ci95": None, "num_points": 0},
        )

    def test_single_point(self):
        self.assertEqual(
            metrics.mean_and_ci95([3.0]),
            {"mean": 3.0, "ci95": 0.0, "num_points": 1},
        )

    def test_multiple_points(self):
        result = metrics.mean_and_ci95([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(result["mean"], 2.5)
        self.assertEqual(result["num_points"], 4)
        self.assertGreater(result["ci95"], 0.0)


if __name__ == "__main__":
    unittest.main()
