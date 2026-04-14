import unittest
from pathlib import Path
from backbone_moe import evaluation




class DummyArgs:
    state_file = "default.pkl"
    output_dir = "out"
    expert_size_mb = 17.2
    h2d_bandwidth_gbps = 16.0
    gpu_compute_time_ms = 2.0
    resident_ratio = 0.9
    resident_policy = "profile_freq"
    resident_profile_ratio = 0.2


class DummyAnalyzer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyEvalAnalyzer:
    def __init__(self):
        self.resident_policy = "profile_freq"
        self._resident_selection_cache = {}
        self.expert_size_mb = 17.2
        self.resident_profile_ratio = 0.2
        self.resident_depth_power = 1.0
        self.select_calls = []

    def _pool_capacities(self, cache_capacity):
        return min(3, cache_capacity), max(0, cache_capacity - min(3, cache_capacity))

    def _select_resident_experts(self, resident_capacity_arg, cache_capacity_arg, reset_mode_arg):
        self.select_calls.append((resident_capacity_arg, cache_capacity_arg, reset_mode_arg))
        return set()

    def simulate_with_config(self, mem_ratio, window, reset_mode=None):
        return {"prefetch_window": window, "throughput_tokens_per_sec": 10.0 + window}


class BackboneEvaluationTest(unittest.TestCase):
    def test_parse_lists(self):
        self.assertEqual(evaluation.parse_float_list("0.05, 0.10"), [0.05, 0.10])
        self.assertEqual(evaluation.parse_int_list("0, 1, 4"), [0, 1, 4])

    def test_build_two_pool_analyzer(self):
        analyzer = evaluation.build_two_pool_analyzer(DummyAnalyzer, DummyArgs(), state_file="train.pkl")
        self.assertEqual(analyzer.kwargs["state_file"], "train.pkl")
        self.assertEqual(analyzer.kwargs["cache_layout"], "two_pool")
        self.assertEqual(analyzer.kwargs["resident_policy"], "profile_freq")

    def test_build_single_cache_analyzer(self):
        analyzer = evaluation.build_single_cache_analyzer(DummyAnalyzer, DummyArgs(), "test.pkl")
        self.assertEqual(analyzer.kwargs["state_file"], "test.pkl")
        self.assertEqual(analyzer.kwargs["cache_layout"], "single")

    def test_evaluate_with_fixed_resident_set_restores_state(self):
        analyzer = DummyEvalAnalyzer()
        original_cache = dict(analyzer._resident_selection_cache)
        rows = evaluation.evaluate_with_fixed_resident_set(analyzer, {(0, 1)}, 0.10, [0, 4], "shared")
        self.assertEqual([row["prefetch_window"] for row in rows], [0, 4])
        self.assertEqual(analyzer.resident_policy, "profile_freq")
        self.assertEqual(analyzer._resident_selection_cache, original_cache)

    def test_best_by_throughput(self):
        row = evaluation.best_by_throughput(
            [
                {"throughput_tokens_per_sec": 1.0},
                {"throughput_tokens_per_sec": 3.0},
                {"throughput_tokens_per_sec": 2.0},
            ]
        )
        self.assertEqual(row["throughput_tokens_per_sec"], 3.0)


if __name__ == "__main__":
    unittest.main()
