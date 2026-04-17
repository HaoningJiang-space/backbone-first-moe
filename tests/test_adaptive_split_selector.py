import unittest

from backbone_moe.evaluation import (
    build_batch_union_demand_steps,
    compute_capacity_knee,
    compute_residual_demand_frontier_curve,
    select_feasible_resident_prefix,
)


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

    def test_build_batch_union_demand_steps_groups_same_iter_layer(self):
        access_sequence = [
            {
                "seq_id": "s0",
                "iter_idx": 0,
                "layer_experts": [[(0, 0), (0, 1)], [(1, 0)]],
                "experts": [(0, 0), (0, 1), (1, 0)],
            },
            {
                "seq_id": "s1",
                "iter_idx": 0,
                "layer_experts": [[(0, 1), (0, 2)], [(1, 0), (1, 1)]],
                "experts": [(0, 1), (0, 2), (1, 0), (1, 1)],
            },
        ]
        steps = build_batch_union_demand_steps(access_sequence)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0], {(0, 0), (0, 1), (0, 2)})
        self.assertEqual(steps[1], {(1, 0), (1, 1)})

    def test_compute_residual_demand_frontier_curve_tracks_tail_union(self):
        ranked = [
            ((0, 0), 10.0),
            ((0, 1), 9.0),
            ((1, 0), 8.0),
            ((0, 2), 7.0),
            ((1, 1), 6.0),
        ]
        access_sequence = [
            {
                "seq_id": "s0",
                "iter_idx": 0,
                "layer_experts": [[(0, 0), (0, 1)], [(1, 0)]],
                "experts": [(0, 0), (0, 1), (1, 0)],
            },
            {
                "seq_id": "s1",
                "iter_idx": 0,
                "layer_experts": [[(0, 1), (0, 2)], [(1, 0), (1, 1)]],
                "experts": [(0, 1), (0, 2), (1, 0), (1, 1)],
            },
        ]
        curve = compute_residual_demand_frontier_curve(ranked, access_sequence, cache_capacity=4)
        self.assertEqual(curve[0], 3)
        self.assertEqual(curve[1], 2)
        self.assertEqual(curve[2], 2)
        self.assertEqual(curve[3], 1)
        self.assertEqual(curve[4], 1)

    def test_select_feasible_resident_prefix_uses_frontier_constraint(self):
        ranked = [
            ((0, 0), 10.0),
            ((0, 1), 9.0),
            ((1, 0), 8.0),
            ((0, 2), 7.0),
            ((1, 1), 6.0),
        ]
        access_sequence = [
            {
                "seq_id": "s0",
                "iter_idx": 0,
                "layer_experts": [[(0, 0), (0, 1)], [(1, 0)]],
                "experts": [(0, 0), (0, 1), (1, 0)],
            },
            {
                "seq_id": "s1",
                "iter_idx": 0,
                "layer_experts": [[(0, 1), (0, 2)], [(1, 0), (1, 1)]],
                "experts": [(0, 1), (0, 2), (1, 0), (1, 1)],
            },
        ]
        selected = select_feasible_resident_prefix(ranked, access_sequence, cache_capacity=4)
        self.assertEqual(selected["resident_capacity"], 3)
        self.assertEqual(selected["frontier_capacity"], 1)
        self.assertEqual(selected["speculative_capacity"], 1)
        self.assertEqual(selected["selection_rule"], "frontier_feasible_prefix")


if __name__ == "__main__":
    unittest.main()
