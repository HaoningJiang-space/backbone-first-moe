import unittest

from backbone_moe.evaluation import (
    build_batch_union_demand_steps,
    cache_capacity_for_mem_ratio,
    compute_capacity_knee,
    compute_residual_demand_frontier_curve,
    infer_frontier_horizon,
    rank_resident_candidates,
    select_feasible_resident_prefix,
    summarize_resident_applicability,
)


class AdaptiveSplitSelectorTest(unittest.TestCase):
    def test_cache_capacity_for_mem_ratio(self):
        self.assertEqual(cache_capacity_for_mem_ratio(0.10, 17.2), 476)

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

    def test_infer_frontier_horizon_uses_layer_granularity(self):
        access_sequence = [
            {"iter_idx": 0, "layer_experts": [[(0, 0)], [(1, 0)], [(2, 0)], [(3, 0)]]},
        ]
        horizon = infer_frontier_horizon(
            access_sequence=access_sequence,
            expert_size_mb=17.2,
            h2d_bandwidth_gbps=16.0,
            gpu_compute_time_ms=2.0,
        )
        self.assertEqual(horizon, 3)

    def test_burst_frontier_shrinks_feasible_prefix(self):
        ranked = [((0, idx), float(10 - idx)) for idx in range(6)]
        access_sequence = [
            {"iter_idx": 0, "layer_experts": [[(0, 0), (0, 4)]]},
            {"iter_idx": 1, "layer_experts": [[(0, 1), (0, 5)]]},
        ]
        single_step = select_feasible_resident_prefix(
            ranked=ranked,
            access_sequence=access_sequence,
            cache_capacity=5,
            frontier_horizon=1,
        )
        burst = select_feasible_resident_prefix(
            ranked=ranked,
            access_sequence=access_sequence,
            cache_capacity=5,
            frontier_horizon=2,
        )
        self.assertEqual(single_step["resident_capacity"], 4)
        self.assertEqual(burst["resident_capacity"], 3)

    def test_summarize_resident_applicability_reports_knee_and_frontier(self):
        ranked = [((0, idx), float(10 - idx)) for idx in range(6)]
        access_sequence = [
            {"iter_idx": 0, "layer_experts": [[(0, 0), (0, 4)]]},
            {"iter_idx": 1, "layer_experts": [[(0, 1), (0, 5)]]},
        ]
        summary = summarize_resident_applicability(
            ranked=ranked,
            access_sequence=access_sequence,
            cache_capacity=5,
            frontier_horizon=2,
        )
        self.assertIn("knee_capacity", summary)
        self.assertIn("frontier_selected_capacity", summary)
        self.assertEqual(summary["frontier_selected_capacity"], 3)
        self.assertEqual(summary["frontier_selected_capacity_tail"], 2)
        self.assertEqual(summary["frontier_selected_slack_capacity"], 2)

    def test_rank_resident_candidates_orders_by_score(self):
        class DummyAnalyzer:
            def _count_expert_accesses(self, resident_profile_ratio, score_mode="freq", depth_power=1.0):
                return {(0, 1): 3.0, (0, 0): 5.0}

        ranked = rank_resident_candidates(
            analyzer=DummyAnalyzer(),
            cache_capacity=8,
            resident_policy="profile_freq",
            resident_profile_ratio=0.2,
        )
        self.assertEqual(ranked[0][0], (0, 0))


if __name__ == "__main__":
    unittest.main()
