from experiments.observation.analyze_serving_bottlenecks import (
    compute_step_reuse_distances,
    compute_working_set_stats,
    summarize_bottleneck,
    summarize_distribution,
)


def test_summarize_distribution_empty():
    stats = summarize_distribution([])
    assert stats["count"] == 0
    assert stats["mean"] == 0.0


def test_compute_step_reuse_distances():
    demand_steps = [
        {(0, 0), (0, 1)},
        {(0, 1), (1, 0)},
        {(0, 0)},
    ]
    distances = compute_step_reuse_distances(demand_steps)
    assert sorted(distances) == [1, 2]


def test_compute_working_set_stats():
    demand_steps = [
        {(0, 0)},
        {(0, 1)},
        {(0, 0), (0, 2)},
    ]
    stats = compute_working_set_stats(demand_steps, [1, 2])
    assert stats["1"]["p50"] == 1.0
    assert stats["2"]["max"] == 3.0


def test_summarize_bottleneck():
    row = {
        "total_residual_stall_ms": 300.0,
        "throughput_tokens_per_sec": 10.0,
    }
    compute_only = {
        "total_compute_time_ms": 700.0,
        "throughput_tokens_per_sec": 20.0,
    }
    summary = summarize_bottleneck(row, compute_only)
    assert round(summary["loading_share"], 3) == 0.3
    assert round(summary["zero_loading_speedup_upper_bound"], 3) == 2.0
