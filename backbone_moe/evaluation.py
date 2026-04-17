import math
from bisect import bisect_right


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_two_pool_analyzer(analyzer_cls, args, state_file=None, resident_policy=None):
    return analyzer_cls(
        state_file=state_file or args.state_file,
        mode="oracle",
        predictor="history_freq",
        output_dir=args.output_dir,
        expert_size_mb=args.expert_size_mb,
        h2d_bandwidth_gbps=args.h2d_bandwidth_gbps,
        gpu_compute_time_ms=args.gpu_compute_time_ms,
        cache_layout="two_pool",
        resident_ratio=args.resident_ratio,
        resident_policy=resident_policy or args.resident_policy,
        resident_profile_ratio=args.resident_profile_ratio,
    )


def build_single_cache_analyzer(analyzer_cls, args, state_file):
    return analyzer_cls(
        state_file=str(state_file),
        mode="oracle",
        predictor="history_freq",
        output_dir=args.output_dir,
        expert_size_mb=args.expert_size_mb,
        h2d_bandwidth_gbps=args.h2d_bandwidth_gbps,
        gpu_compute_time_ms=args.gpu_compute_time_ms,
        cache_layout="single",
    )


def resident_set_from_analyzer(analyzer, mem_ratio, reset_mode):
    info = analyzer.get_resident_set(mem_ratio, reset_mode=reset_mode)
    return {
        (int(item["layer"]), int(item["expert"]))
        for item in info["resident_set"]
    }


def evaluate_with_fixed_resident_set(eval_analyzer, resident_set, mem_ratio, windows, reset_mode):
    original_policy = eval_analyzer.resident_policy
    original_cache = dict(eval_analyzer._resident_selection_cache)
    eval_analyzer.resident_policy = "none"
    rows = []
    try:
        for window in windows:
            total_gpu_memory_mb = 80 * 1024
            available_memory_mb = total_gpu_memory_mb * mem_ratio
            cache_capacity = int(available_memory_mb / eval_analyzer.expert_size_mb)
            resident_capacity, _ = eval_analyzer._pool_capacities(cache_capacity)
            cache_key = (
                "frozen_set",
                resident_capacity,
                cache_capacity,
                reset_mode,
                eval_analyzer.resident_profile_ratio,
                eval_analyzer.resident_depth_power,
            )
            eval_analyzer._resident_selection_cache[cache_key] = set(resident_set)
            eval_analyzer.resident_policy = "frozen_set"
            original_select = eval_analyzer._select_resident_experts

            def select_override(resident_capacity_arg, cache_capacity_arg, reset_mode_arg):
                return set(resident_set)

            eval_analyzer._select_resident_experts = select_override
            try:
                rows.append(eval_analyzer.simulate_with_config(mem_ratio, window, reset_mode=reset_mode))
            finally:
                eval_analyzer._select_resident_experts = original_select
    finally:
        eval_analyzer.resident_policy = original_policy
        eval_analyzer._resident_selection_cache = original_cache
    return rows


def evaluate_with_fixed_resident_layout(
    eval_analyzer,
    resident_set,
    resident_capacity,
    cache_capacity,
    mem_ratio,
    windows,
    reset_mode,
):
    """
    Evaluate a two-pool layout with an explicit resident set and explicit pool sizes.

    This is used by adaptive split selection, where the split variable is the
    resident pool size itself rather than a pre-chosen resident_ratio.
    """
    original_policy = eval_analyzer.resident_policy
    original_cache = dict(eval_analyzer._resident_selection_cache)
    original_select = eval_analyzer._select_resident_experts
    original_pool_capacities = eval_analyzer._pool_capacities
    rows = []
    try:
        eval_analyzer.resident_policy = "none"

        def pool_override(cache_capacity_arg):
            return resident_capacity, max(0, cache_capacity - resident_capacity)

        def select_override(resident_capacity_arg, cache_capacity_arg, reset_mode_arg):
            return set(resident_set)

        eval_analyzer._pool_capacities = pool_override
        eval_analyzer._select_resident_experts = select_override

        cache_key = (
            "frozen_layout",
            resident_capacity,
            cache_capacity,
            reset_mode,
            eval_analyzer.resident_profile_ratio,
            eval_analyzer.resident_depth_power,
        )
        eval_analyzer._resident_selection_cache[cache_key] = set(resident_set)
        eval_analyzer.resident_policy = "frozen_layout"

        for window in windows:
            rows.append(eval_analyzer.simulate_with_config(mem_ratio, window, reset_mode=reset_mode))
    finally:
        eval_analyzer.resident_policy = original_policy
        eval_analyzer._resident_selection_cache = original_cache
        eval_analyzer._select_resident_experts = original_select
        eval_analyzer._pool_capacities = original_pool_capacities
    return rows


def compute_capacity_knee(ranked, cache_capacity):
    """
    Compute the resident-slot knee directly from the cumulative utility curve.

    The selected k maximizes the gap between cumulative utility share and
    cumulative capacity share, i.e. the Lorenz-style distance from the diagonal.
    This gives a single backbone cutoff instead of a search over many split
    candidates.
    """
    if cache_capacity <= 0:
        return 0

    if not ranked:
        return 0

    top_scores = [max(0.0, float(score)) for _, score in ranked[:cache_capacity]]
    if not top_scores:
        return 0

    total_score = sum(top_scores)
    max_capacity = min(cache_capacity, len(top_scores))

    if total_score <= 0.0:
        return max_capacity

    cumulative = []
    running = 0.0
    for score in top_scores:
        running += score
        cumulative.append(running)

    max_gap = None
    knee_idx = max_capacity
    for idx, prefix_sum in enumerate(cumulative, start=1):
        x = idx / float(max_capacity)
        y = prefix_sum / float(total_score)
        gap = y - x
        if max_gap is None or gap > max_gap:
            max_gap = gap
            knee_idx = idx

    return max(0, min(cache_capacity, knee_idx))


def build_batch_union_demand_steps(access_sequence):
    """
    Collapse a routed trace into batch-step demand sets.

    Each step corresponds to one (iter_idx, layer_idx) pair across all active
    sequences. Resident memory covers the backbone prefix; the remaining slack
    must absorb the non-resident union at each batch step.
    """
    grouped = {}
    for token_data in access_sequence:
        iter_idx = int(token_data["iter_idx"])
        for layer_idx, layer_experts in enumerate(token_data["layer_experts"]):
            if not layer_experts:
                continue
            grouped.setdefault((iter_idx, layer_idx), set()).update(layer_experts)
    return [grouped[key] for key in sorted(grouped.keys())]


def infer_frontier_horizon(
    access_sequence,
    expert_size_mb,
    h2d_bandwidth_gbps,
    gpu_compute_time_ms,
):
    """
    Infer the tail burst horizon from hardware transfer/compute overlap.

    The demand trace is represented at the (iter_idx, layer_idx) granularity.
    We therefore convert token-level compute time into per-layer slices, then
    ask how many such slices can overlap with one expert H2D transfer.
    """
    if not access_sequence:
        return 1

    if expert_size_mb <= 0 or h2d_bandwidth_gbps <= 0 or gpu_compute_time_ms <= 0:
        return 1

    num_layers = max(
        (len(token_data.get("layer_experts", [])) for token_data in access_sequence),
        default=1,
    )
    num_layers = max(1, int(num_layers))
    step_compute_time_ms = gpu_compute_time_ms / float(num_layers)
    if step_compute_time_ms <= 0:
        return 1

    transfer_time_ms = expert_size_mb / float(h2d_bandwidth_gbps)
    return max(1, int(math.ceil(transfer_time_ms / step_compute_time_ms)))


def _percentile_int(values, percentile):
    if not values:
        return 0
    if percentile >= 1.0:
        return int(max(values))
    if percentile <= 0.0:
        return int(min(values))
    ordered = sorted(int(v) for v in values)
    idx = int(round((len(ordered) - 1) * percentile))
    return ordered[max(0, min(len(ordered) - 1, idx))]


def compute_residual_demand_frontier_curve(
    ranked,
    access_sequence,
    cache_capacity,
    frontier_percentile=1.0,
    frontier_horizon=1,
):
    """
    Compute F(k): the residual demand frontier after pinning the top-k prefix.

    For each batch-step (iter_idx, layer_idx), we take the union of demanded
    experts across active sequences, remove the resident prefix, and count the
    remaining tail demand. We then widen this into a burst frontier by taking a
    rolling union across `frontier_horizon` consecutive steps.
    """
    if cache_capacity <= 0:
        return [0]

    if not ranked or not access_sequence:
        return [0] * (cache_capacity + 1)

    rank_index = {
        expert_key: idx + 1
        for idx, (expert_key, _) in enumerate(ranked)
    }
    demand_steps = build_batch_union_demand_steps(access_sequence)
    if not demand_steps:
        return [0] * (cache_capacity + 1)

    step_rank_lists = []
    for step_set in demand_steps:
        step_ranks = sorted(
            rank_index[expert_key]
            for expert_key in step_set
            if expert_key in rank_index
        )
        step_rank_lists.append(step_ranks)

    frontier_horizon = max(1, int(frontier_horizon))
    frontier_curve = []
    for resident_capacity in range(cache_capacity + 1):
        residual_counts = []
        window_counts = {}
        window_unique = 0

        def add_step(step_ranks):
            nonlocal window_unique
            start_idx = bisect_right(step_ranks, resident_capacity)
            for rank in step_ranks[start_idx:]:
                prev = window_counts.get(rank, 0)
                window_counts[rank] = prev + 1
                if prev == 0:
                    window_unique += 1

        def remove_step(step_ranks):
            nonlocal window_unique
            start_idx = bisect_right(step_ranks, resident_capacity)
            for rank in step_ranks[start_idx:]:
                prev = window_counts.get(rank, 0)
                if prev <= 1:
                    window_counts.pop(rank, None)
                    if prev == 1:
                        window_unique -= 1
                else:
                    window_counts[rank] = prev - 1

        window_end = min(frontier_horizon, len(step_rank_lists))
        for idx in range(window_end):
            add_step(step_rank_lists[idx])
        residual_counts.append(window_unique)

        for start in range(1, len(step_rank_lists)):
            remove_step(step_rank_lists[start - 1])
            next_idx = start + frontier_horizon - 1
            if next_idx < len(step_rank_lists):
                add_step(step_rank_lists[next_idx])
            residual_counts.append(window_unique)

        frontier_curve.append(_percentile_int(residual_counts, frontier_percentile))
    return frontier_curve


def select_feasible_resident_prefix(
    ranked,
    access_sequence,
    cache_capacity,
    frontier_percentile=1.0,
    frontier_horizon=1,
):
    """
    Select the largest resident prefix that still leaves enough tail slack.

    k* = max { k : k + F(k) <= B }

    where F(k) is the residual demand frontier after pinning the top-k experts,
    and B is the total cache capacity. This is the non-incremental selector:
    resident sizing becomes a feasibility rule induced by the routed trace,
    rather than a sweep over ratios or capacities.
    """
    frontier_curve = compute_residual_demand_frontier_curve(
        ranked=ranked,
        access_sequence=access_sequence,
        cache_capacity=cache_capacity,
        frontier_percentile=frontier_percentile,
        frontier_horizon=frontier_horizon,
    )
    max_ranked = min(cache_capacity, len(ranked))
    feasible_capacity = 0
    feasible_frontier = frontier_curve[0] if frontier_curve else 0

    for resident_capacity in range(max_ranked + 1):
        frontier = frontier_curve[resident_capacity]
        if resident_capacity + frontier <= cache_capacity:
            feasible_capacity = resident_capacity
            feasible_frontier = frontier

    return {
        "resident_capacity": int(feasible_capacity),
        "speculative_capacity": int(max(0, cache_capacity - feasible_capacity)),
        "resident_ratio": float(feasible_capacity / cache_capacity) if cache_capacity > 0 else 0.0,
        "frontier_capacity": int(feasible_frontier),
        "selection_rule": "frontier_feasible_prefix",
        "frontier_percentile": float(frontier_percentile),
        "frontier_horizon": int(frontier_horizon),
        "frontier_curve": [int(x) for x in frontier_curve],
    }


def best_by_throughput(rows):
    return max(rows, key=lambda row: row["throughput_tokens_per_sec"])


def cache_capacity_for_mem_ratio(mem_ratio, expert_size_mb, total_gpu_memory_mb=80 * 1024):
    available_memory_mb = float(total_gpu_memory_mb) * float(mem_ratio)
    if expert_size_mb <= 0:
        return 0
    return max(0, int(available_memory_mb / float(expert_size_mb)))


def rank_resident_candidates(
    analyzer,
    cache_capacity,
    resident_policy,
    resident_profile_ratio,
    resident_depth_power=1.0,
    reset_mode="shared",
):
    if resident_policy == "oracle_freq":
        scores = dict(analyzer.expert_access_count)
    elif resident_policy == "profile_freq":
        scores = analyzer._count_expert_accesses(resident_profile_ratio, score_mode="freq")
    elif resident_policy == "profile_depth_freq":
        scores = analyzer._count_expert_accesses(
            resident_profile_ratio,
            score_mode="depth_freq",
            depth_power=resident_depth_power,
        )
    elif resident_policy == "profile_miss_stall":
        scores = analyzer._profile_miss_stall_scores(
            resident_profile_ratio,
            cache_capacity,
            0,
            reset_mode,
        )
    else:
        raise ValueError(f"Unsupported resident_policy: {resident_policy}")

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )


def summarize_resident_applicability(
    ranked,
    access_sequence,
    cache_capacity,
    frontier_percentile=1.0,
    frontier_horizon=1,
):
    knee_capacity = compute_capacity_knee(ranked, cache_capacity)
    selected = select_feasible_resident_prefix(
        ranked=ranked,
        access_sequence=access_sequence,
        cache_capacity=cache_capacity,
        frontier_percentile=frontier_percentile,
        frontier_horizon=frontier_horizon,
    )
    frontier_curve = list(selected["frontier_curve"])
    frontier_at_knee = int(frontier_curve[knee_capacity]) if knee_capacity < len(frontier_curve) else 0
    knee_slack_capacity = max(0, cache_capacity - knee_capacity)
    feasible_slack_capacity = max(0, cache_capacity - int(selected["resident_capacity"]))
    knee_slack_utilization = (
        float(frontier_at_knee) / float(knee_slack_capacity)
        if knee_slack_capacity > 0
        else (math.inf if frontier_at_knee > 0 else 0.0)
    )
    feasible_slack_utilization = (
        float(selected["frontier_capacity"]) / float(feasible_slack_capacity)
        if feasible_slack_capacity > 0
        else (math.inf if int(selected["frontier_capacity"]) > 0 else 0.0)
    )
    return {
        "cache_capacity": int(cache_capacity),
        "knee_capacity": int(knee_capacity),
        "knee_ratio": float(knee_capacity / cache_capacity) if cache_capacity > 0 else 0.0,
        "knee_frontier_capacity": int(frontier_at_knee),
        "knee_slack_capacity": int(knee_slack_capacity),
        "knee_slack_utilization": float(knee_slack_utilization),
        "frontier_selected_capacity": int(selected["resident_capacity"]),
        "frontier_selected_ratio": float(selected["resident_ratio"]),
        "frontier_selected_capacity_tail": int(selected["frontier_capacity"]),
        "frontier_selected_slack_capacity": int(feasible_slack_capacity),
        "frontier_selected_slack_utilization": float(feasible_slack_utilization),
        "frontier_horizon": int(selected["frontier_horizon"]),
        "frontier_percentile": float(selected["frontier_percentile"]),
        "frontier_curve": frontier_curve,
    }
