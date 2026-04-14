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
        # Flat utility implies no clear backbone concentration; fall back to
        # using the full capacity as the neutral cutoff.
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


def best_by_throughput(rows):
    return max(rows, key=lambda row: row["throughput_tokens_per_sec"])
