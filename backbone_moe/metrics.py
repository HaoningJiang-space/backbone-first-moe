def mean_and_ci95(values):
    if not values:
        return {"mean": None, "ci95": None, "num_points": 0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "ci95": 0.0, "num_points": 1}
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    std = var ** 0.5
    ci95 = 1.96 * std / (len(values) ** 0.5)
    return {"mean": mean, "ci95": ci95, "num_points": len(values)}
