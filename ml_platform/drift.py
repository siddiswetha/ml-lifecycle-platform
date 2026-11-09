"""Drift detection against a training baseline.

Two signals are reported per feature:
- PSI (Population Stability Index) over 10 quantile bins built from
  the baseline, with a small smoothing constant. PSI > 0.5 is treated
  as drifted here: the usual 0.25 bar is too noisy for the small
  batches this platform works with (a 40-row held-out split can
  exceed 0.25 by chance alone), so the bar is set higher and
  documented as such.
- Mean shift: (current mean - baseline mean) / baseline std.
  |mean shift| > 0.5 is treated as drifted.

A batch is drifted if any feature is drifted on either signal.
"""

import math

PSI_THRESHOLD = 0.5
MEAN_SHIFT_THRESHOLD = 0.5


def baseline_stats(rows):
    """Per-feature mean/std/min/max of the baseline (training) rows."""
    stats = {}
    for feature in ["temperature", "vibration", "pressure", "operating_hours", "load"]:
        values = [float(r[feature]) for r in rows]
        mean = sum(values) / len(values) if values else 0.0
        var = sum((v - mean) ** 2 for v in values) / len(values) if values else 0.0
        stats[feature] = {
            "mean": mean,
            "std": math.sqrt(var) if var > 0 else 1.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "n": len(values),
        }
    return stats


def _psi(expected_values, actual_values, bins=10):
    """PSI over quantile bins built from the expected (baseline) values.

    Equal-frequency bins keep the expected share near 1/bins in every
    bin, so a fresh sample from the same distribution scores near
    zero instead of tripping on empty equal-width bins.
    """
    ordered = sorted(expected_values)
    if not ordered or ordered[0] == ordered[-1]:
        return 0.0
    edges = []
    for i in range(1, bins):
        edges.append(ordered[int(i * len(ordered) / bins)])
    edges = sorted(set(edges))
    # Bin boundaries: -inf, edges..., +inf.
    bounds = [float("-inf")] + edges + [float("inf")]
    total = 0.0
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if i == 0:
            expected_share = sum(1 for v in expected_values if v <= hi) / len(expected_values)
            actual_share = sum(1 for v in actual_values if v <= hi) / len(actual_values)
        elif i == len(bounds) - 2:
            expected_share = sum(1 for v in expected_values if v > lo) / len(expected_values)
            actual_share = sum(1 for v in actual_values if v > lo) / len(actual_values)
        else:
            expected_share = sum(1 for v in expected_values if lo < v <= hi) / len(expected_values)
            actual_share = sum(1 for v in actual_values if lo < v <= hi) / len(actual_values)
        expected_share = max(expected_share, 0.0001)
        actual_share = max(actual_share, 0.0001)
        total += (actual_share - expected_share) * math.log(actual_share / expected_share)
    return total


def drift_report(baseline_rows, current_rows):
    """Return {features: {...}, drifted: bool, n_baseline, n_current}."""
    baseline = baseline_stats(baseline_rows)
    features = {}
    any_drifted = False
    for feature, stats in baseline.items():
        expected = [float(r[feature]) for r in baseline_rows]
        actual = [float(r[feature]) for r in current_rows]
        psi = _psi(expected, actual) if actual else 0.0
        current_mean = sum(actual) / len(actual) if actual else 0.0
        mean_shift = (current_mean - stats["mean"]) / stats["std"] if stats["std"] else 0.0
        drifted = abs(psi) > PSI_THRESHOLD or abs(mean_shift) > MEAN_SHIFT_THRESHOLD
        any_drifted = any_drifted or drifted
        features[feature] = {
            "psi": round(psi, 4),
            "mean_shift": round(mean_shift, 4),
            "baseline_mean": round(stats["mean"], 4),
            "current_mean": round(current_mean, 4),
            "drifted": drifted,
        }
    return {
        "features": features,
        "drifted": any_drifted,
        "n_baseline": len(baseline_rows),
        "n_current": len(current_rows),
    }
