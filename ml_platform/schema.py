"""Feature schema: the contract every row must satisfy before training
or prediction. Ranges are the documented operating envelope of the
synthetic sensor data; rows outside it are quarantined, never silently
clipped into a prediction.
"""

import math

FEATURES = {
    "temperature": {"min": 40.0, "max": 120.0, "unit": "degC"},
    "vibration": {"min": 0.0, "max": 10.0, "unit": "mm/s"},
    "pressure": {"min": 50.0, "max": 160.0, "unit": "bar"},
    "operating_hours": {"min": 0.0, "max": 20000.0, "unit": "hours"},
    "load": {"min": 0.0, "max": 1.0, "unit": "fraction"},
}

LABEL = "will_fail_30d"


def validate_row(row):
    """Return a list of error strings; empty means the row is valid."""
    errors = []
    if not isinstance(row, dict):
        return ["row is not an object"]
    for name, spec in FEATURES.items():
        if name not in row or row[name] is None:
            errors.append(f"missing feature: {name}")
            continue
        value = row[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{name} is not a number")
            continue
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            errors.append(f"{name} is not finite")
            continue
        if value < spec["min"] or value > spec["max"]:
            errors.append(
                f"{name}={value} outside range [{spec['min']}, {spec['max']}]"
            )
    return errors


def validate_dataset(rows):
    """Split rows into (valid_rows, errors) where errors are {index, errors}."""
    valid = []
    errors = []
    for index, row in enumerate(rows):
        row_errors = validate_row(row)
        if row_errors:
            errors.append({"index": index, "errors": row_errors})
        else:
            valid.append(row)
    return valid, errors


def feature_statistics(rows):
    """Mean / std / min / max per feature, for data-workflow reports."""
    stats = {}
    for name in FEATURES:
        values = [float(r[name]) for r in rows]
        if not values:
            stats[name] = {"n": 0, "mean": None, "std": None, "min": None, "max": None}
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        stats[name] = {
            "n": len(values),
            "mean": round(mean, 4),
            "std": round(math.sqrt(var), 4),
            "min": min(values),
            "max": max(values),
        }
    return stats
