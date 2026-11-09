"""Prediction serving: live / shadow / canary, drift and quarantine.

Serving loads the currently promoted model from the registry. A
request is quarantined (no prediction) when a feature is missing,
non-numeric, out of range, or more than three baseline standard
deviations from the baseline mean — that is an outlier for this
equipment, not a routine reading. Batch drift is checked separately
by drift-check, which reports PSI and mean shift per feature.

Canary: a deterministic hash of the input decides the bucket, so the
same reading always takes the same path. Shadow: when enabled, the
candidate model is also scored and its probability is logged and
returned alongside the live prediction, but it never changes the
returned prediction.
"""

import hashlib

from .drift import baseline_stats, drift_report
from .features import engineer
from .model import predict_proba
from .registry import Registry
from .schema import validate_row


def _canary_bucket(row):
    digest = hashlib.sha256(str(sorted(row.items())).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _is_outlier(row, baseline):
    for feature, stats in baseline.items():
        value = float(row[feature])
        if abs(value - stats["mean"]) > 3.0 * stats["std"]:
            return True, feature
    return False, None


def predict_one(registry_path, row, baseline_rows=None, canary_percent=0,
                shadow_enabled=False, trace_id=None):
    """Serve one prediction. Returns a dict with status live/quarantined."""
    errors = validate_row(row)
    if errors:
        return {"status": "quarantined", "reason": "schema", "errors": errors,
                "trace_id": trace_id}
    registry = Registry(registry_path)
    current = registry.get_current()
    if current is None:
        return {"status": "unavailable", "reason": "no promoted model",
                "trace_id": trace_id}
    baseline = baseline_stats(baseline_rows) if baseline_rows else None
    if baseline is not None:
        outlier, feature = _is_outlier(row, baseline)
        if outlier:
            return {"status": "quarantined", "reason": "outlier",
                    "errors": [f"{feature} is more than 3 std from baseline mean"],
                    "trace_id": trace_id}
    live_model = current["model"]
    candidate = registry.get_candidate()
    use_canary = (
        candidate is not None
        and canary_percent > 0
        and _canary_bucket(row) < canary_percent
    )
    model_in_use = candidate["model"] if use_canary else live_model
    version_in_use = candidate["version"] if use_canary else current["version"]
    engineered = engineer(row)
    probability = predict_proba(model_in_use, [engineered])[0]
    result = {
        "status": "ok",
        "model_version": version_in_use,
        "mode": "canary" if use_canary else "live",
        "probability_fail_30d": round(probability, 4),
        "prediction": 1 if probability >= 0.5 else 0,
        "trace_id": trace_id,
    }
    if shadow_enabled and candidate is not None and not use_canary:
        shadow_probability = predict_proba(candidate["model"], [engineered])[0]
        result["shadow_model_version"] = candidate["version"]
        result["shadow_probability_fail_30d"] = round(shadow_probability, 4)
    return result


def drift_check(registry_path, current_rows, baseline_rows):
    """Batch drift check against the training baseline."""
    report = drift_report(baseline_rows, current_rows)
    return {
        "drifted": report["drifted"],
        "features": report["features"],
        "n_baseline": report["n_baseline"],
        "n_current": report["n_current"],
        "action": "quarantine" if report["drifted"] else "serve",
    }
