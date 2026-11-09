#!/usr/bin/env python3
"""Held-out evaluation harness.

Loads the currently promoted model from the registry, scores the
held-out split of a dataset, and writes metrics to evals/holdout_metrics.json
(and prints them). Exits non-zero when no promoted model exists.

Usage:
  python scripts/evaluate.py --dataset data/synthetic.csv --out evals/holdout_metrics.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_platform import data as data_module  # noqa: E402
from ml_platform.features import engineer_all  # noqa: E402
from ml_platform.model import evaluate_metrics, predict, predict_proba  # noqa: E402
from ml_platform.registry import Registry  # noqa: E402
from ml_platform.schema import validate_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Held-out evaluation harness")
    parser.add_argument("--dataset", default="data/synthetic.csv")
    parser.add_argument("--registry", default="data/registry.json")
    parser.add_argument("--out", default="evals/holdout_metrics.json")
    args = parser.parse_args()

    registry = Registry(args.registry)
    current = registry.get_current()
    if current is None:
        print("no promoted model in registry", file=sys.stderr)
        raise SystemExit(2)
    rows = data_module.read_csv(args.dataset)
    valid, errors = validate_dataset(rows)
    if errors:
        print(f"dataset failed schema validation: {errors[:3]}", file=sys.stderr)
        raise SystemExit(2)
    _, held_out = data_module.split(valid, test_size=0.2, seed=42)
    held_out = engineer_all(held_out)
    labels = [r["will_fail_30d"] for r in held_out]
    scores = predict_proba(current["model"], held_out)
    predictions = predict(current["model"], held_out)
    metrics = evaluate_metrics(labels, predictions, scores)
    result = {
        "model_version": current["version"],
        "dataset": args.dataset,
        "held_out_n": len(held_out),
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
