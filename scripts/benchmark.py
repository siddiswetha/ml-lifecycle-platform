#!/usr/bin/env python3
"""Model benchmarking: train the same logistic model under several
configurations (epochs x learning rate) on the same deterministic
split, evaluate each on the same held-out set, and rank them.

This is benchmarking of one model type under different settings, not
a comparison with other model types. Writes evals/benchmark.json and
evals/benchmark.md.

Usage: python scripts/benchmark.py --dataset data/synthetic.csv
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_platform import data as data_module  # noqa: E402
from ml_platform.features import engineer_all  # noqa: E402
from ml_platform.model import evaluate_metrics, predict, predict_proba, train_logistic_regression  # noqa: E402
from ml_platform.schema import validate_dataset  # noqa: E402

CONFIGS = [
    {"epochs": 100, "learning_rate": 0.05},
    {"epochs": 300, "learning_rate": 0.05},
    {"epochs": 300, "learning_rate": 0.10},
    {"epochs": 600, "learning_rate": 0.05},
    {"epochs": 600, "learning_rate": 0.10},
]


def main():
    parser = argparse.ArgumentParser(description="Model benchmarking")
    parser.add_argument("--dataset", default="data/synthetic.csv")
    parser.add_argument("--out", default="evals/benchmark.json")
    args = parser.parse_args()
    rows = data_module.read_csv(args.dataset)
    valid, errors = validate_dataset(rows)
    if errors:
        print(f"dataset failed schema validation: {errors[:3]}", file=sys.stderr)
        raise SystemExit(2)
    train_rows, held_out_rows = data_module.split(valid, test_size=0.2, seed=42)
    train_rows = engineer_all(train_rows)
    held_out_rows = engineer_all(held_out_rows)
    held_labels = [r["will_fail_30d"] for r in held_out_rows]
    results = []
    for config in CONFIGS:
        labels = [r["will_fail_30d"] for r in train_rows]
        model = train_logistic_regression(train_rows, labels, **config)
        scores = predict_proba(model, held_out_rows)
        predictions = predict(model, held_out_rows)
        metrics = evaluate_metrics(held_labels, predictions, scores)
        results.append({"config": config, "metrics": metrics})
    results.sort(key=lambda r: r["metrics"]["f1"], reverse=True)
    best = results[0]
    report = {"held_out_n": len(held_out_rows), "results": results,
              "best_config": best["config"], "best_metrics": best["metrics"]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(args.out.replace(".json", ".md"), "w", encoding="utf-8") as fh:
        fh.write("# Model Benchmark\n\n")
        fh.write(f"Held-out n: {len(held_out_rows)}\n\n")
        fh.write("| epochs | learning rate | accuracy | precision | recall | F1 | AUC |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            m = r["metrics"]
            fh.write(f"| {r['config']['epochs']} | {r['config']['learning_rate']} | "
                     f"{m['accuracy']} | {m['precision']} | {m['recall']} | "
                     f"{m['f1']} | {m['auc']} |\n")
        fh.write(f"\nBest by F1: {best['config']} — F1 {best['metrics']['f1']}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
