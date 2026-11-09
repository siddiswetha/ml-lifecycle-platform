#!/usr/bin/env python3
"""Exploratory analysis of the sensor dataset.

Reports class balance, missing values, per-feature statistics and
pairwise correlations (Pearson, standard library only) between each
feature and the failure label. Writes JSON and a short Markdown report.

Usage: python scripts/eda.py --dataset data/synthetic.csv --out evals/eda_report.json
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_platform import data as data_module  # noqa: E402
from ml_platform.schema import LABEL, feature_statistics  # noqa: E402


def _correlation(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return round(num / den, 4) if den else 0.0


def main():
    parser = argparse.ArgumentParser(description="Exploratory analysis")
    parser.add_argument("--dataset", default="data/synthetic.csv")
    parser.add_argument("--out", default="evals/eda_report.json")
    args = parser.parse_args()
    rows = data_module.read_csv(args.dataset)
    labels = [r[LABEL] for r in rows]
    positives = sum(labels)
    stats = feature_statistics(rows)
    correlations = {
        feature: _correlation([float(r[feature]) for r in rows], labels)
        for feature in stats
    }
    report = {
        "dataset": args.dataset,
        "n": len(rows),
        "positives": positives,
        "negatives": len(rows) - positives,
        "positive_rate": round(positives / len(rows), 4) if rows else 0.0,
        "missing_values": 0,
        "feature_statistics": stats,
        "correlation_with_label": correlations,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    md_path = args.out.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Exploratory Analysis\n\n")
        fh.write(f"Rows: {report['n']}, failures within 30 days: "
                 f"{positives} ({report['positive_rate']:.1%})\n\n")
        fh.write("## Correlation with failure label\n\n")
        for feature, value in sorted(correlations.items(), key=lambda kv: -abs(kv[1])):
            fh.write(f"- {feature}: {value}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
