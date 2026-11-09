#!/usr/bin/env python3
"""Performance reporting: query the SQL experiment store and the
registry, and write a performance report (JSON + Markdown).

The report covers training runs (from SQL), served predictions
(from SQL), drift checks (from SQL), the current registry state, and
the latest held-out metrics file when it exists. It is a reporting
script, not a live dashboard: re-run it after new runs or predictions
to refresh the report.

Usage: python scripts/report.py --out evals/performance_report.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_platform import experiments as experiments_module  # noqa: E402
from ml_platform.registry import Registry  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Performance reporting")
    parser.add_argument("--db", default="data/experiments.db")
    parser.add_argument("--registry", default="data/registry.json")
    parser.add_argument("--metrics", default="evals/holdout_metrics.json")
    parser.add_argument("--out", default="evals/performance_report.json")
    args = parser.parse_args()

    report = {"runs": [], "predictions": {"total_predictions": 0,
              "quarantined": 0, "mean_probability": None},
              "drift": {"total_drift_checks": 0, "drifted": 0},
              "current_model": None, "held_out_metrics": None}
    if os.path.exists(args.db):
        conn = experiments_module.connect(args.db)
        report["runs"] = experiments_module.query_runs(conn)
        report["predictions"] = experiments_module.query_prediction_summary(conn)
        report["drift"] = experiments_module.query_drift_summary(conn)
        conn.close()
    if os.path.exists(args.registry):
        current = Registry(args.registry).get_current()
        if current:
            report["current_model"] = {
                "version": current["version"], "status": current["status"],
                "metrics": current["metrics"]}
    if os.path.exists(args.metrics):
        with open(args.metrics, encoding="utf-8") as fh:
            report["held_out_metrics"] = json.load(fh)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(args.out.replace(".json", ".md"), "w", encoding="utf-8") as fh:
        fh.write("# Performance Report\n\n")
        fh.write(f"Training runs: {len(report['runs'])}\n\n")
        for run in report["runs"]:
            m = run["metrics"]
            fh.write(f"- {run['job_id']} ({run['model_version']}): "
                     f"accuracy {m['accuracy']}, precision {m['precision']}, "
                     f"recall {m['recall']}, F1 {m['f1']}, AUC {m.get('auc')}, "
                     f"gate passed: {run['gate_passed']}\n")
        p = report["predictions"]
        fh.write(f"\nPredictions served: {p['total_predictions']}, "
                 f"quarantined: {p['quarantined']}, "
                 f"mean probability: {p['mean_probability']}\n")
        d = report["drift"]
        fh.write(f"Drift checks: {d['total_drift_checks']}, drifted: {d['drifted']}\n")
        if report["current_model"]:
            fh.write(f"\nCurrent model: {report['current_model']['version']} "
                     f"({report['current_model']['status']})\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
