#!/usr/bin/env python3
"""CLI for the ML Lifecycle Platform.

Usage:
  python scripts/cli.py generate --n 600 --seed 42 --out data/synthetic.csv
  python scripts/cli.py train --job-id job-001 --dataset data/synthetic.csv
  python scripts/cli.py register --model data/model.json --metrics data/metrics.json
  python scripts/cli.py promote --version v1
  python scripts/cli.py rollback
  python scripts/cli.py predict --features '{"temperature": 78, "vibration": 3.1, ...}'
  python scripts/cli.py drift-check --baseline data/synthetic.csv --current data/current.csv

All paths default under data/. The CLI never needs a running service:
it works directly on the same JSON registry / job files the services use.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_platform import data as data_module  # noqa: E402
from ml_platform.model import load_model  # noqa: E402
from ml_platform.registry import Registry  # noqa: E402
from ml_platform.serving import drift_check, predict_one  # noqa: E402
from ml_platform.trainer import JobStore, run_job  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")


def _registry_path(args):
    return getattr(args, "registry", None) or os.path.join(DATA, "registry.json")


def cmd_generate(args):
    rows = data_module.generate_synthetic(n=args.n, seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    data_module.write_csv(rows, args.out)
    positives = sum(r["will_fail_30d"] for r in rows)
    print(json.dumps({"out": args.out, "n": len(rows), "positives": positives,
                      "seed": args.seed}, indent=2))


def cmd_train(args):
    # Keep the job store next to the registry, so a job id is
    # idempotent against the same registry, not against a different one.
    jobs_path = _registry_path(args).replace("registry.json", "jobs.json")
    store = JobStore(jobs_path)
    store.create_job(args.job_id, args.dataset)
    job = run_job(store, args.job_id, _registry_path(args), epochs=args.epochs)
    print(json.dumps(job, indent=2))


def cmd_register(args):
    model = load_model(args.model)
    with open(args.metrics, encoding="utf-8") as fh:
        metrics = json.load(fh)
    registry = Registry(_registry_path(args))
    entry = registry.register(model, metrics, notes=args.notes or "")
    print(json.dumps({"version": entry["version"], "status": entry["status"],
                      "metrics": entry["metrics"]}, indent=2))


def cmd_promote(args):
    registry = Registry(_registry_path(args))
    entry = registry.promote(args.version)
    print(json.dumps({"version": entry["version"], "status": entry["status"]}, indent=2))


def cmd_rollback(args):
    registry = Registry(_registry_path(args))
    entry = registry.rollback()
    print(json.dumps({"version": entry["version"], "status": entry["status"]}, indent=2))


def cmd_predict(args):
    row = json.loads(args.features)
    baseline = None
    if args.baseline and os.path.exists(args.baseline):
        baseline = data_module.read_csv(args.baseline)
    result = predict_one(_registry_path(args), row, baseline_rows=baseline,
                         canary_percent=args.canary_percent,
                         shadow_enabled=args.shadow)
    print(json.dumps(result, indent=2))


def cmd_drift_check(args):
    baseline = data_module.read_csv(args.baseline)
    current = data_module.read_csv(args.current)
    result = drift_check(_registry_path(args), current, baseline)
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="ML Lifecycle Platform CLI")
    parser.add_argument("--registry", default=None, help="registry JSON path")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="generate synthetic sensor data")
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.path.join(DATA, "synthetic.csv"))
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("train", help="run a training job and register if it passes the gate")
    p.add_argument("--job-id", required=True)
    p.add_argument("--dataset", default=os.path.join(DATA, "synthetic.csv"))
    p.add_argument("--epochs", type=int, default=300)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("register", help="register a model file as a candidate")
    p.add_argument("--model", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--notes", default="")
    p.set_defaults(func=cmd_register)

    p = sub.add_parser("promote", help="promote a candidate version to live")
    p.add_argument("--version", required=True)
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("rollback", help="roll back to the previous promoted version")
    p.set_defaults(func=cmd_rollback)

    p = sub.add_parser("predict", help="predict failure risk for one sensor reading")
    p.add_argument("--features", required=True, help="JSON object of the five features")
    p.add_argument("--baseline", default=None)
    p.add_argument("--canary-percent", type=int, default=0)
    p.add_argument("--shadow", action="store_true")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("drift-check", help="check a batch for drift against a baseline")
    p.add_argument("--baseline", required=True)
    p.add_argument("--current", required=True)
    p.set_defaults(func=cmd_drift_check)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
