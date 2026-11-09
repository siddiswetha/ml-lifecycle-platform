"""Training-job queue and orchestration.

A job is created with a client-supplied job id. Creating the same job
id twice is idempotent: the original job is returned and no second
training run is started. A run validates the dataset against the
feature schema, splits it deterministically, trains the logistic
model, evaluates on the held-out split, and registers a candidate
*only* when the held-out metrics pass the promotion gate. A model
that fails the gate is reported, not registered.

Transient failures are retried up to three times; a permanent failure
(schema errors, empty data) fails the job immediately.
"""

import json
import os
from datetime import datetime, timezone

from . import data as data_module
from . import experiments as experiments_module
from .features import engineer_all
from .model import evaluate_metrics, predict, predict_proba, train_logistic_regression
from .registry import Registry, passes_gate
from .schema import validate_dataset


class TransientError(Exception):
    """A failure worth retrying (for example a flaky downstream call)."""


def retry(fn, attempts=3):
    """Run fn, retrying TransientError up to `attempts` times in total."""
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except TransientError as exc:  # noqa: BLE001 - retry is the point
            last = exc
    raise last


def _now():
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, path):
        self.path = path
        self._jobs = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                self._jobs = json.load(fh)

    def _save(self):
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._jobs, fh, indent=2)

    def create_job(self, job_id, dataset_path):
        """Idempotent creation: an existing job id returns the stored job."""
        if job_id in self._jobs:
            return self._jobs[job_id], False
        job = {
            "job_id": job_id,
            "dataset_path": dataset_path,
            "status": "queued",
            "attempts": 0,
            "result": None,
            "error": None,
            "created_at": _now(),
        }
        self._jobs[job_id] = job
        self._save()
        return job, True

    def get_job(self, job_id):
        return self._jobs.get(job_id)

    def update_job(self, job_id, **fields):
        self._jobs[job_id].update(fields)
        self._save()
        return self._jobs[job_id]


def train_and_register(dataset_path, registry_path, job_id, epochs=300):
    """One training run: validate, split, train, evaluate, gate, register.

    Returns a result dict with held-out metrics, the registered version
    (or None when the gate failed), and the train / held-out sizes.
    """
    rows = data_module.read_csv(dataset_path)
    valid, errors = validate_dataset(rows)
    if errors:
        raise ValueError(f"dataset failed schema validation: {errors[:3]}")
    if len(valid) < 20:
        raise ValueError("not enough valid rows to train")
    train_rows, held_out_rows = data_module.split(valid, test_size=0.2, seed=42)
    # Feature engineering is applied identically here, in evaluation
    # and in serving, so the model never sees a different feature set.
    train_rows = engineer_all(train_rows)
    held_out_rows = engineer_all(held_out_rows)
    labels = [r["will_fail_30d"] for r in train_rows]
    model = train_logistic_regression(train_rows, labels, epochs=epochs)
    held_labels = [r["will_fail_30d"] for r in held_out_rows]
    scores = predict_proba(model, held_out_rows)
    predictions = predict(model, held_out_rows)
    metrics = evaluate_metrics(held_labels, predictions, scores)
    result = {
        "job_id": job_id,
        "train_n": len(train_rows),
        "held_out_n": len(held_out_rows),
        "metrics": metrics,
        "gate_passed": passes_gate(metrics),
        "registered_version": None,
    }
    if passes_gate(metrics):
        registry = Registry(registry_path)
        entry = registry.register(model, metrics, notes=f"training job {job_id}")
        result["registered_version"] = entry["version"]
    # Experiment tracking: record the run in SQL next to the registry,
    # so the performance report can query runs instead of re-reading files.
    try:
        db_path = registry_path.replace("registry.json", "experiments.db")
        conn = experiments_module.connect(db_path)
        experiments_module.record_run(
            conn, job_id, result["registered_version"], result["train_n"],
            result["held_out_n"], metrics, result["gate_passed"], _now())
        conn.close()
    except Exception:  # noqa: BLE001 - tracking must never break training
        pass
    return result


def run_job(store, job_id, registry_path, epochs=300):
    """Run a queued job with retry on transient failures."""
    job = store.get_job(job_id)
    if job is None:
        raise ValueError(f"unknown job: {job_id}")
    if job["status"] == "succeeded":
        return job  # idempotent re-run: do not train twice
    store.update_job(job_id, status="running")

    def _attempt():
        store.update_job(job_id, attempts=job.get("attempts", 0) + 1)
        return train_and_register(job["dataset_path"], registry_path, job_id, epochs=epochs)

    try:
        result = retry(_attempt, attempts=3)
    except TransientError as exc:
        store.update_job(job_id, status="failed", error=f"transient failure: {exc}")
        raise
    except ValueError as exc:
        store.update_job(job_id, status="failed", error=str(exc))
        raise
    store.update_job(job_id, status="succeeded", result=result)
    return store.get_job(job_id)
