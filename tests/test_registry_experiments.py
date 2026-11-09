import os

from ml_platform import experiments as experiments_module
from ml_platform.model import train_logistic_regression
from ml_platform.registry import Registry, passes_gate
from ml_platform import data as data_module
from ml_platform.features import engineer_all


def _model_and_metrics():
    rows = data_module.generate_synthetic(n=200, seed=7)
    train, held = data_module.split(rows, test_size=0.2, seed=7)
    train = engineer_all(train)
    held = engineer_all(held)
    model = train_logistic_regression(train, [r["will_fail_30d"] for r in train], epochs=100)
    from ml_platform.model import evaluate_metrics, predict, predict_proba
    metrics = evaluate_metrics(
        [r["will_fail_30d"] for r in held],
        predict(model, held), predict_proba(model, held))
    return model, metrics


def test_gate_rejects_weak_metrics():
    assert passes_gate({"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9}) is True
    assert passes_gate({"accuracy": 0.5, "precision": 0.9, "recall": 0.9, "f1": 0.9}) is False
    assert passes_gate({}) is False


def test_registry_register_and_promote(tmp_path):
    registry = Registry(str(tmp_path / "registry.json"))
    model, metrics = _model_and_metrics()
    # Use metrics that pass the gate for the unit test, regardless of
    # the small training set's exact held-out result.
    metrics = {"accuracy": 0.85, "precision": 0.8, "recall": 0.8, "f1": 0.8, "n": 40}
    entry = registry.register(model, metrics, notes="test")
    assert entry["version"] == "v1"
    assert entry["status"] == "candidate"
    assert registry.get_current() is None
    registry.promote("v1")
    assert registry.get_current()["version"] == "v1"


def test_registry_rejects_failing_gate(tmp_path):
    registry = Registry(str(tmp_path / "registry.json"))
    model, _ = _model_and_metrics()
    try:
        registry.register(model, {"accuracy": 0.1, "precision": 0.1, "recall": 0.1, "f1": 0.1})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_registry_rollback(tmp_path):
    registry = Registry(str(tmp_path / "registry.json"))
    model, _ = _model_and_metrics()
    good = {"accuracy": 0.85, "precision": 0.8, "recall": 0.8, "f1": 0.8, "n": 40}
    registry.register(model, good)
    registry.register(model, good)
    registry.promote("v1")
    registry.promote("v2")
    rolled = registry.rollback()
    assert rolled["version"] == "v1"
    assert registry.get_current()["version"] == "v1"


def test_registry_rollback_without_previous_fails(tmp_path):
    registry = Registry(str(tmp_path / "registry.json"))
    try:
        registry.rollback()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_experiments_sql_round_trip(tmp_path):
    db = str(tmp_path / "experiments.db")
    conn = experiments_module.connect(db)
    experiments_module.record_run(conn, "job-1", "v1", 160, 40,
                                  {"accuracy": 0.85}, True, "2026-01-01T00:00:00Z")
    experiments_module.record_prediction(conn, "v1", "live", 0.7, 1, False, "abc", "2026-01-01T00:00:00Z")
    experiments_module.record_drift(conn, False, {"temperature": {"psi": 0.01}}, 100, 20, "2026-01-01T00:00:00Z")
    runs = experiments_module.query_runs(conn)
    assert len(runs) == 1
    assert runs[0]["job_id"] == "job-1"
    summary = experiments_module.query_prediction_summary(conn)
    assert summary["total_predictions"] == 1
    assert summary["quarantined"] == 0
    drift = experiments_module.query_drift_summary(conn)
    assert drift["total_drift_checks"] == 1
    conn.close()
    assert os.path.exists(db)
