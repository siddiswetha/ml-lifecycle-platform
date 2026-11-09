from ml_platform import data as data_module
from ml_platform.registry import Registry
from ml_platform.serving import drift_check, predict_one
from ml_platform.trainer import JobStore, TransientError, retry, run_job, train_and_register


def _dataset(tmp_path):
    rows = data_module.generate_synthetic(n=600, seed=42)
    path = tmp_path / "synthetic.csv"
    data_module.write_csv(rows, str(path))
    return str(path), rows


def test_retry_succeeds_after_transient_failure():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError("flaky")
        return "ok"

    assert retry(flaky, attempts=3) == "ok"
    assert calls["n"] == 3


def test_retry_gives_up():
    def always():
        raise TransientError("always")

    try:
        retry(always, attempts=2)
    except TransientError:
        return
    raise AssertionError("expected TransientError")


def test_job_creation_is_idempotent(tmp_path):
    store = JobStore(str(tmp_path / "jobs.json"))
    job, created = store.create_job("job-1", "data/synthetic.csv")
    assert created is True
    _, created_again = store.create_job("job-1", "data/synthetic.csv")
    assert created_again is False
    assert store.get_job("job-1")["job_id"] == "job-1"


def test_train_and_register_passes_gate(tmp_path):
    dataset, _ = _dataset(tmp_path)
    registry_path = str(tmp_path / "registry.json")
    result = train_and_register(dataset, registry_path, "job-1", epochs=300)
    assert result["train_n"] == 480
    assert result["held_out_n"] == 120
    assert result["gate_passed"] is True
    assert result["registered_version"] == "v1"
    assert result["metrics"]["accuracy"] >= 0.75


def test_train_rejects_invalid_dataset(tmp_path):
    rows = data_module.generate_synthetic(n=50, seed=42)
    rows[0]["temperature"] = 999.0
    path = tmp_path / "bad.csv"
    data_module.write_csv(rows, str(path))
    try:
        train_and_register(str(path), str(tmp_path / "registry.json"), "job-bad")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_run_job_is_idempotent(tmp_path):
    dataset, _ = _dataset(tmp_path)
    store = JobStore(str(tmp_path / "jobs.json"))
    store.create_job("job-1", dataset)
    first = run_job(store, "job-1", str(tmp_path / "registry.json"), epochs=300)
    second = run_job(store, "job-1", str(tmp_path / "registry.json"), epochs=300)
    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert len(Registry(str(tmp_path / "registry.json")).list_models()) == 1


def test_serving_predict_live(tmp_path):
    dataset, rows = _dataset(tmp_path)
    registry_path = str(tmp_path / "registry.json")
    train_and_register(dataset, registry_path, "job-1", epochs=300)
    Registry(registry_path).promote("v1")
    row = {k: rows[0][k] for k in
           ["temperature", "vibration", "pressure", "operating_hours", "load"]}
    result = predict_one(registry_path, row, baseline_rows=rows)
    assert result["status"] == "ok"
    assert result["model_version"] == "v1"
    assert result["mode"] == "live"
    assert 0.0 <= result["probability_fail_30d"] <= 1.0


def test_serving_quarantines_invalid_row(tmp_path):
    dataset, rows = _dataset(tmp_path)
    registry_path = str(tmp_path / "registry.json")
    train_and_register(dataset, registry_path, "job-1", epochs=300)
    Registry(registry_path).promote("v1")
    result = predict_one(registry_path, {"temperature": 999.0}, baseline_rows=rows)
    assert result["status"] == "quarantined"
    assert result["reason"] == "schema"


def test_serving_unavailable_without_promoted_model(tmp_path):
    result = predict_one(str(tmp_path / "registry.json"),
                         {"temperature": 72.0, "vibration": 2.5, "pressure": 100.0,
                          "operating_hours": 1000.0, "load": 0.6})
    assert result["status"] == "unavailable"


def test_serving_shadow_returns_candidate_probability(tmp_path):
    dataset, rows = _dataset(tmp_path)
    registry_path = str(tmp_path / "registry.json")
    train_and_register(dataset, registry_path, "job-1", epochs=300)
    Registry(registry_path).promote("v1")
    train_and_register(dataset, registry_path, "job-2", epochs=300)
    row = {k: rows[0][k] for k in
           ["temperature", "vibration", "pressure", "operating_hours", "load"]}
    result = predict_one(registry_path, row, baseline_rows=rows, shadow_enabled=True)
    assert result["status"] == "ok"
    assert result["shadow_model_version"] == "v2"
    assert "shadow_probability_fail_30d" in result


def test_drift_check_function(tmp_path):
    _, rows = _dataset(tmp_path)
    result = drift_check(str(tmp_path / "registry.json"), rows[:200], rows)
    assert result["drifted"] is False
    assert result["action"] == "serve"
