import json
import os
import subprocess
import sys
import threading
import urllib.request
import urllib.error

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run_cli(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "scripts/cli.py", *args],
        cwd=cwd or ROOT, capture_output=True, text=True, timeout=120)


def test_cli_generate(tmp_path):
    out = tmp_path / "synthetic.csv"
    result = _run_cli("generate", "--n", "100", "--seed", "42", "--out", str(out))
    assert result.returncode == 0, result.stderr
    assert out.exists()
    payload = json.loads(result.stdout)
    assert payload["n"] == 100


def test_cli_train_promote_predict(tmp_path):
    dataset = tmp_path / "synthetic.csv"
    registry = tmp_path / "registry.json"
    assert _run_cli("generate", "--n", "600", "--seed", "42",
                    "--out", str(dataset)).returncode == 0
    train = _run_cli("--registry", str(registry), "train",
                     "--job-id", "job-1", "--dataset", str(dataset))
    assert train.returncode == 0, train.stderr
    assert json.loads(train.stdout)["status"] == "succeeded"
    promote = _run_cli("--registry", str(registry), "promote", "--version", "v1")
    assert promote.returncode == 0, promote.stderr
    features = json.dumps({"temperature": 78.0, "vibration": 3.1, "pressure": 105.0,
                           "operating_hours": 4000.0, "load": 0.8})
    predict = _run_cli("--registry", str(registry), "predict",
                       "--features", features, "--baseline", str(dataset))
    assert predict.returncode == 0, predict.stderr
    assert json.loads(predict.stdout)["status"] == "ok"


def test_cli_drift_check(tmp_path):
    dataset = tmp_path / "synthetic.csv"
    assert _run_cli("generate", "--n", "100", "--seed", "42",
                    "--out", str(dataset)).returncode == 0
    result = _run_cli("drift-check", "--baseline", str(dataset),
                      "--current", str(dataset))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["drifted"] is False


def _start_service(module_path, port, env):
    import importlib.util
    spec = importlib.util.spec_from_file_location("service_app", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["service_app"] = module
    spec.loader.exec_module(module)
    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(("127.0.0.1", port), module.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_registry_service_health_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("REGISTRY_PATH", str(tmp_path / "registry.json"))
    server = _start_service(os.path.join(ROOT, "services/registry/app.py"), 18201, {})
    try:
        status, body = _get(18201, "/health")
        assert status == 200
        assert body["service"] == "registry"
        status, body = _get(18201, "/metrics")
        assert status == 200
        assert "counters" in body
    finally:
        server.shutdown()


def test_serving_service_health(tmp_path, monkeypatch):
    monkeypatch.setenv("REGISTRY_PATH", str(tmp_path / "registry.json"))
    server = _start_service(os.path.join(ROOT, "services/serving/app.py"), 18203, {})
    try:
        status, body = _get(18203, "/health")
        assert status == 200
        assert body["service"] == "serving"
    finally:
        server.shutdown()


def test_trainer_service_health(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setenv("REGISTRY_PATH", str(tmp_path / "registry.json"))
    server = _start_service(os.path.join(ROOT, "services/trainer/app.py"), 18202, {})
    try:
        status, body = _get(18202, "/health")
        assert status == 200
        assert body["service"] == "trainer"
    finally:
        server.shutdown()
