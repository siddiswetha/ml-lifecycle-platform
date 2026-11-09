"""Serving service (port 8203).

Prediction serving with live / shadow / canary modes, drift checks and
quarantine of invalid or outlier inputs. Loads the promoted model from
the registry (REGISTRY_PATH). Baseline rows for outlier checks can be
supplied per request or via BASELINE_PATH. Standard library only.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml_platform import data as data_module  # noqa: E402
from ml_platform import experiments as experiments_module  # noqa: E402
from ml_platform.serving import drift_check, predict_one  # noqa: E402
from ml_platform.telemetry import Metrics, log_event, new_trace_id  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

PORT = int(os.environ.get("PORT", "8203"))
REGISTRY_PATH = os.environ.get("REGISTRY_PATH", "data/registry.json")
BASELINE_PATH = os.environ.get("BASELINE_PATH", "")
METRICS = Metrics()
CONFIG = {"canary_percent": 0, "shadow_enabled": False}


def _baseline_rows():
    if BASELINE_PATH and os.path.exists(BASELINE_PATH):
        return data_module.read_csv(BASELINE_PATH)
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        METRICS.increment("requests")
        if self.path == "/health":
            self._send(200, {"status": "ok", "service": "serving"})
        elif self.path == "/metrics":
            self._send(200, {"service": "serving", "counters": METRICS.snapshot()})
        elif self.path == "/config":
            self._send(200, CONFIG)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        trace_id = new_trace_id()
        METRICS.increment("requests")
        try:
            body = self._read_json()
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON", "trace_id": trace_id})
            return
        if self.path == "/predict":
            row = body.get("features") or body
            baseline = body.get("baseline") or _baseline_rows()
            result = predict_one(
                REGISTRY_PATH, row, baseline_rows=baseline,
                canary_percent=int(body.get("canary_percent", CONFIG["canary_percent"])),
                shadow_enabled=bool(body.get("shadow_enabled", CONFIG["shadow_enabled"])),
                trace_id=trace_id,
            )
            METRICS.increment("predictions")
            # Experiment tracking: record the served prediction in SQL.
            try:
                db_path = REGISTRY_PATH.replace("registry.json", "experiments.db")
                conn = experiments_module.connect(db_path)
                experiments_module.record_prediction(
                    conn, result.get("model_version"), result.get("mode", "live"),
                    result.get("probability_fail_30d"), result.get("prediction"),
                    result["status"] == "quarantined", trace_id,
                    datetime.now(timezone.utc).isoformat())
                conn.close()
            except Exception:  # noqa: BLE001 - tracking must never break serving
                pass
            if result["status"] == "quarantined":
                METRICS.increment("quarantined")
                log_event("prediction_quarantined", trace_id, reason=result.get("reason"))
                self._send(422, result)
            elif result["status"] == "unavailable":
                self._send(503, result)
            else:
                log_event("prediction_served", trace_id,
                          model_version=result["model_version"], mode=result["mode"])
                self._send(200, result)
        elif self.path == "/drift-check":
            baseline = body.get("baseline_rows") or _baseline_rows() or []
            current = body.get("current_rows") or []
            if not baseline or not current:
                self._send(400, {"error": "baseline_rows and current_rows are required",
                                 "trace_id": trace_id})
                return
            result = drift_check(REGISTRY_PATH, current, baseline)
            try:
                db_path = REGISTRY_PATH.replace("registry.json", "experiments.db")
                conn = experiments_module.connect(db_path)
                experiments_module.record_drift(
                    conn, result["drifted"], result["features"],
                    result["n_baseline"], result["n_current"],
                    datetime.now(timezone.utc).isoformat())
                conn.close()
            except Exception:  # noqa: BLE001 - tracking must never break serving
                pass
            log_event("drift_checked", trace_id, drifted=result["drifted"])
            self._send(200, result)
        elif self.path == "/config":
            if "canary_percent" in body:
                CONFIG["canary_percent"] = int(body["canary_percent"])
            if "shadow_enabled" in body:
                CONFIG["shadow_enabled"] = bool(body["shadow_enabled"])
            self._send(200, CONFIG)
        else:
            self._send(404, {"error": "not found", "trace_id": trace_id})


if __name__ == "__main__":
    log_event("service_start", service="serving", port=PORT)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
