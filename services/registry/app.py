"""Registry service (port 8201).

Model registry with versioning, a promotion gate, promotion and
rollback. State lives in a JSON file (REGISTRY_PATH). Standard library
only.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml_platform.registry import Registry  # noqa: E402
from ml_platform.telemetry import Metrics, log_event, new_trace_id  # noqa: E402

PORT = int(os.environ.get("PORT", "8201"))
REGISTRY_PATH = os.environ.get("REGISTRY_PATH", "data/registry.json")
METRICS = Metrics()


def _summary(entry):
    return {
        "version": entry["version"],
        "status": entry["status"],
        "metrics": entry["metrics"],
        "notes": entry["notes"],
        "created_at": entry["created_at"],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep stdout for JSON logs only
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
        trace_id = new_trace_id()
        METRICS.increment("requests")
        if self.path == "/health":
            self._send(200, {"status": "ok", "service": "registry"})
        elif self.path == "/metrics":
            self._send(200, {"service": "registry", "counters": METRICS.snapshot()})
        elif self.path == "/models":
            registry = Registry(REGISTRY_PATH)
            self._send(200, {"models": [_summary(m) for m in registry.list_models()]})
        elif self.path == "/current":
            registry = Registry(REGISTRY_PATH)
            current = registry.get_current()
            self._send(200, {"current": _summary(current) if current else None})
        elif self.path.startswith("/models/"):
            registry = Registry(REGISTRY_PATH)
            entry = registry.get_model(self.path.split("/")[-1])
            if entry is None:
                self._send(404, {"error": "unknown version", "trace_id": trace_id})
            else:
                self._send(200, _summary(entry))
        else:
            self._send(404, {"error": "not found", "trace_id": trace_id})

    def do_POST(self):
        trace_id = new_trace_id()
        METRICS.increment("requests")
        try:
            body = self._read_json()
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON", "trace_id": trace_id})
            return
        registry = Registry(REGISTRY_PATH)
        try:
            if self.path == "/register":
                entry = registry.register(body["model"], body["metrics"], body.get("notes", ""))
                log_event("model_registered", trace_id, version=entry["version"])
                self._send(201, _summary(entry))
            elif self.path == "/promote":
                entry = registry.promote(body["version"])
                log_event("model_promoted", trace_id, version=entry["version"])
                self._send(200, _summary(entry))
            elif self.path == "/rollback":
                entry = registry.rollback()
                log_event("model_rolled_back", trace_id, version=entry["version"])
                self._send(200, _summary(entry))
            else:
                self._send(404, {"error": "not found", "trace_id": trace_id})
        except (KeyError, ValueError) as exc:
            self._send(400, {"error": str(exc), "trace_id": trace_id})


if __name__ == "__main__":
    log_event("service_start", service="registry", port=PORT)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
