"""Trainer service (port 8202).

Training-job queue and orchestration. Jobs are idempotent by job id,
validated against the feature schema, trained, evaluated on a held-out
split, and registered as candidates only when they pass the promotion
gate. State lives in JSON files (JOBS_PATH, REGISTRY_PATH).
Standard library only.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml_platform.telemetry import Metrics, log_event, new_trace_id  # noqa: E402
from ml_platform.trainer import JobStore, TransientError, run_job  # noqa: E402

PORT = int(os.environ.get("PORT", "8202"))
JOBS_PATH = os.environ.get("JOBS_PATH", "data/jobs.json")
REGISTRY_PATH = os.environ.get("REGISTRY_PATH", "data/registry.json")
METRICS = Metrics()


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
        trace_id = new_trace_id()
        METRICS.increment("requests")
        store = JobStore(JOBS_PATH)
        if self.path == "/health":
            self._send(200, {"status": "ok", "service": "trainer"})
        elif self.path == "/metrics":
            self._send(200, {"service": "trainer", "counters": METRICS.snapshot()})
        elif self.path == "/jobs":
            self._send(200, {"jobs": list(store._jobs.values())})
        elif self.path.startswith("/jobs/"):
            job = store.get_job(self.path.split("/")[-1])
            if job is None:
                self._send(404, {"error": "unknown job", "trace_id": trace_id})
            else:
                self._send(200, job)
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
        store = JobStore(JOBS_PATH)
        try:
            if self.path == "/jobs":
                job, created = store.create_job(body["job_id"], body["dataset_path"])
                log_event("job_created", trace_id, job_id=job["job_id"], created=created)
                self._send(201 if created else 200, job)
            elif self.path.startswith("/jobs/") and self.path.endswith("/run"):
                job_id = self.path.split("/")[2]
                job = run_job(store, job_id, REGISTRY_PATH,
                              epochs=int(body.get("epochs", 300)))
                log_event("job_succeeded", trace_id, job_id=job_id)
                self._send(200, job)
            else:
                self._send(404, {"error": "not found", "trace_id": trace_id})
        except TransientError as exc:
            self._send(502, {"error": str(exc), "trace_id": trace_id})
        except (KeyError, ValueError) as exc:
            self._send(400, {"error": str(exc), "trace_id": trace_id})


if __name__ == "__main__":
    log_event("service_start", service="trainer", port=PORT)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
