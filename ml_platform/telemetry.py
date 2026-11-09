"""Structured JSON logging, trace ids and in-process metrics.

Every log line is one JSON object on stdout, so logs can be collected
and searched without a log-parsing library. A trace id is generated per
request and echoed in responses so a single request can be followed
across the registry, trainer and serving services.
"""

import json
import sys
import uuid
from datetime import datetime, timezone


def new_trace_id():
    return uuid.uuid4().hex[:16]


def log_event(event, trace_id=None, level="info", **fields):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        "trace_id": trace_id or new_trace_id(),
    }
    record.update(fields)
    print(json.dumps(record), file=sys.stdout, flush=True)
    return record


class Metrics:
    """Simple counters; each service keeps one and exposes /metrics."""

    def __init__(self):
        self._counters = {}

    def increment(self, name, value=1):
        self._counters[name] = self._counters.get(name, 0) + value

    def snapshot(self):
        return dict(self._counters)

    def reset(self):
        self._counters = {}
