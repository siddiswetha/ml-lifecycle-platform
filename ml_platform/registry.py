"""Model registry: versioned JSON model artifacts with a promotion gate.

A model is registered as a *candidate* only. It becomes the *current*
(live) model when it is promoted, and promotion is only allowed when
its held-out metrics pass the gate. Rollback returns the live pointer
to the previously promoted version. History is kept so a rollback can
be audited. State is a single JSON file, so the registry is inspectable
without a database.
"""

import json
import os
from datetime import datetime, timezone

# Held-out promotion gate. A candidate must clear every bar.
PROMOTION_GATE = {"accuracy": 0.75, "precision": 0.65, "recall": 0.65, "f1": 0.65}


def passes_gate(metrics):
    """True only when every gate metric is present and at or above its bar."""
    if not metrics:
        return False
    return all(metrics.get(k, 0.0) >= v for k, v in PROMOTION_GATE.items())


def _now():
    return datetime.now(timezone.utc).isoformat()


class Registry:
    def __init__(self, path):
        self.path = path
        self._state = {"models": [], "current_version": None, "history": []}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                self._state = json.load(fh)

    def _save(self):
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2)

    def register(self, model, metrics, notes=""):
        """Register a candidate. Raises ValueError if it fails the gate."""
        if not passes_gate(metrics):
            raise ValueError(f"model fails promotion gate {PROMOTION_GATE}: {metrics}")
        version = f"v{len(self._state['models']) + 1}"
        entry = {
            "version": version,
            "status": "candidate",
            "model": model,
            "metrics": metrics,
            "notes": notes,
            "created_at": _now(),
        }
        self._state["models"].append(entry)
        self._state["history"].append(
            {"action": "register", "version": version, "at": _now()}
        )
        self._save()
        return entry

    def list_models(self):
        return list(self._state["models"])

    def get_model(self, version):
        for entry in self._state["models"]:
            if entry["version"] == version:
                return entry
        return None

    def get_current(self):
        current = self._state.get("current_version")
        return self.get_model(current) if current else None

    def get_candidate(self):
        """Most recent candidate (used for shadow / canary)."""
        candidates = [m for m in self._state["models"] if m["status"] == "candidate"]
        return candidates[-1] if candidates else None

    def promote(self, version):
        entry = self.get_model(version)
        if entry is None:
            raise ValueError(f"unknown version: {version}")
        if not passes_gate(entry["metrics"]):
            raise ValueError(f"version {version} fails promotion gate")
        previous = self._state.get("current_version")
        if previous:
            old = self.get_model(previous)
            if old is not None:
                old["status"] = "superseded"
        entry["status"] = "promoted"
        self._state["current_version"] = version
        self._state["history"].append(
            {"action": "promote", "version": version, "previous": previous, "at": _now()}
        )
        self._save()
        return entry

    def rollback(self):
        """Return the live pointer to the most recent superseded version."""
        current = self.get_current()
        superseded = [m for m in self._state["models"] if m["status"] == "superseded"]
        if not superseded:
            raise ValueError("no previous version to roll back to")
        target = superseded[-1]
        if current is not None:
            current["status"] = "superseded"
        target["status"] = "promoted"
        self._state["current_version"] = target["version"]
        self._state["history"].append(
            {"action": "rollback", "version": target["version"], "at": _now()}
        )
        self._save()
        return target
