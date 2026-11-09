"""Experiment tracking and performance reporting in SQL (sqlite3).

Standard-library sqlite3 only — no external database is needed for a
small platform. Three tables:

- runs: one row per training job, with held-out metrics as JSON.
- predictions: one row per served prediction (model version, mode,
  probability, prediction, quarantined flag, trace id).
- drift_reports: one row per drift check, with the drifted flag and
  per-feature PSI as JSON.

scripts/report.py queries these tables for the performance report.
"""

import json
import os
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    job_id TEXT PRIMARY KEY,
    model_version TEXT,
    train_n INTEGER,
    held_out_n INTEGER,
    metrics_json TEXT,
    gate_passed INTEGER,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT,
    mode TEXT,
    probability REAL,
    prediction INTEGER,
    quarantined INTEGER,
    trace_id TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS drift_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drifted INTEGER,
    features_json TEXT,
    n_baseline INTEGER,
    n_current INTEGER,
    created_at TEXT
);
"""


def connect(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def record_run(conn, job_id, model_version, train_n, held_out_n, metrics,
               gate_passed, created_at):
    conn.execute(
        "INSERT OR REPLACE INTO runs"
        " (job_id, model_version, train_n, held_out_n, metrics_json,"
        "  gate_passed, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (job_id, model_version, train_n, held_out_n, json.dumps(metrics),
         1 if gate_passed else 0, created_at),
    )
    conn.commit()


def record_prediction(conn, model_version, mode, probability, prediction,
                      quarantined, trace_id, created_at):
    conn.execute(
        "INSERT INTO predictions"
        " (model_version, mode, probability, prediction, quarantined,"
        "  trace_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (model_version, mode, probability, prediction,
         1 if quarantined else 0, trace_id, created_at),
    )
    conn.commit()


def record_drift(conn, drifted, features, n_baseline, n_current, created_at):
    conn.execute(
        "INSERT INTO drift_reports"
        " (drifted, features_json, n_baseline, n_current, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (1 if drifted else 0, json.dumps(features), n_baseline, n_current,
         created_at),
    )
    conn.commit()


def query_runs(conn):
    rows = conn.execute(
        "SELECT job_id, model_version, train_n, held_out_n, metrics_json,"
        " gate_passed, created_at FROM runs ORDER BY created_at"
    ).fetchall()
    return [
        {"job_id": r[0], "model_version": r[1], "train_n": r[2],
         "held_out_n": r[3], "metrics": json.loads(r[4]),
         "gate_passed": bool(r[5]), "created_at": r[6]}
        for r in rows
    ]


def query_prediction_summary(conn):
    row = conn.execute(
        "SELECT COUNT(*), SUM(quarantined), AVG(probability) FROM predictions"
    ).fetchone()
    return {
        "total_predictions": row[0] or 0,
        "quarantined": row[1] or 0,
        "mean_probability": round(row[2], 4) if row[2] is not None else None,
    }


def query_drift_summary(conn):
    row = conn.execute(
        "SELECT COUNT(*), SUM(drifted) FROM drift_reports"
    ).fetchone()
    return {"total_drift_checks": row[0] or 0, "drifted": row[1] or 0}
