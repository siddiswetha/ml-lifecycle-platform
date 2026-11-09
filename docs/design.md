# Design

## Why three services

The lifecycle has three different failure modes, so it has three
services with three separate stores:

- **Registry (8201)** owns model versions. If it is wrong, the wrong
  model goes live, so its writes are gated and its history is kept.
- **Trainer (8202)** owns jobs. Training is slow and retryable; a job
  is a durable record, not a single HTTP call.
- **Serving (8203)** owns predictions. It must answer fast and must
  never serve a prediction for data it cannot trust, so validation,
  outlier checks and quarantine live here, not in the trainer.

The services share a library (`ml_platform/`) for the model, schema,
feature engineering and drift math, so training, evaluation and
serving cannot silently disagree about features. They share state
only through files: a JSON registry, a JSON job store, and a sqlite3
experiment database. That is a deliberate small-scale choice — it
makes the whole system inspectable with `cat` and `sqlite3`, and it
is an honest limit (see README).

## Model and features

Logistic regression from scratch: standardised features, batch
gradient descent, L2 regularisation, 300 epochs, learning rate 0.05.
Weights are plain JSON, so a registry entry is self-contained and
human-readable. Nine features: five raw sensor readings plus four
engineered features (`ml_platform/features.py`). Engineering runs in
exactly one place and is called by the trainer, the evaluator and
serving, so there is no train/serve skew by construction.

## Promotion gate and rollback

The gate (accuracy ≥ 0.75, precision ≥ 0.65, recall ≥ 0.65,
F1 ≥ 0.65) is checked in two places: the trainer registers a
candidate only when the held-out metrics pass it, and the registry
refuses to register or promote a model that fails it. A model that
fails the gate is reported in the job result, not registered, and
not promoted — a failed experiment leaves no live artefact behind.

Promotion moves the live pointer to a candidate and marks the
previous live version `superseded`. Rollback moves the pointer back
to the most recent superseded version. Both actions are appended to
registry history. Rollback was exercised during the local run:
v1 promoted, v2 promoted, rollback returned the live pointer to v1.

## Jobs, idempotence and retry

A training job is created with a client-supplied job id. Creating
the same id twice returns the original job; re-running a succeeded
job returns it without training again. That makes a retried client
request safe. Transient failures (the `TransientError` path) are
retried up to three times; permanent failures — schema errors, too
little data — fail the job immediately and are recorded with the
error message. The job store is a JSON file next to the registry,
so a job id is idempotent against the same registry.

## Serving: shadow, canary, quarantine

- **Live** is the default: the promoted model answers.
- **Canary** is a deterministic hash of the input row, so the same
  reading always takes the same path. A canary percent routes that
  share of traffic to the newest candidate.
- **Shadow** scores the candidate as well and returns its
  probability alongside the live prediction, but never changes the
  returned prediction. Shadow output is for comparison, not for
  serving.
- **Quarantine** has two triggers: schema validation (missing,
  non-numeric, out-of-range features) and outliers (a feature more
  than 3 baseline standard deviations from the baseline mean).
  Quarantined requests get a 422 with the reason, no prediction, and
  are counted in `/metrics` and in the SQL prediction log.

## Drift

Batch drift is PSI plus mean shift per feature, against the training
baseline. PSI uses quantile bins from the baseline, and the drift
bar is PSI > 0.5 or |mean shift| > 0.5. The PSI bar is higher than
the textbook 0.25 on purpose: on the 40-row held-out split used
here, pressure alone scored PSI 0.347 by sampling noise, and a
platform that pages on noise is a platform people learn to ignore.
A genuinely shifted batch (temperature +15 °C, vibration +1.5 mm/s)
scores PSI 4.051 / 2.561 and is quarantined. Drift checks and
predictions are recorded in the SQL experiment store
(`ml_platform/experiments.py`), and `scripts/report.py` queries
that store for the performance report.

## Feature and data workflows

- Schema validation runs before training and before every
  prediction; invalid rows are rejected, never clipped silently.
- `scripts/eda.py` reports class balance, missing values, per-feature
  statistics and correlation with the label.
- `scripts/benchmark.py` trains the same model under five
  configurations on the same split and ranks them by F1.
- `scripts/report.py` builds the performance report from SQL:
  training runs, served predictions, drift checks, current model,
  and the latest held-out metrics.

## Telemetry

Every service logs one JSON object per line with a timestamp, level,
event and trace id. A trace id is generated per HTTP request and
echoed in the response, so a request can be followed across
services in plain text logs. `/metrics` exposes request,
prediction and quarantine counters per service. There is no
external monitoring dependency.

## Rollout

1. Generate data, train, and let the gate decide whether a
   candidate exists at all.
2. Promote the candidate; the previous live version becomes
   superseded, not deleted.
3. Enable shadow mode first and compare candidate probabilities
   against live ones in the logs.
4. Raise the canary percent in small steps; the hash bucketing
   keeps each reading on a stable path.
5. Promote the candidate to live when shadow/canary look healthy,
   or roll back to the superseded version — one call, no redeploy.

## Failure handling

| Failure | Behaviour |
| --- | --- |
| Invalid dataset row | Job fails with the first validation errors; nothing is trained or registered |
| Model fails the gate | Job succeeds with `gate_passed: false`; no candidate is registered |
| No promoted model | Serving returns 503 `unavailable` |
| Invalid / outlier input | Serving returns 422 `quarantined`, no prediction |
| Drifted batch | Drift-check returns `action: quarantine` |
| Transient training failure | Retried up to 3 times, then the job is marked failed |
| Tracking DB unavailable | Training/serving continue; tracking failures are swallowed and never break a request |

## What was deliberately not built

No distributed training, no streaming ingestion, no autoscaling,
no authentication, no external database, and no cloud services.
The registry and job store are files; the experiment store is
sqlite3. Those are the right tools at this size and the wrong tools
at production size, and the README says so.
