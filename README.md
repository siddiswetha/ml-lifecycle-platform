# ML Lifecycle Platform

A small, from-scratch platform for the full lifecycle of one machine
learning model — training, registry, promotion, serving, drift
detection and evaluation — built for applied scientists to use as
shared tooling, not as a one-off notebook.

**Domain:** predictive maintenance for industrial equipment. From
sensor readings (temperature, vibration, pressure, operating hours,
load) the model predicts whether equipment will fail within 30 days.
The model is logistic regression implemented from scratch in the
Python standard library (gradient descent, 9 features: 5 raw sensor
features plus 4 engineered features).

**Stack used, and only what was actually used:** Python (standard
library only for application code), SQL via sqlite3 for experiment
tracking, Docker, Kubernetes manifests, GitHub Actions CI/CD, JSON
structured logging with trace ids, and `/health` / `/metrics`
endpoints for monitoring. Tests use pytest; CI installs pytest and
PyYAML itself. No external ML library, no LLM calls, no cloud
services.

## Services

| Service | Port | What it does |
| --- | --- | --- |
| Registry | 8201 | Model registry: versioned JSON model artifacts, candidate registration, promotion gate, promotion, rollback, history |
| Trainer | 8202 | Training-job queue and orchestration: idempotent jobs by job id, schema validation, training, held-out evaluation, registers a candidate only when it passes the promotion gate, retry on transient failure |
| Serving | 8203 | Prediction serving: live / shadow / canary modes, drift detection against the training baseline (PSI and mean shift), quarantine of invalid or outlier inputs |

Each service has `/health` and `/metrics` endpoints and writes
one-JSON-object-per-line logs with a trace id per request.

## Quick start

```bash
python3 -m pip install pytest pyyaml   # test tools only; app code needs none

# 1. Generate synthetic data (600 rows, seed 42, 480 train / 120 held-out)
python3 scripts/cli.py generate --n 600 --seed 42 --out data/synthetic.csv

# 2. Explore it
python3 scripts/eda.py --dataset data/synthetic.csv --out evals/eda_report.json

# 3. Train (registers v1 only if it passes the promotion gate)
python3 scripts/cli.py train --job-id job-001 --dataset data/synthetic.csv

# 4. Promote v1 to live
python3 scripts/cli.py promote --version v1

# 5. Evaluate the promoted model on the held-out split
python3 scripts/evaluate.py --dataset data/synthetic.csv --out evals/holdout_metrics.json

# 6. Benchmark configurations and write the performance report
python3 scripts/benchmark.py --dataset data/synthetic.csv
python3 scripts/report.py --out evals/performance_report.json

# 7. Predict
python3 scripts/cli.py predict \
  --features '{"temperature": 78, "vibration": 3.1, "pressure": 105, "operating_hours": 4000, "load": 0.8}' \
  --baseline data/synthetic.csv

# 8. Drift-check a batch against the baseline
python3 scripts/cli.py drift-check --baseline data/synthetic.csv --current data/current_shifted.csv
```

Run the services (from the repo root):

```bash
python3 services/registry/app.py   # port 8201
python3 services/trainer/app.py    # port 8202
python3 services/serving/app.py    # port 8203
```

Or with Docker Compose (not run here — see Honest limits):

```bash
docker compose up --build
```

## Feature engineering

Raw features are temperature, vibration, pressure, operating hours
and load. Four derived features are computed identically in training,
evaluation and serving (`ml_platform/features.py`):

- `temp_vibration` — temperature × vibration / 100
- `load_hours` — load × operating hours / 1000
- `pressure_deviation` — pressure − 100
- `vibration_per_load` — vibration / (load + 0.1)

## Promotion gate

A candidate is registered and may be promoted only when its held-out
metrics clear every bar: accuracy ≥ 0.75, precision ≥ 0.65,
recall ≥ 0.65, F1 ≥ 0.65 (`ml_platform/registry.py`).

## Evaluation results (actually measured, 2026-11-09)

Data: 600 synthetic rows, seed 42 — 237 failures (39.5% positive
rate), split 480 train / 120 held-out. Model v1 (300 epochs,
learning rate 0.05), held-out n = 120:

| Metric | Value |
| --- | --- |
| Accuracy | 0.7583 |
| Precision | 0.6667 |
| Recall | 0.6818 |
| F1 | 0.6742 |
| AUC | 0.7925 |
| TP / TN / FP / FN | 30 / 61 / 15 / 14 |

The gate was passed, but only just — accuracy 0.7583 against a 0.75
bar. Benchmarking five configurations (100–600 epochs,
learning rates 0.05–0.10) gave the same F1 0.6742 for every
configuration except 100 epochs (F1 0.6444); the best AUC in the
benchmark was 0.7949 at 100 epochs, with a lower F1. See
`evals/benchmark.json` and `evals/benchmark.md`.

Drift demo: a batch with temperature +15 °C and vibration +1.5 mm/s
(`data/current_shifted.csv`, n = 120) is flagged drifted
(temperature PSI 4.051, mean shift 1.85; vibration PSI 2.561,
mean shift 1.34) with action `quarantine`. A fresh sample from the
same distribution is not flagged.

Exploratory analysis (`evals/eda_report.json`): strongest correlation
with the failure label is vibration (0.3596), then operating hours
(0.2595); no missing values in the synthetic data.

## Tests

```bash
python3 -m pytest -q
```

48 tests, all passing locally: model, schema, feature engineering,
data generation, drift, registry, experiment tracking (SQL), trainer
(idempotence, retry, gate), serving (live / shadow / quarantine),
CLI, and HTTP health checks for all three services.

## Project layout

```
ml_platform/        shared library: model, schema, data, features,
                    drift, telemetry, registry, trainer, serving,
                    experiments (SQL tracking)
services/registry/  registry service (8201) + Dockerfile
services/trainer/   trainer service (8202) + Dockerfile
services/serving/   serving service (8203) + Dockerfile
scripts/            cli.py, evaluate.py, generate_data.py, eda.py,
                    benchmark.py, report.py
evals/              holdout_metrics.json, benchmark.*, eda_report.*,
                    performance_report.*
data/               synthetic.csv, current_shifted.csv, registry.json,
                    jobs.json, experiments.db
k8s/                namespace, configmap, and one Deployment + Service
                    per service
.github/workflows/ test.yml (installs pytest and pyyaml, runs pytest)
docs/design.md      design decisions, failure handling, rollout/rollback
```

## Honest limits

- **Synthetic data only.** 600 rows generated from a known logistic
  function. It is not real equipment data, and the model has never
  seen a real sensor.
- **Small scale.** One logistic model, one 600-row dataset, file-based
  JSON registry and job store, sqlite3 experiment tracking. This is
  not training or serving infrastructure at scale, and it does not
  claim to be: no distributed training, no streaming, no autoscaling.
- **Modest accuracy.** Held-out accuracy 0.7583 and F1 0.6742 — the
  model is a baseline, not a production-grade predictor.
- **Docker and Kubernetes were not run.** Dockerfiles,
  `docker-compose.yml` and the Kubernetes manifests are YAML-valid
  (checked with PyYAML) but were not run on Docker or a real cluster.
- **Drift thresholds are tuned for small batches.** PSI > 0.5 (not
  the usual 0.25) is used because a 40-row sample can exceed 0.25 by
  chance alone; see `docs/design.md`.
- **No authentication or authorisation** on any service endpoint.
  This is a local platform demo, not a hardened service.
