# Operations

This document collects the project commands that make the repository usable beyond the notebook.

## Local Checks

```bash
make install
make test
```

The `test` target compiles the application, API, scripts, and tests, then runs the test suite.

## Run Locally

API-first web dashboard:

```bash
make web
```

Open `http://127.0.0.1:8000/dashboard/`.

Optional Streamlit research UI:

```bash
make app
```

## Analysis Reports

Calibration:

```bash
make calibration
```

Confidence intervals:

```bash
make confidence-intervals
```

Robustness:

```bash
make robustness
```

Error gallery:

```bash
make error-gallery
```

## MLOps Utilities

Export API contracts:

```bash
make contracts
```

Validate active model registry:

```bash
python scripts/register_model.py --check
```

Regenerate model registry:

```bash
make register-model
```

Validate Kubernetes manifests:

```bash
make k8s-validate
```

Run a lightweight health endpoint load test:

```bash
make load-test
```

Run drift monitoring after collecting prediction logs:

```bash
make drift
```

Create a dry-run retraining plan:

```bash
make retraining-plan
```

## Docker

Build the image:

```bash
make docker-build
```

Run the API-first web deployment:

```bash
make docker-up
```

Run the optional Streamlit research UI:

```bash
make docker-streamlit
```

## Continuous Integration

GitHub Actions runs:

- dependency installation
- Python compile checks
- test suite
- script help smoke checks
- Docker image build

The CI workflow is defined in `.github/workflows/ci.yml`.

## Human Review Workflow

The local SQLite database stores prediction logs and reviewer feedback. The dashboard exposes
an active-learning queue for cases that are near threshold, fail quality checks, or require
review. The FastAPI service also exposes:

```text
GET  /review/queue
GET  /review/feedback
POST /review/feedback
GET  /events/recent
GET  /trace/{request_id}
GET  /trace/correlation/{correlation_id}
```

Review feedback and operational events support workflow debugging and failure analysis. They are
not regulated clinical audit storage.

## Service Health Metrics

The FastAPI middleware records request-level service metrics into the local SQLite log store and
CSV export:

```text
logs/predictions.sqlite3
logs/api_requests.csv
```

The monitoring endpoint and dashboard summarize recent request count, average latency, p95
latency, max latency, and API error rate. These values are useful for local load testing and for
showing where a production deployment would attach real metrics infrastructure.
