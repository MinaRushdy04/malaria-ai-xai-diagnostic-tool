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
```

Review feedback is intended for academic workflow demonstration and failure analysis, not
regulated clinical audit.
