# Operations

This document collects the project commands that make the repository usable beyond the notebook.

## Local Checks

```bash
make install
make test
```

The `test` target compiles the application, API, scripts, and tests, then runs the test suite.

## Run Locally

Dashboard:

```bash
make app
```

FastAPI service:

```bash
make api
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

Run dashboard and API:

```bash
make docker-up
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
