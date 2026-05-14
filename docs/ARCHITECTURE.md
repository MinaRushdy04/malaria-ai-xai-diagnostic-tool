# Architecture

This project is organized as a software system first and a notebook-backed experiment second.
The original notebook is kept under `notebooks/` as a reproducibility and exploration artifact,
but the main project behavior lives in Python modules, scripts, tests, reports, and docs.

## Runtime Layers

```text
FastAPI Web Dashboard
    |-- served at /dashboard/
    |-- uploads files to /predict
    |-- reads monitoring summary
    |-- reads trace timelines
    |-- reads active review queue
    `-- submits reviewer feedback

Optional Streamlit Research UI
    |-- Analysis Workbench
    |-- Monitoring
    |-- Review Queue
    |-- Audit Log
    `-- System Notes

FastAPI Service
    |-- GET /health
    |-- POST /predict
    |-- GET /monitoring/summary
    |-- GET /monitoring/history
    |-- GET /trace/{request_id}
    |-- GET /trace/correlation/{correlation_id}
    |-- GET /events/recent
    |-- GET /review/queue
    |-- GET /review/feedback
    `-- POST /review/feedback

Shared Diagnostic Core
    |-- model loading
    |-- image validation
    |-- quality scoring
    |-- preprocessing
    |-- thresholding
    |-- expert-review routing
    |-- Grad-CAM / activation maps
    |-- provider explanation packet
    |-- prediction logging
    |-- reviewer feedback storage
    |-- review lifecycle status
    |-- operational event logging
    |-- API request metric logging
    |-- trace bundle retrieval
    |-- active-learning queue
    `-- monitoring summaries

MLOps Layer
    |-- GitHub Actions CI and release workflows
    |-- model registry manifest
    |-- API data contracts
    |-- Kubernetes manifests
    |-- drift monitor
    |-- retraining orchestrator
    `-- load-test script
```

## Key Files

- `malaria_App/diagnostic_core.py`: shared inference, XAI, validation, logging, and monitoring logic.
- `malaria_App/api.py`: FastAPI service for inference, monitoring, review endpoints, and the static web dashboard.
- `malaria_App/static_dashboard/`: lightweight browser dashboard served by FastAPI.
- `malaria_App/app.py`: optional Streamlit research UI for demos and local inspection.
- `malaria_App/middleware.py`: correlation ID, request timing header, and local API metric middleware.
- `malaria_App/schemas.py`: typed FastAPI response schemas.
- `Makefile`: repeatable local commands for checks, reports, and services.
- `.github/workflows/ci.yml`: CI for compile checks, tests, script smoke checks, and Docker build.
- `.github/workflows/mlops-governance.yml`: validates model registry, API contracts, and Kubernetes manifests.
- `.github/workflows/release-container.yml`: publishes tagged/manual container builds to GHCR.
- `registry/`: active model registry and model manifest.
- `contracts/api/`: JSON Schema API contracts exported from Pydantic models.
- `deploy/kubernetes/`: Kubernetes deployment, service, ingress, secret template, and HPA scaffold.
- `scripts/train_model.py`: script-first training entry point.
- `scripts/drift_monitor.py`: baseline-vs-current drift report generation.
- `scripts/retraining_pipeline.py`: gated retraining orchestration.
- `scripts/load_test.py`: lightweight API load/scalability smoke test.
- `scripts/evaluate_threshold.py`: reproducible threshold and metrics report generation.
- `scripts/predict_image.py`: CLI inference entry point.
- `tests/test_core_safety.py`: safety behavior tests.
- `reports/evaluation/`: committed evaluation artifacts.
- `notebooks/`: exploratory notebook material.

## Design Principle

The model is treated as one component inside a decision-support workflow. The surrounding system
handles input validation, quality warnings, configurable review routing, provider-facing
explanations, logs, human feedback, and monitoring summaries so the output is easier to inspect
and reason about.

The deployed path is API-first. The browser dashboard is a thin client over FastAPI endpoints,
which makes the project easier to containerize, document, test, and replace with a larger frontend
later. Streamlit remains useful for exploration, but it is not the only interface.

## Human-In-The-Loop Review

Predictions marked as high priority are added to an active-learning queue when they are:

- routed to expert review,
- associated with validation warnings,
- associated with quality-gate warnings.

The dashboard lets a reviewer label cases as `correct`, `incorrect`, `uncertain`, or
`needs_follow_up`. Feedback is stored locally in SQLite and can be exported as CSV. This creates
a realistic loop for retrospective failure analysis without claiming clinical deployment.

The review workflow also tracks assignee, priority, and lifecycle status (`pending`, `assigned`,
`reviewed`, `escalated`, or `closed`) so human oversight is represented as a workflow state, not
only as a one-time label.

## Current Scope Boundary

The current dataset contains cropped cell images. The system does not claim full-slide diagnosis,
patient-level aggregation, species identification, or parasitemia quantification.
