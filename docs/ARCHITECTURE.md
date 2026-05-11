# Architecture

This project is organized as a software system first and a notebook-backed experiment second.
The original notebook is kept under `notebooks/` as a reproducibility and exploration artifact,
but the main project behavior lives in Python modules, scripts, tests, reports, and docs.

## Runtime Layers

```text
Streamlit Dashboard
    |-- Analysis Workbench
    |-- Monitoring
    |-- Audit Log
    `-- System Notes

FastAPI Service
    |-- GET /health
    |-- POST /predict
    `-- GET /monitoring/summary

Shared Diagnostic Core
    |-- model loading
    |-- image validation
    |-- quality scoring
    |-- preprocessing
    |-- thresholding
    |-- expert-review routing
    |-- Grad-CAM / activation maps
    |-- prediction logging
    `-- monitoring summaries
```

## Key Files

- `malaria_App/diagnostic_core.py`: shared inference, XAI, validation, logging, and monitoring logic.
- `malaria_App/app.py`: Streamlit dashboard for demos and human-facing inspection.
- `malaria_App/api.py`: FastAPI service for backend-style inference.
- `malaria_App/middleware.py`: correlation ID and request timing middleware.
- `malaria_App/schemas.py`: typed FastAPI response schemas.
- `scripts/train_model.py`: script-first training entry point.
- `scripts/evaluate_threshold.py`: reproducible threshold and metrics report generation.
- `scripts/predict_image.py`: CLI inference entry point.
- `tests/test_core_safety.py`: safety behavior tests.
- `reports/evaluation/`: committed evaluation artifacts.
- `notebooks/`: exploratory notebook material.

## Design Principle

The model is treated as one component inside a decision-support workflow. The surrounding system
handles input validation, quality warnings, configurable review routing, explainability, logs,
and monitoring summaries so the output is easier to inspect and reason about.

## Current Scope Boundary

The current dataset contains cropped cell images. The system does not claim full-slide diagnosis,
patient-level aggregation, species identification, or parasitemia quantification.
