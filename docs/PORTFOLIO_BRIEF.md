# Portfolio Brief

## Project Positioning

This project is a responsible AI integration prototype for malaria cell-smear analysis. The core
idea is that a healthcare AI project should not stop at a model accuracy number. It should also
define how inputs are validated, how uncertain cases are routed, how predictions are explained,
how failures are inspected, and how model behavior is monitored after inference.

## What It Demonstrates

- ML evaluation beyond accuracy: sensitivity, specificity, ROC-AUC, PR-AUC, calibration, confidence
  intervals, threshold sweeps, and confusion matrices.
- Healthcare-aware decision policy: configurable thresholding and expert-review routing instead of
  unconditional automated decisions.
- Input safety layer: upload validation, RGB normalization, image-size checks, and quality scoring
  for brightness, contrast, focus, and saturation.
- XAI integration: Grad-CAM overlays returned by both the API and dashboard.
- API-first deployment: FastAPI owns inference, validation, logging, monitoring, and reviewer
  feedback; the browser dashboard is a thin client over the same API.
- Audit-style traceability: correlation IDs, request timing, model version, model hash, threshold,
  review reason, image hashes, and quality metrics.
- Human-in-the-loop workflow: review queue and reviewer feedback records for retrospective failure
  analysis.
- Robustness and failure analysis: synthetic corruption tests and false-positive/false-negative
  gallery.
- MLOps literacy: CI/CD workflows, model registry, API contracts, Kubernetes manifests, drift
  monitoring, retraining orchestration, optional experiment tracking, and load testing.

## Honest Scope

The current model works on cropped cell images. It does not perform full-slide diagnosis,
parasitemia estimation, patient-level aggregation, species identification, or clinical decision
making. The strongest technical claim is system integration around a healthcare AI model, not
clinical readiness.

## Strong Next Expansion

The highest-value next step is a field-of-view workflow: detect cells or parasite candidates,
classify each candidate, aggregate results into a case-level report, and route the most suspicious
regions to expert review. That would move the project from image classification toward a more
realistic microscopy workflow.

The strongest way to talk about the MLOps pieces is honest: they are not proof of production
clinical readiness, but they show that the project was designed with deployment, versioning,
monitoring, and review loops in mind.
