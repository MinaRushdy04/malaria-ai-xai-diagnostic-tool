# MLOps And Operations Layer

This project now includes lightweight MLOps scaffolding. It is intentionally sized for an
academic/portfolio project: the scripts are runnable and the contracts are committed, but this is
not a managed production platform.

## Implemented

- CI checks for Python compile, tests, script smoke checks, and Docker build.
- Governance workflow for model registry validation, API contract export, and Kubernetes manifest validation.
- Release workflow for publishing a Docker image to GHCR on tags or manual dispatch.
- Kubernetes manifests for FastAPI deployment, service, ingress, config, secret template, and HPA scaffold.
- Model registry manifest with model hash, threshold, evaluation metrics, confidence intervals, calibration, and robustness summary.
- API contracts exported from Pydantic schemas.
- Optional API-key protection through `MALARIA_API_KEY` and `X-API-Key`.
- Local/optional MLflow/W&B experiment tracking hooks in the training script.
- Drift monitoring script comparing baseline test predictions against recent logged predictions.
- Gated retraining orchestration script that produces a retraining plan and can execute training/report commands when explicitly enabled.
- Lightweight load testing script for health or prediction endpoints.

## Commands

```bash
make contracts
make register-model
make k8s-validate
make load-test
```

Drift monitoring requires recent logs from API/dashboard predictions:

```bash
python scripts/drift_monitor.py
```

Retraining is dry-run by default:

```bash
python scripts/retraining_pipeline.py
```

Execute retraining only when you intentionally want to run the training/report sequence:

```bash
python scripts/retraining_pipeline.py --force --execute-training
```

## Experiment Tracking

The training script writes a local run record by default:

```bash
python scripts/train_model.py --epochs 5 --tracking-backend local
```

Optional MLflow:

```bash
python scripts/train_model.py --tracking-backend mlflow
```

Optional W&B:

```bash
python scripts/train_model.py --tracking-backend wandb
```

The MLflow and W&B integrations are optional imports. If those packages are not installed, the
local JSON run record still works.

## Production Boundary

Before calling this production-grade, the following would still be required:

- managed database for audit/review records,
- object storage for larger artifacts,
- real user authentication and authorization,
- secrets management,
- TLS and network policy,
- external validation dataset,
- data retention policy,
- model rollback process,
- monitored deployment in an actual cloud account or Kubernetes cluster.
