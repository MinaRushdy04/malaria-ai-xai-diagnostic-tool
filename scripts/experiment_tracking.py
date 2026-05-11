from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNS_DIR = ROOT / "runs"


def write_local_run(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[Path] | None = None,
) -> Path:
    run_dir = LOCAL_RUNS_DIR / experiment_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        "metrics": metrics,
        "artifacts": [path.relative_to(ROOT).as_posix() for path in artifacts or [] if path.exists()],
    }
    output_path = run_dir / "run.json"
    output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return output_path


def log_to_mlflow(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[Path] | None = None,
) -> bool:
    try:
        import mlflow
    except ImportError:
        return False

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", str(ROOT / "mlruns")))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))
        for artifact in artifacts or []:
            if artifact.exists():
                mlflow.log_artifact(str(artifact))
    return True


def log_to_wandb(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[Path] | None = None,
) -> bool:
    try:
        import wandb
    except ImportError:
        return False

    mode = os.environ.get("WANDB_MODE", "offline")
    run = wandb.init(project=experiment_name, name=run_name, config=params, mode=mode)
    run.log(metrics)
    for artifact_path in artifacts or []:
        if artifact_path.exists():
            artifact = wandb.Artifact(artifact_path.stem, type="model-artifact")
            artifact.add_file(str(artifact_path))
            run.log_artifact(artifact)
    run.finish()
    return True


def log_experiment(
    backend: str,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[Path] | None = None,
) -> dict[str, Any]:
    local_path = write_local_run(experiment_name, run_name, params, metrics, artifacts)
    result = {"local_run": local_path.relative_to(ROOT).as_posix()}

    if backend == "mlflow":
        result["mlflow_logged"] = log_to_mlflow(experiment_name, run_name, params, metrics, artifacts)
    elif backend == "wandb":
        result["wandb_logged"] = log_to_wandb(experiment_name, run_name, params, metrics, artifacts)
    elif backend not in {"none", "local"}:
        raise ValueError("tracking backend must be one of: none, local, mlflow, wandb")

    return result
