from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "malaria_App" / "malaria_cell_parasite_prediction_model.h5"
DEFAULT_METRICS_PATH = ROOT / "reports" / "evaluation" / "metrics_summary.csv"
DEFAULT_CALIBRATION_PATH = ROOT / "reports" / "calibration" / "calibration_summary.json"
DEFAULT_INTERVALS_PATH = ROOT / "reports" / "confidence_intervals" / "confidence_interval_summary.json"
DEFAULT_ROBUSTNESS_PATH = ROOT / "reports" / "robustness" / "robustness_metrics.csv"
DEFAULT_REGISTRY_PATH = ROOT / "registry" / "model_registry.json"
DEFAULT_MODEL_NAME = "malaria-cell-mobilenetv2"
DEFAULT_VERSION = "mobilenetv2-malaria-cell-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_selected_test_metrics(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        if row.get("split") == "test" and row.get("policy") == "selected":
            return {key: coerce_number(value) for key, value in row.items()}
    raise ValueError("Could not find test/selected row in metrics summary.")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def coerce_number(value: str) -> Any:
    if value is None:
        return value
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except (ValueError, AttributeError):
        return value


def read_robustness(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {
        row["corruption"]: {
            key: coerce_number(value)
            for key, value in row.items()
            if key != "corruption"
        }
        for row in rows
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    metrics = read_selected_test_metrics(args.metrics)
    intervals = read_json(args.confidence_intervals)
    calibration = read_json(args.calibration)
    robustness = read_robustness(args.robustness)

    model_hash = sha256_file(args.model_path)
    relative_model_path = args.model_path.resolve().relative_to(ROOT).as_posix()

    return {
        "registry_schema_version": "1.0",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "name": args.model_name,
            "version": args.version,
            "path": relative_model_path,
            "sha256": model_hash,
            "framework": "tensorflow-keras",
            "architecture": "mobilenetv2-transfer-learning",
            "input": {
                "media_type": "image/jpeg or image/png",
                "shape": [224, 224, 3],
                "color_space": "RGB",
            },
            "output": {
                "raw_score": "sigmoid score for Uninfected",
                "derived_score": "parasitized_score = 1 - raw_uninfected_score",
            },
        },
        "decision_policy": {
            "positive_class": "Parasitized",
            "threshold": metrics["threshold"],
            "review_margin": 0.075,
            "review_rule": "route to review when score is near threshold or input validation warns",
        },
        "dataset_scope": {
            "source": "NIH/NLM malaria cropped cell images",
            "task": "cropped-cell binary classification",
            "limitations": [
                "not full-slide diagnosis",
                "not patient-level aggregation",
                "not parasitemia estimation",
                "not externally validated for clinical deployment",
            ],
        },
        "evaluation": {
            "test_metrics": metrics,
            "confidence_intervals": intervals,
            "calibration": calibration,
            "robustness": robustness,
        },
        "approval": {
            "stage": args.stage,
            "notes": args.notes,
        },
    }


def write_manifest(manifest: dict[str, Any], registry_path: Path) -> Path:
    model_version = manifest["model"]["version"]
    model_manifest_path = registry_path.parent / "models" / model_version / "manifest.json"
    model_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    model_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry = {
        "active_model_version": model_version,
        "models": {
            model_version: {
                "manifest": model_manifest_path.relative_to(ROOT).as_posix(),
                "sha256": manifest["model"]["sha256"],
                "stage": manifest["approval"]["stage"],
                "threshold": manifest["decision_policy"]["threshold"],
            }
        },
    }
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return model_manifest_path


def check_registry(registry_path: Path) -> None:
    registry = read_json(registry_path)
    active_version = registry["active_model_version"]
    manifest_path = ROOT / registry["models"][active_version]["manifest"]
    manifest = read_json(manifest_path)
    model_path = ROOT / manifest["model"]["path"]
    actual_hash = sha256_file(model_path)
    expected_hash = manifest["model"]["sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            f"Model hash mismatch for {active_version}: expected {expected_hash}, got {actual_hash}"
        )
    print(f"Registry check passed for {active_version}: {actual_hash}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or validate the local model registry manifest.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--confidence-intervals", type=Path, default=DEFAULT_INTERVALS_PATH)
    parser.add_argument("--robustness", type=Path, default=DEFAULT_ROBUSTNESS_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--stage", default="academic-demo", choices=["candidate", "academic-demo", "archived"])
    parser.add_argument("--notes", default="Registered from committed evaluation artifacts.")
    parser.add_argument("--check", action="store_true", help="Validate the active manifest without rewriting it.")
    args = parser.parse_args()

    if args.check:
        check_registry(args.registry_path)
        return

    manifest = build_manifest(args)
    manifest_path = write_manifest(manifest, args.registry_path)
    print(f"Registered {args.version} at {manifest_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
