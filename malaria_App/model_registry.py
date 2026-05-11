from __future__ import annotations

import json
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
REGISTRY_PATH = ROOT_DIR / "registry" / "model_registry.json"


def read_active_model_record() -> dict[str, Any] | None:
    if not REGISTRY_PATH.exists():
        return None
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    active_version = registry.get("active_model_version")
    if not active_version:
        return None
    record = registry.get("models", {}).get(active_version)
    if not record:
        return None
    manifest_path = ROOT_DIR / record["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "active_model_version": active_version,
        "stage": record.get("stage"),
        "threshold": record.get("threshold"),
        "manifest": record.get("manifest"),
        "sha256": record.get("sha256"),
        "dataset_scope": manifest.get("dataset_scope", {}),
        "decision_policy": manifest.get("decision_policy", {}),
    }
