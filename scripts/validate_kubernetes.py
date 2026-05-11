from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = ROOT / "deploy" / "kubernetes"
REQUIRED_TOP_LEVEL_KEYS = {"apiVersion", "kind", "metadata"}


def validate_document(document: dict, source: Path) -> list[str]:
    errors = []
    if document.get("kind") == "Kustomization":
        for key in ["apiVersion", "kind", "resources"]:
            if key not in document:
                errors.append(f"{source}: missing required Kustomization key {key}")
        return errors

    missing = REQUIRED_TOP_LEVEL_KEYS - set(document)
    if missing:
        errors.append(f"{source}: missing required keys {sorted(missing)}")
    metadata = document.get("metadata") or {}
    if not metadata.get("name"):
        errors.append(f"{source}: metadata.name is required")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate basic Kubernetes manifest structure.")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    args = parser.parse_args()

    errors: list[str] = []
    manifest_files = sorted(args.manifest_dir.glob("*.yaml"))
    if not manifest_files:
        raise FileNotFoundError(f"No Kubernetes manifests found in {args.manifest_dir}")

    document_count = 0
    for path in manifest_files:
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if document is None:
                continue
            document_count += 1
            errors.extend(validate_document(document, path))

    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)

    print(f"Validated {document_count} Kubernetes manifest documents.")


if __name__ == "__main__":
    main()
