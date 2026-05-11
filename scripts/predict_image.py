from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from malaria_App.diagnostic_core import (
    DEFAULT_REVIEW_MARGIN,
    PARASITIZED_THRESHOLD,
    ImageValidationError,
    diagnose_image,
    load_keras_model,
    package_to_api_payload,
    validate_image_bytes,
    write_prediction_log,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one malaria cell-smear prediction from the CLI.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--threshold", type=float, default=PARASITIZED_THRESHOLD)
    parser.add_argument("--review-margin", type=float, default=DEFAULT_REVIEW_MARGIN)
    parser.add_argument("--include-xai", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--correlation-id", default=None)
    args = parser.parse_args()

    image_bytes = args.image_path.read_bytes()
    try:
        validated = validate_image_bytes(image_bytes, filename=args.image_path.name)
    except ImageValidationError as exc:
        raise SystemExit(json.dumps({"error": str(exc), "details": exc.details}, indent=2))

    model, model_error = load_keras_model()
    if model_error:
        raise SystemExit(json.dumps({"error": model_error}, indent=2))

    package = diagnose_image(
        model,
        validated,
        threshold=args.threshold,
        review_margin=args.review_margin,
        include_xai=args.include_xai,
        include_activation=False,
        route_warnings_to_review=True,
        correlation_id=args.correlation_id,
    )
    payload = package_to_api_payload(package)
    if args.log:
        payload["logging"] = write_prediction_log(package, source="cli")
    else:
        payload["logging"] = {"enabled": False}

    if not args.include_xai:
        payload["xai"]["gradcam_overlay"] = None
        payload["xai"]["heatmap"] = None

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
