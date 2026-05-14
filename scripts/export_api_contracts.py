from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from malaria_App.schemas import (
    HealthResponse,
    MonitoringHistoryResponse,
    MonitoringSummaryResponse,
    PredictionApiResponse,
    ReviewFeedbackRequest,
    TraceBundleResponse,
)


DEFAULT_OUTPUT_DIR = ROOT / "contracts" / "api"


SCHEMAS = {
    "health_response.schema.json": HealthResponse,
    "prediction_response.schema.json": PredictionApiResponse,
    "monitoring_summary_response.schema.json": MonitoringSummaryResponse,
    "monitoring_history_response.schema.json": MonitoringHistoryResponse,
    "review_feedback_request.schema.json": ReviewFeedbackRequest,
    "trace_bundle_response.schema.json": TraceBundleResponse,
}


def write_multipart_contract(output_dir: Path) -> None:
    content = """# Predict Request Contract

`POST /predict` uses `multipart/form-data`.

## Headers

- `X-Correlation-ID`: optional caller-provided trace ID.
- `X-API-Key`: required only when `MALARIA_API_KEY` is configured on the server.

## Form Fields

| Field | Type | Required | Default | Notes |
|---|---|---:|---:|---|
| `file` | binary JPG/PNG | yes | - | Cropped microscopy cell image. |
| `threshold` | float | no | `0.285` | Parasitized-score cutoff. |
| `review_margin` | float | no | `0.075` | Near-threshold band routed to review. |
| `include_xai` | boolean | no | `true` | Generate Grad-CAM when possible. |
| `route_warnings_to_review` | boolean | no | `true` | Route validation/quality warnings. |
| `enable_logging` | boolean | no | `true` | Write local audit-style log row. |

## Response

See `prediction_response.schema.json`.
"""
    (output_dir / "predict_request.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export API response/request contracts as JSON Schema.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, schema_model in SCHEMAS.items():
        schema = schema_model.model_json_schema()
        (args.output_dir / filename).write_text(json.dumps(schema, indent=2), encoding="utf-8")
    write_multipart_contract(args.output_dir)
    print(f"Exported {len(SCHEMAS)} JSON schemas to {args.output_dir}")


if __name__ == "__main__":
    main()
