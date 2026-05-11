# Predict Request Contract

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
