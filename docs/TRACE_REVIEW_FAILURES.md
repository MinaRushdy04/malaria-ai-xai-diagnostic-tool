# Traceability, Review, And Failure Handling

This project treats inference as a service workflow, not just a model call. Three engineering axes
are intentionally strengthened here.

## Traceability

Every successful prediction can be linked through:

- `request_id`
- `correlation_id`
- model version
- model file hash
- threshold and review margin
- image content hash
- filename hash
- quality metrics
- operational events

Useful endpoints:

```text
GET /trace/{request_id}
GET /trace/correlation/{correlation_id}
GET /events/recent
```

`/trace/{request_id}` returns the prediction row, linked review feedback, and operational events
for the same request/correlation chain. It also returns a `timeline` array that converts the raw
records into readable case steps:

- input validation
- model inference
- review routing
- explainability status
- reviewer feedback
- operational failures or warnings

`/trace/correlation/{correlation_id}` is useful for rejected requests that never produced a
prediction `request_id`.

## Human Review

The review workflow supports more than a yes/no label:

- reviewer ID
- assigned reviewer or owner
- reviewer decision: `correct`, `incorrect`, `uncertain`, `needs_follow_up`
- lifecycle status: `pending`, `assigned`, `reviewed`, `escalated`, `closed`
- priority: `routine`, `high`, `urgent`
- final label: `parasitized`, `uninfected`, `unknown`, `not_assessable`
- follow-up action: `none`, `repeat_image`, `senior_review`, `add_to_retraining`, `exclude_from_retraining`
- notes

`needs_follow_up` keeps the case in the active queue. Terminal decisions remove the case from the
active queue.

The goal is to represent clinical oversight as a stateful workflow: a case can be assigned,
reviewed, escalated, or closed while remaining traceable through request ID and correlation ID.

## Failure Handling

The service logs operational events for:

- rejected inputs,
- invalid threshold/review policies,
- model-load failures,
- inference failures,
- XAI failures,
- review feedback creation.

Failure events are included in monitoring summaries so operators can see recent warning/error
counts and the last failure stage.

## Service Metrics

The API middleware records lightweight request metrics for every route:

- method
- path
- status code
- elapsed milliseconds
- correlation ID

Monitoring summaries include recent request count, average latency, p95 latency, max latency,
and API error rate. This is intentionally small, but it gives the project an observable service
boundary instead of only model-level metrics.

`GET /monitoring/history` groups recent prediction and request logs into day or hour buckets so
review rate, warning rate, quality pass rate, class mix, latency, and error rate can be watched
over time.

## Provider Explanation

Each prediction response includes a `provider_explanation` object. It turns the raw score,
threshold, quality checks, Grad-CAM status, and review policy into a provider-facing summary:

- why the decision was produced,
- whether uncertainty or image quality should trigger review,
- what the clinician should verify,
- and what the model cannot infer from a cropped cell image.

## Current Boundary

This is still local file/SQLite-backed traceability. A production-grade service would move these
records into durable database tables with migrations, retention policies, access controls, and
separate immutable audit storage.
