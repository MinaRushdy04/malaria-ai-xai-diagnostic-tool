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
- reviewer decision: `correct`, `incorrect`, `uncertain`, `needs_follow_up`
- final label: `parasitized`, `uninfected`, `unknown`, `not_assessable`
- follow-up action: `none`, `repeat_image`, `senior_review`, `add_to_retraining`, `exclude_from_retraining`
- notes

`needs_follow_up` keeps the case in the active queue. Terminal decisions remove the case from the
active queue.

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

## Current Boundary

This is still local file/SQLite-backed traceability. A production-grade service would move these
records into durable database tables with migrations, retention policies, access controls, and
separate immutable audit storage.
