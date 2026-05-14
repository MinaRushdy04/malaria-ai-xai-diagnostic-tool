from __future__ import annotations

import os
import hashlib
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

try:
    from .diagnostic_core import (
        DEFAULT_REVIEW_MARGIN,
        MODEL_VERSION,
        PARASITIZED_THRESHOLD,
        ImageValidationError,
        diagnose_image,
        hash_filename,
        load_keras_model,
        model_sha256,
        package_to_api_payload,
        read_active_learning_queue,
        read_recent_events,
        read_prediction_by_request_id,
        read_review_feedback,
        read_trace_bundle,
        read_trace_bundle_by_correlation_id,
        summarize_api_metrics,
        summarize_monitoring_history,
        summarize_logs,
        validate_image_bytes,
        write_system_event,
        write_review_feedback,
        write_prediction_log,
    )
    from .middleware import CorrelationIdMiddleware
    from .model_registry import read_active_model_record
    from .schemas import (
        HealthResponse,
        MonitoringSummaryResponse,
        MonitoringHistoryResponse,
        PredictionApiResponse,
        ReviewFeedbackRequest,
        TraceBundleResponse,
    )
except ImportError:
    from diagnostic_core import (
        DEFAULT_REVIEW_MARGIN,
        MODEL_VERSION,
        PARASITIZED_THRESHOLD,
        ImageValidationError,
        diagnose_image,
        hash_filename,
        load_keras_model,
        model_sha256,
        package_to_api_payload,
        read_active_learning_queue,
        read_recent_events,
        read_prediction_by_request_id,
        read_review_feedback,
        read_trace_bundle,
        read_trace_bundle_by_correlation_id,
        summarize_api_metrics,
        summarize_monitoring_history,
        summarize_logs,
        validate_image_bytes,
        write_system_event,
        write_review_feedback,
        write_prediction_log,
    )
    from middleware import CorrelationIdMiddleware
    from model_registry import read_active_model_record
    from schemas import (
        HealthResponse,
        MonitoringSummaryResponse,
        MonitoringHistoryResponse,
        PredictionApiResponse,
        ReviewFeedbackRequest,
        TraceBundleResponse,
    )


app = FastAPI(
    title="Malaria Cell-Smear AI API",
    version="1.0.0",
    description=(
        "Inference API for malaria cell-smear classification with "
        "input validation, Grad-CAM, review routing, and prediction logging."
    ),
)
app.add_middleware(CorrelationIdMiddleware)

STATIC_DASHBOARD_DIR = Path(__file__).resolve().parent / "static_dashboard"


_MODEL = None
_MODEL_ERROR = None


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/dashboard/")


def auth_required() -> bool:
    return bool(os.environ.get("MALARIA_API_KEY"))


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected_api_key = os.environ.get("MALARIA_API_KEY")
    if expected_api_key and x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_model():
    global _MODEL, _MODEL_ERROR
    if _MODEL is None and _MODEL_ERROR is None:
        _MODEL, _MODEL_ERROR = load_keras_model()
    if _MODEL_ERROR:
        raise HTTPException(status_code=503, detail=_MODEL_ERROR)
    return _MODEL


@app.get("/health", response_model=HealthResponse)
def health():
    model, model_error = load_keras_model()
    return {
        "status": "ok" if model is not None and not model_error else "degraded",
        "model_loaded": model is not None and not model_error,
        "model_error": model_error,
        "model_version": MODEL_VERSION,
        "model_sha256": model_sha256(),
        "default_parasitized_threshold": PARASITIZED_THRESHOLD,
        "default_review_margin": DEFAULT_REVIEW_MARGIN,
        "auth_required": auth_required(),
        "registry": read_active_model_record(),
        "service_metrics": summarize_api_metrics(limit=200),
    }


@app.get(
    "/monitoring/summary",
    response_model=MonitoringSummaryResponse,
    dependencies=[Depends(require_api_key)],
)
def monitoring_summary(limit: int = 200):
    return summarize_logs(limit=limit)


@app.get(
    "/monitoring/history",
    response_model=MonitoringHistoryResponse,
    dependencies=[Depends(require_api_key)],
)
def monitoring_history(limit: int = 500, bucket: str = "day"):
    try:
        return summarize_monitoring_history(limit=limit, bucket=bucket)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/review/queue", dependencies=[Depends(require_api_key)])
def review_queue(limit: int = 50):
    return {"items": read_active_learning_queue(limit=limit)}


@app.get("/review/feedback", dependencies=[Depends(require_api_key)])
def review_feedback(limit: int = 100):
    return {"items": read_review_feedback(limit=limit)}


@app.get("/events/recent", dependencies=[Depends(require_api_key)])
def recent_events(limit: int = 100):
    return {"items": read_recent_events(limit=limit)}


@app.get(
    "/trace/{request_id}",
    response_model=TraceBundleResponse,
    dependencies=[Depends(require_api_key)],
)
def trace_request(request_id: str):
    trace = read_trace_bundle(request_id)
    if trace["prediction"] is None and not trace["events"]:
        raise HTTPException(status_code=404, detail="No trace records found for request_id.")
    return trace


@app.get(
    "/trace/correlation/{correlation_id}",
    response_model=TraceBundleResponse,
    dependencies=[Depends(require_api_key)],
)
def trace_correlation(correlation_id: str):
    trace = read_trace_bundle_by_correlation_id(correlation_id)
    if trace["prediction"] is None and not trace["events"]:
        raise HTTPException(status_code=404, detail="No trace records found for correlation_id.")
    return trace


@app.post("/review/feedback", dependencies=[Depends(require_api_key)])
def create_review_feedback(feedback: ReviewFeedbackRequest):
    prediction_row = read_prediction_by_request_id(feedback.request_id)
    if prediction_row is None:
        raise HTTPException(status_code=404, detail="Prediction request_id not found in local logs.")
    return write_review_feedback(
        prediction_row,
        reviewer_decision=feedback.reviewer_decision,
        reviewer_notes=feedback.reviewer_notes,
        reviewer_id=feedback.reviewer_id,
        final_label=feedback.final_label,
        follow_up_action=feedback.follow_up_action,
        review_status=feedback.review_status,
        assigned_to=feedback.assigned_to,
        priority=feedback.priority,
    )


@app.post(
    "/predict",
    response_model=PredictionApiResponse,
    dependencies=[Depends(require_api_key)],
)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Form(PARASITIZED_THRESHOLD),
    review_margin: float = Form(DEFAULT_REVIEW_MARGIN),
    include_xai: bool = Form(True),
    route_warnings_to_review: bool = Form(True),
    enable_logging: bool = Form(True),
):
    image_bytes = await file.read()
    content_sha256 = sha256_bytes(image_bytes)
    filename_hash = hash_filename(file.filename)

    try:
        model = get_model()
    except HTTPException as exc:
        write_system_event(
            event_type="prediction.failed",
            stage="model_load",
            status="failed",
            severity="error",
            message="Model could not be loaded for prediction.",
            correlation_id=request.state.correlation_id,
            details={"status_code": exc.status_code, "detail": exc.detail},
            content_sha256=content_sha256,
            filename_hash=filename_hash,
        )
        raise

    try:
        validated = validate_image_bytes(image_bytes, filename=file.filename)
        package = diagnose_image(
            model,
            validated,
            threshold=threshold,
            review_margin=review_margin,
            include_xai=include_xai,
            include_activation=False,
            route_warnings_to_review=route_warnings_to_review,
            correlation_id=request.state.correlation_id,
        )
    except ImageValidationError as exc:
        write_system_event(
            event_type="prediction.rejected",
            stage="input_validation",
            status="rejected",
            severity="warning",
            message=str(exc),
            correlation_id=request.state.correlation_id,
            details={"details": exc.details, "filename": file.filename},
            content_sha256=content_sha256,
            filename_hash=filename_hash,
        )
        raise HTTPException(status_code=422, detail={"message": str(exc), "details": exc.details})
    except ValueError as exc:
        write_system_event(
            event_type="prediction.rejected",
            stage="policy_validation",
            status="rejected",
            severity="warning",
            message=str(exc),
            correlation_id=request.state.correlation_id,
            details={"filename": file.filename},
            content_sha256=content_sha256,
            filename_hash=filename_hash,
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        write_system_event(
            event_type="prediction.failed",
            stage="inference",
            status="failed",
            severity="error",
            message="Unexpected inference failure.",
            correlation_id=request.state.correlation_id,
            details={"error": str(exc), "filename": file.filename},
            content_sha256=content_sha256,
            filename_hash=filename_hash,
        )
        raise HTTPException(status_code=500, detail="Unexpected inference failure.")

    payload = package_to_api_payload(package)
    payload["logging"] = (
        write_prediction_log(package, source="fastapi") if enable_logging else {"enabled": False}
    )
    return payload


if STATIC_DASHBOARD_DIR.exists():
    app.mount(
        "/dashboard",
        StaticFiles(directory=STATIC_DASHBOARD_DIR, html=True),
        name="dashboard",
    )
