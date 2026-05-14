from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_error: str | None
    model_version: str
    model_sha256: str
    default_parasitized_threshold: float
    default_review_margin: float
    auth_required: bool
    registry: dict[str, Any] | None = None
    service_metrics: dict[str, Any] | None = None


class ImageQualityResponse(BaseModel):
    brightness_mean: float
    contrast_std: float
    focus_score: float
    saturation_mean: float
    warnings: list[str]
    passed: bool


class ValidationResponse(BaseModel):
    content_sha256: str
    filename_hash: str | None
    file_extension: str | None
    byte_size: int
    format: str
    mode: str
    width: int
    height: int
    warnings: list[str]
    quality: ImageQualityResponse


class PredictionResponse(BaseModel):
    request_id: str
    correlation_id: str
    model_version: str
    model_sha256: str
    predicted_class: str
    raw_uninfected_score: float
    parasitized_score: float
    threshold: float
    prediction_score: float
    decision_margin: float
    review_margin: float
    review_required: bool
    review_reason: str
    recommendation: str
    gradcam_layer: str | None = None


class XAIResponse(BaseModel):
    gradcam_layer: str | None = None
    xai_error: str | None = None
    gradcam_overlay: str | None = Field(default=None, description="PNG data URI")
    heatmap: str | None = Field(default=None, description="PNG data URI")


class ModelResponse(BaseModel):
    version: str
    sha256: str
    input_size: tuple[int, int]


class PredictionApiResponse(BaseModel):
    prediction: PredictionResponse
    validation: ValidationResponse
    tensor_shape: tuple[int, ...]
    xai: XAIResponse
    model: ModelResponse
    logging: dict[str, Any]


class MonitoringSummaryResponse(BaseModel):
    total_predictions: int
    review_rate: float
    validation_warning_rate: float
    quality_pass_rate: float
    avg_focus_score: float
    avg_brightness: float
    class_counts: dict[str, int]
    failure_event_count: int = 0
    last_failure_stage: str | None = None
    recent_request_count: int = 0
    avg_request_latency_ms: float = 0.0
    p95_request_latency_ms: float = 0.0
    max_request_latency_ms: float = 0.0
    api_error_rate: float = 0.0


class ReviewFeedbackRequest(BaseModel):
    request_id: str
    reviewer_decision: str
    reviewer_id: str = "anonymous"
    final_label: str = "unknown"
    follow_up_action: str = "none"
    reviewer_notes: str = ""


class TraceBundleResponse(BaseModel):
    request_id: str | None = None
    correlation_id: str | None = None
    prediction: dict[str, Any] | None = None
    review_feedback: list[dict[str, Any]]
    events: list[dict[str, Any]]
    timeline: list[dict[str, Any]] = Field(default_factory=list)
