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


class ReviewFeedbackRequest(BaseModel):
    request_id: str
    reviewer_decision: str
    reviewer_notes: str = ""
