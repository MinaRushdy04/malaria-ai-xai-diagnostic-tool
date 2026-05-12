from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from malaria_App.diagnostic_core import (
    DEFAULT_REVIEW_MARGIN,
    PARASITIZED_THRESHOLD,
    DiagnosisPackage,
    ImageQualityReport,
    PredictionResult,
    ValidationMetadata,
    assess_image_quality,
    build_review_decision,
    read_trace_bundle,
    validate_image_bytes,
    write_system_event,
)
import malaria_App.diagnostic_core as core
from malaria_App.api import app as api_app
from malaria_App.middleware import CORRELATION_ID_HEADER, CorrelationIdMiddleware


def png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_validation_accepts_decodable_png_and_reports_quality():
    image = Image.new("RGB", (120, 120), color=(128, 40, 40))
    validated = validate_image_bytes(png_bytes(image), filename="cell.png")

    assert validated.metadata.format == "PNG"
    assert validated.metadata.width == 120
    assert validated.metadata.quality.brightness_mean > 0
    assert isinstance(validated.metadata.quality.warnings, list)


def test_quality_gate_flags_low_detail_images():
    image = Image.new("RGB", (120, 120), color=(128, 128, 128))
    report = assess_image_quality(image)

    assert not report.passed
    assert any("low-detail" in warning or "contrast" in warning for warning in report.warnings)


def test_near_threshold_prediction_routes_to_review():
    review_required, reason = build_review_decision(
        parasitized_score=PARASITIZED_THRESHOLD + 0.01,
        threshold=PARASITIZED_THRESHOLD,
        review_margin=DEFAULT_REVIEW_MARGIN,
        validation_warnings=[],
        route_warnings_to_review=True,
    )

    assert review_required is True
    assert "decision threshold" in reason


def test_middleware_preserves_correlation_id_header():
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    client = TestClient(app)
    response = client.get("/ping", headers={CORRELATION_ID_HEADER: "case-123"})

    assert response.status_code == 200
    assert response.headers[CORRELATION_ID_HEADER] == "case-123"
    assert "X-Process-Time-Ms" in response.headers


def test_api_serves_dashboard_entrypoint():
    client = TestClient(api_app)

    root_response = client.get("/", follow_redirects=False)
    assert root_response.status_code in {307, 308}
    assert root_response.headers["location"] == "/dashboard/"

    dashboard_response = client.get("/dashboard/")
    assert dashboard_response.status_code == 200
    assert "Clinical Review Workbench" in dashboard_response.text


def test_api_key_can_protect_operational_endpoints(monkeypatch):
    monkeypatch.setenv("MALARIA_API_KEY", "secret-test-key")
    client = TestClient(api_app)

    rejected_response = client.get("/monitoring/summary")
    assert rejected_response.status_code == 401

    accepted_response = client.get("/monitoring/summary", headers={"X-API-Key": "secret-test-key"})
    assert accepted_response.status_code == 200


def test_review_feedback_removes_case_from_active_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "LOG_DIR", tmp_path)
    monkeypatch.setattr(core, "SQLITE_LOG_PATH", tmp_path / "predictions.sqlite3")
    monkeypatch.setattr(core, "CSV_LOG_PATH", tmp_path / "predictions.csv")
    monkeypatch.setattr(core, "EVENT_CSV_PATH", tmp_path / "events.csv")
    monkeypatch.setattr(core, "FEEDBACK_CSV_EXPORT_PATH", tmp_path / "review_feedback_export.csv")

    quality = ImageQualityReport(
        brightness_mean=20.0,
        contrast_std=5.0,
        focus_score=2.0,
        saturation_mean=0.1,
        warnings=["Image appears underexposed."],
    )
    validation = ValidationMetadata(
        content_sha256="abc",
        filename_hash="def",
        file_extension=".png",
        byte_size=123,
        format="PNG",
        mode="RGB",
        width=120,
        height=120,
        warnings=["Image appears underexposed."],
        quality=quality,
    )
    result = PredictionResult(
        request_id="req-1",
        correlation_id="case-1",
        model_version="test",
        model_sha256="hash",
        predicted_class="Parasitized",
        raw_uninfected_score=0.6,
        parasitized_score=0.4,
        threshold=0.285,
        prediction_score=0.4,
        decision_margin=0.115,
        review_margin=0.075,
        review_required=True,
        review_reason="Quality warning.",
        recommendation="Review.",
        gradcam_layer=None,
    )
    package = DiagnosisPackage(result=result, validation=validation, tensor_shape=(1, 224, 224, 3))

    core.write_prediction_log(package, source="test")
    queue = core.read_active_learning_queue(limit=10)
    assert len(queue) == 1
    assert queue[0]["request_id"] == "req-1"

    status = core.write_review_feedback(queue[0], reviewer_decision="incorrect", reviewer_notes="bad focus")
    assert status["status"] == "ok"
    assert core.read_active_learning_queue(limit=10) == []

    feedback = core.read_review_feedback(limit=10)
    assert feedback[0]["reviewer_decision"] == "incorrect"
    export_path = core.export_review_feedback_csv()
    assert "review_feedback_export.csv" in export_path


def test_follow_up_review_stays_traceable_and_queued(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "LOG_DIR", tmp_path)
    monkeypatch.setattr(core, "SQLITE_LOG_PATH", tmp_path / "predictions.sqlite3")
    monkeypatch.setattr(core, "CSV_LOG_PATH", tmp_path / "predictions.csv")
    monkeypatch.setattr(core, "EVENT_CSV_PATH", tmp_path / "events.csv")
    monkeypatch.setattr(core, "FEEDBACK_CSV_EXPORT_PATH", tmp_path / "review_feedback_export.csv")

    quality = ImageQualityReport(
        brightness_mean=18.0,
        contrast_std=4.0,
        focus_score=1.0,
        saturation_mean=0.1,
        warnings=["Image appears underexposed."],
    )
    validation = ValidationMetadata(
        content_sha256="trace-content",
        filename_hash="trace-file",
        file_extension=".png",
        byte_size=123,
        format="PNG",
        mode="RGB",
        width=120,
        height=120,
        warnings=["Image appears underexposed."],
        quality=quality,
    )
    result = PredictionResult(
        request_id="req-trace",
        correlation_id="corr-trace",
        model_version="test",
        model_sha256="hash",
        predicted_class="Parasitized",
        raw_uninfected_score=0.55,
        parasitized_score=0.45,
        threshold=0.285,
        prediction_score=0.45,
        decision_margin=0.165,
        review_margin=0.075,
        review_required=True,
        review_reason="Quality warning.",
        recommendation="Review.",
        gradcam_layer=None,
    )
    package = DiagnosisPackage(result=result, validation=validation, tensor_shape=(1, 224, 224, 3))

    core.write_prediction_log(package, source="test")
    queue = core.read_active_learning_queue(limit=10)
    assert queue[0]["request_id"] == "req-trace"

    core.write_review_feedback(
        queue[0],
        reviewer_decision="needs_follow_up",
        reviewer_id="reviewer-a",
        final_label="not_assessable",
        follow_up_action="repeat_image",
        reviewer_notes="blurred field",
    )

    queued_after_follow_up = core.read_active_learning_queue(limit=10)
    assert queued_after_follow_up[0]["request_id"] == "req-trace"

    trace = read_trace_bundle("req-trace")
    assert trace["prediction"]["request_id"] == "req-trace"
    assert trace["review_feedback"][0]["reviewer_id"] == "reviewer-a"
    assert any(event["event_type"] == "review.feedback_created" for event in trace["events"])

    write_system_event(
        event_type="prediction.rejected",
        stage="input_validation",
        status="rejected",
        message="bad input",
        severity="warning",
        correlation_id="corr-failed",
        details={"reason": "empty"},
    )
    failed_trace = core.read_trace_bundle_by_correlation_id("corr-failed")
    assert failed_trace["request_id"] is None
    assert failed_trace["events"][0]["event_type"] == "prediction.rejected"
    summary = core.summarize_logs(limit=50)
    assert summary["failure_event_count"] >= 1
