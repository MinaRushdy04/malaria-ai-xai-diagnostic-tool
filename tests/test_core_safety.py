from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from malaria_App.diagnostic_core import (
    DEFAULT_REVIEW_MARGIN,
    PARASITIZED_THRESHOLD,
    assess_image_quality,
    build_review_decision,
    validate_image_bytes,
)
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
