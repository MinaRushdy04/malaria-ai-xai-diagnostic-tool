from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

try:
    from .diagnostic_core import (
        DEFAULT_REVIEW_MARGIN,
        PARASITIZED_THRESHOLD,
        ImageValidationError,
        diagnose_image,
        load_keras_model,
        package_to_api_payload,
        validate_image_bytes,
        write_prediction_log,
    )
except ImportError:
    from diagnostic_core import (
        DEFAULT_REVIEW_MARGIN,
        PARASITIZED_THRESHOLD,
        ImageValidationError,
        diagnose_image,
        load_keras_model,
        package_to_api_payload,
        validate_image_bytes,
        write_prediction_log,
    )


app = FastAPI(
    title="Malaria Cell-Smear AI API",
    version="1.0.0",
    description=(
        "Academic inference API for malaria cell-smear classification with "
        "input validation, Grad-CAM, review routing, and prediction logging."
    ),
)


_MODEL = None
_MODEL_ERROR = None


def get_model():
    global _MODEL, _MODEL_ERROR
    if _MODEL is None and _MODEL_ERROR is None:
        _MODEL, _MODEL_ERROR = load_keras_model()
    if _MODEL_ERROR:
        raise HTTPException(status_code=503, detail=_MODEL_ERROR)
    return _MODEL


@app.get("/health")
def health():
    model, model_error = load_keras_model()
    return {
        "status": "ok" if model is not None and not model_error else "degraded",
        "model_loaded": model is not None and not model_error,
        "model_error": model_error,
        "default_parasitized_threshold": PARASITIZED_THRESHOLD,
        "default_review_margin": DEFAULT_REVIEW_MARGIN,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(PARASITIZED_THRESHOLD),
    review_margin: float = Form(DEFAULT_REVIEW_MARGIN),
    include_xai: bool = Form(True),
    route_warnings_to_review: bool = Form(True),
    enable_logging: bool = Form(True),
):
    model = get_model()
    image_bytes = await file.read()

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
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc), "details": exc.details})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    payload = package_to_api_payload(package)
    payload["logging"] = (
        write_prediction_log(package, source="fastapi") if enable_logging else {"enabled": False}
    )
    return payload
