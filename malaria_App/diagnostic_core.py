from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODEL_PATH = APP_DIR / "malaria_cell_parasite_prediction_model.h5"
LOG_DIR = Path(os.environ.get("MALARIA_LOG_DIR", ROOT_DIR / "logs"))
SQLITE_LOG_PATH = LOG_DIR / "predictions.sqlite3"
CSV_LOG_PATH = LOG_DIR / "predictions.csv"
EVENT_CSV_PATH = LOG_DIR / "events.csv"
API_METRIC_CSV_PATH = LOG_DIR / "api_requests.csv"
FEEDBACK_CSV_EXPORT_PATH = LOG_DIR / "review_feedback_export.csv"

MODEL_VERSION = "mobilenetv2-malaria-cell-v1"
IMG_SIZE = (224, 224)
PARASITIZED_THRESHOLD = 0.285
DEFAULT_REVIEW_MARGIN = 0.075
MIN_IMAGE_SIDE = 50
WARN_IMAGE_SIDE = 100
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MIN_FOCUS_SCORE = 100.0
MIN_CONTRAST_STD = 40.0
MIN_BRIGHTNESS_MEAN = 25.0
MAX_BRIGHTNESS_MEAN = 235.0
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}

BILINEAR_RESAMPLE = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class ImageValidationError(ValueError):
    def __init__(self, message: str, details: list[str] | None = None):
        super().__init__(message)
        self.details = details or [message]


@dataclass
class ImageQualityReport:
    brightness_mean: float
    contrast_std: float
    focus_score: float
    saturation_mean: float
    warnings: list[str]

    @property
    def passed(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"passed": self.passed}


@dataclass
class ValidationMetadata:
    content_sha256: str
    filename_hash: str | None
    file_extension: str | None
    byte_size: int
    format: str
    mode: str
    width: int
    height: int
    warnings: list[str]
    quality: ImageQualityReport

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"quality": self.quality.to_dict()}


@dataclass
class ValidatedImage:
    image: Image.Image
    metadata: ValidationMetadata


@dataclass
class PredictionResult:
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosisPackage:
    result: PredictionResult
    validation: ValidationMetadata
    tensor_shape: tuple[int, ...]
    overlay_image: Image.Image | None = None
    heatmap_image: Image.Image | None = None
    activation_grid: Image.Image | None = None
    activation_layer: str | None = None
    xai_error: str | None = None


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    return _sha256_hex(Path(filename).name.lower().encode("utf-8"))[:16]


@lru_cache(maxsize=1)
def model_sha256() -> str:
    if not MODEL_PATH.exists():
        return "missing"
    digest = hashlib.sha256()
    with MODEL_PATH.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_float(value: float, minimum: float, maximum: float, name: str) -> float:
    value = float(value)
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum:.2f} and {maximum:.2f}.")
    return value


def validate_policy(threshold: float, review_margin: float) -> tuple[float, float]:
    threshold = _safe_float(threshold, 0.01, 0.99, "threshold")
    review_margin = _safe_float(review_margin, 0.0, 0.40, "review_margin")
    return threshold, review_margin


def assess_image_quality(image: Image.Image) -> ImageQualityReport:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = (
        0.299 * image_array[:, :, 0]
        + 0.587 * image_array[:, :, 1]
        + 0.114 * image_array[:, :, 2]
    )
    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    horizontal_detail = np.diff(gray, axis=1)
    vertical_detail = np.diff(gray, axis=0)
    focus_score = float((np.var(horizontal_detail) + np.var(vertical_detail)) / 2.0)

    max_rgb = np.max(image_array, axis=2)
    min_rgb = np.min(image_array, axis=2)
    saturation = np.divide(
        max_rgb - min_rgb,
        np.maximum(max_rgb, 1.0),
        out=np.zeros_like(max_rgb),
        where=max_rgb > 0,
    )
    saturation_mean = float(np.mean(saturation))

    warnings: list[str] = []
    if brightness_mean < MIN_BRIGHTNESS_MEAN:
        warnings.append("Image appears underexposed.")
    elif brightness_mean > MAX_BRIGHTNESS_MEAN:
        warnings.append("Image appears overexposed.")
    if contrast_std < MIN_CONTRAST_STD:
        warnings.append("Image has low contrast; staining or illumination may be poor.")
    if focus_score < MIN_FOCUS_SCORE:
        warnings.append("Image appears low-detail or out of focus.")

    return ImageQualityReport(
        brightness_mean=brightness_mean,
        contrast_std=contrast_std,
        focus_score=focus_score,
        saturation_mean=saturation_mean,
        warnings=warnings,
    )


def validate_image_bytes(image_bytes: bytes, filename: str | None = None) -> ValidatedImage:
    """Validate and normalize an uploaded microscopy image before inference.

    The checks are intentionally practical rather than clinical: file type, size,
    decodability, dimensions, static image format, and RGB normalization.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not image_bytes:
        raise ImageValidationError("The uploaded file is empty.")

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        errors.append(f"File is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")

    extension = None
    filename_hash = None
    if filename:
        path = Path(filename)
        extension = path.suffix.lower()
        filename_hash = hash_filename(filename)
        if extension and extension not in ALLOWED_EXTENSIONS:
            errors.append("Only JPG, JPEG, and PNG images are accepted.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as candidate:
            image_format = candidate.format or "UNKNOWN"
            mode = candidate.mode
            width, height = candidate.size
            is_animated = bool(getattr(candidate, "is_animated", False))
            image = ImageOps.exif_transpose(candidate).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError("The uploaded file could not be decoded as an image.", [str(exc)])

    if image_format not in ALLOWED_FORMATS:
        errors.append("Only JPEG and PNG image encodings are supported.")

    if is_animated:
        errors.append("Animated images are not supported for inference.")

    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        errors.append(f"Image must be at least {MIN_IMAGE_SIDE} x {MIN_IMAGE_SIDE} pixels.")

    if width * height > MAX_IMAGE_PIXELS:
        errors.append("Image dimensions are too large to process safely.")

    if width < WARN_IMAGE_SIDE or height < WARN_IMAGE_SIDE:
        warnings.append(
            "Image is small; resizing may remove cellular detail and should be reviewed cautiously."
        )

    aspect_ratio = width / height if height else 0
    if aspect_ratio < 0.35 or aspect_ratio > 2.85:
        warnings.append(
            "Image aspect ratio is unusual for a cropped single-cell smear image."
        )

    quality = assess_image_quality(image)
    warnings.extend(quality.warnings)

    if errors:
        raise ImageValidationError("Input validation failed.", errors)

    metadata = ValidationMetadata(
        content_sha256=_sha256_hex(image_bytes),
        filename_hash=filename_hash,
        file_extension=extension,
        byte_size=len(image_bytes),
        format=image_format,
        mode=mode,
        width=width,
        height=height,
        warnings=warnings,
        quality=quality,
    )
    return ValidatedImage(image=image, metadata=metadata)


@lru_cache(maxsize=1)
def load_keras_model(model_path: str = str(MODEL_PATH)):
    import tensorflow as tf

    model_path = str(model_path)
    try:
        return tf.keras.models.load_model(model_path, compile=False), None
    except Exception:
        pass

    try:
        base = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        )
        base.trainable = False
        model = tf.keras.Sequential(
            [
                base,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.predict(np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32), verbose=0)

        with h5py.File(model_path, "r") as model_file:
            weights = model_file["model_weights"]
            kernel = np.array(weights["dense"]["sequential"]["dense"]["kernel"])
            bias = np.array(weights["dense"]["sequential"]["dense"]["bias"])
        model.layers[-1].set_weights([kernel, bias])
        return model, None
    except Exception as exc:
        return None, f"Failed to load model: {exc}"


def preprocess_image(image: Image.Image) -> np.ndarray:
    resized = image.resize(IMG_SIZE, BILINEAR_RESAMPLE)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def normalise_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - np.min(values)
    max_value = np.max(values)
    if max_value > 0:
        values = values / max_value
    return values


def get_feature_layer(model):
    base_model = model.layers[0]
    for layer in reversed(base_model.layers):
        try:
            if len(layer.output.shape) == 4:
                return base_model, layer
        except Exception:
            continue
    raise ValueError("No spatial feature layer found for Grad-CAM.")


def apply_classifier_head(model, features):
    x = features
    for layer in model.layers[1:]:
        try:
            x = layer(x, training=False)
        except TypeError:
            x = layer(x)
    return x


def make_gradcam_heatmap(model, img_batch: np.ndarray, predicted_class_index: int):
    import tensorflow as tf

    base_model, feature_layer = get_feature_layer(model)
    feature_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[feature_layer.output, base_model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = feature_model(img_batch, training=False)
        predictions = apply_classifier_head(model, base_outputs)
        uninfected_score = predictions[:, 0]
        target_score = uninfected_score if predicted_class_index == 1 else 1.0 - uninfected_score

    grads = tape.gradient(target_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy(), feature_layer.name


def colorize_heatmap(heatmap: np.ndarray) -> Image.Image:
    heatmap = normalise_map(heatmap)
    red = np.clip(255 * np.minimum(1.0, heatmap * 2.2), 0, 255)
    green = np.clip(255 * np.maximum(0.0, (heatmap - 0.28) / 0.72), 0, 255)
    blue = np.clip(120 * (1.0 - heatmap), 0, 120)
    rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def build_gradcam_images(original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.48):
    heatmap = normalise_map(heatmap)
    heatmap_image = colorize_heatmap(heatmap).resize(original_image.size, BILINEAR_RESAMPLE)
    alpha_mask = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        original_image.size,
        BILINEAR_RESAMPLE,
    )

    base = np.asarray(original_image).astype(np.float32)
    heat = np.asarray(heatmap_image).astype(np.float32)
    mask = (np.asarray(alpha_mask).astype(np.float32) / 255.0) ** 1.35
    overlay = base * (1.0 - alpha * mask[..., None]) + heat * (alpha * mask[..., None])
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return heatmap_image, Image.fromarray(overlay)


def make_activation_grid(model, img_batch: np.ndarray, max_maps: int = 8, tile_size: int = 96):
    import tensorflow as tf

    base_model, feature_layer = get_feature_layer(model)
    activation_model = tf.keras.Model(inputs=base_model.inputs, outputs=feature_layer.output)
    activations = activation_model(img_batch, training=False)[0].numpy()
    channel_strength = np.mean(np.abs(activations), axis=(0, 1))
    selected_channels = np.argsort(channel_strength)[-max_maps:][::-1]

    columns = 4
    rows = int(np.ceil(len(selected_channels) / columns))
    grid = Image.new("RGB", (columns * tile_size, rows * tile_size), color=(14, 18, 25))
    for index, channel in enumerate(selected_channels):
        feature_map = normalise_map(activations[:, :, channel])
        tile = colorize_heatmap(feature_map).resize((tile_size, tile_size), BILINEAR_RESAMPLE)
        x = (index % columns) * tile_size
        y = (index // columns) * tile_size
        grid.paste(tile, (x, y))

    return grid, feature_layer.name


def build_review_decision(
    parasitized_score: float,
    threshold: float,
    review_margin: float,
    validation_warnings: list[str],
    route_warnings_to_review: bool,
) -> tuple[bool, str]:
    margin = abs(parasitized_score - threshold)
    reasons: list[str] = []
    review_required = False
    if margin <= review_margin:
        review_required = True
        reasons.append(
            f"Score is within {review_margin:.3f} of the decision threshold."
        )
    if route_warnings_to_review and validation_warnings:
        review_required = True
        reasons.append("Input validation raised image-quality warnings.")
    if not reasons:
        reasons.append("Score is outside the configured review band.")
    return review_required, " ".join(reasons)


def build_recommendation(predicted_class: str, review_required: bool) -> str:
    if review_required:
        return (
            "Route this case to expert review before acting on the model output. "
            "The AI result should be treated as decision-support evidence only."
        )
    if predicted_class == "Parasitized":
        return (
            "Positive screening flag. Confirm with a qualified microscopy workflow before "
            "any clinical interpretation."
        )
    return (
        "No parasite signal above the configured threshold. Continue expert review if the "
        "specimen, symptoms, or acquisition quality are unclear."
    )


def diagnose_image(
    model,
    validated: ValidatedImage,
    threshold: float = PARASITIZED_THRESHOLD,
    review_margin: float = DEFAULT_REVIEW_MARGIN,
    include_xai: bool = True,
    include_activation: bool = False,
    route_warnings_to_review: bool = True,
    correlation_id: str | None = None,
) -> DiagnosisPackage:
    threshold, review_margin = validate_policy(threshold, review_margin)
    img_batch = preprocess_image(validated.image)

    raw_score = float(model.predict(img_batch, verbose=0)[0][0])
    parasitized_score = 1.0 - raw_score
    predicted_class_index = 0 if parasitized_score >= threshold else 1
    predicted_class = "Parasitized" if predicted_class_index == 0 else "Uninfected"
    prediction_score = parasitized_score if predicted_class == "Parasitized" else raw_score
    decision_margin = abs(parasitized_score - threshold)
    review_required, review_reason = build_review_decision(
        parasitized_score,
        threshold,
        review_margin,
        validated.metadata.warnings,
        route_warnings_to_review,
    )

    result = PredictionResult(
        request_id=str(uuid.uuid4()),
        correlation_id=correlation_id or str(uuid.uuid4()),
        model_version=MODEL_VERSION,
        model_sha256=model_sha256(),
        predicted_class=predicted_class,
        raw_uninfected_score=raw_score,
        parasitized_score=parasitized_score,
        threshold=threshold,
        prediction_score=prediction_score,
        decision_margin=decision_margin,
        review_margin=review_margin,
        review_required=review_required,
        review_reason=review_reason,
        recommendation=build_recommendation(predicted_class, review_required),
    )

    package = DiagnosisPackage(
        result=result,
        validation=validated.metadata,
        tensor_shape=tuple(int(dim) for dim in img_batch.shape),
    )

    if include_xai:
        try:
            heatmap, gradcam_layer_name = make_gradcam_heatmap(model, img_batch, predicted_class_index)
            package.heatmap_image, package.overlay_image = build_gradcam_images(validated.image, heatmap)
            package.result.gradcam_layer = gradcam_layer_name
        except Exception as exc:
            package.xai_error = str(exc)

    if include_activation:
        try:
            package.activation_grid, package.activation_layer = make_activation_grid(model, img_batch)
        except Exception as exc:
            package.xai_error = f"{package.xai_error or ''} Activation maps failed: {exc}".strip()

    return package


def image_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def data_uri_to_image(data_uri: str) -> Image.Image:
    _, encoded = data_uri.split(",", 1) if "," in data_uri else ("", data_uri)
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def build_provider_explanation(package: DiagnosisPackage) -> dict[str, Any]:
    result = package.result
    validation = package.validation
    quality = validation.quality
    near_threshold = result.decision_margin <= result.review_margin
    validation_warnings = validation.warnings

    rationale = [
        (
            f"Parasitized score {result.parasitized_score:.3f} "
            f"was compared with threshold {result.threshold:.3f}."
        ),
        f"Decision margin is {result.decision_margin:.3f}.",
    ]
    if validation_warnings:
        rationale.append("Image validation or quality warnings were present.")
    else:
        rationale.append("Image validation and quality checks did not raise warnings.")
    if result.gradcam_layer:
        rationale.append(f"Grad-CAM was generated from layer {result.gradcam_layer}.")
    elif package.xai_error:
        rationale.append("Grad-CAM generation failed and should not be used for review.")

    if result.review_required:
        action = "Send to human review before using this model output as decision-support evidence."
        uncertainty_level = "review_required"
    elif near_threshold:
        action = "Consider review because the score is close to the configured decision threshold."
        uncertainty_level = "near_threshold"
    elif not quality.passed:
        action = "Repeat or review the image because the quality gate raised concerns."
        uncertainty_level = "quality_concern"
    else:
        action = "No automatic review trigger fired; routine clinical confirmation still applies."
        uncertainty_level = "routine"

    clinician_checks = [
        "Verify that the cell crop is in focus and representative of the smear.",
        "Compare the highlighted region with microscopy morphology rather than treating heatmap intensity as diagnosis.",
        "Confirm the result through the appropriate human microscopy workflow.",
    ]
    if validation_warnings:
        clinician_checks.insert(0, "Inspect the acquisition-quality warnings before trusting the score.")
    if result.predicted_class == "Parasitized":
        clinician_checks.append(
            "If positive, confirm the parasite-like morphology and do not infer parasite burden "
            "from this single crop."
        )

    return {
        "summary": (
            f"Model flagged this image as {result.predicted_class} "
            f"with score {result.prediction_score:.3f}."
        ),
        "decision_basis": rationale,
        "uncertainty_level": uncertainty_level,
        "review_action": action,
        "clinician_checks": clinician_checks,
        "limitations": [
            "This is a cropped-cell classifier, not a patient-level malaria diagnosis.",
            "Grad-CAM shows model attention, not medical causality.",
            "The model does not estimate species, parasitemia, symptom severity, or treatment urgency.",
        ],
        "quality_context": {
            "quality_passed": quality.passed,
            "warnings": validation_warnings,
            "brightness_mean": quality.brightness_mean,
            "contrast_std": quality.contrast_std,
            "focus_score": quality.focus_score,
        },
    }


def package_to_api_payload(package: DiagnosisPackage) -> dict[str, Any]:
    xai = {
        "gradcam_layer": package.result.gradcam_layer,
        "xai_error": package.xai_error,
        "gradcam_overlay": image_to_data_uri(package.overlay_image) if package.overlay_image else None,
        "heatmap": image_to_data_uri(package.heatmap_image) if package.heatmap_image else None,
    }
    return {
        "prediction": package.result.to_dict(),
        "validation": package.validation.to_dict(),
        "tensor_shape": package.tensor_shape,
        "xai": xai,
        "model": {
            "version": package.result.model_version,
            "sha256": package.result.model_sha256,
            "input_size": IMG_SIZE,
        },
        "provider_explanation": build_provider_explanation(package),
    }


LOG_COLUMNS = [
    "timestamp_utc",
    "request_id",
    "correlation_id",
    "model_version",
    "model_sha256",
    "source",
    "predicted_class",
    "raw_uninfected_score",
    "parasitized_score",
    "threshold",
    "prediction_score",
    "decision_margin",
    "review_margin",
    "review_required",
    "review_reason",
    "content_sha256",
    "filename_hash",
    "file_extension",
    "image_format",
    "width",
    "height",
    "byte_size",
    "validation_warnings",
    "quality_brightness_mean",
    "quality_contrast_std",
    "quality_focus_score",
    "quality_saturation_mean",
    "quality_passed",
    "gradcam_layer",
    "xai_error",
]

FEEDBACK_COLUMNS = [
    "timestamp_utc",
    "feedback_id",
    "request_id",
    "correlation_id",
    "reviewer_id",
    "reviewer_decision",
    "final_label",
    "follow_up_action",
    "review_status",
    "assigned_to",
    "priority",
    "reviewer_notes",
    "model_predicted_class",
    "model_parasitized_score",
    "model_threshold",
    "review_required",
    "quality_passed",
    "content_sha256",
]

EVENT_COLUMNS = [
    "timestamp_utc",
    "event_id",
    "correlation_id",
    "request_id",
    "event_type",
    "severity",
    "stage",
    "status",
    "message",
    "details_json",
    "model_version",
    "model_sha256",
    "content_sha256",
    "filename_hash",
]

API_METRIC_COLUMNS = [
    "timestamp_utc",
    "correlation_id",
    "method",
    "path",
    "status_code",
    "elapsed_ms",
]


def _log_record(package: DiagnosisPackage, source: str) -> dict[str, Any]:
    result = package.result
    validation = package.validation
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request_id": result.request_id,
        "correlation_id": result.correlation_id,
        "model_version": result.model_version,
        "model_sha256": result.model_sha256,
        "source": source,
        "predicted_class": result.predicted_class,
        "raw_uninfected_score": result.raw_uninfected_score,
        "parasitized_score": result.parasitized_score,
        "threshold": result.threshold,
        "prediction_score": result.prediction_score,
        "decision_margin": result.decision_margin,
        "review_margin": result.review_margin,
        "review_required": int(result.review_required),
        "review_reason": result.review_reason,
        "content_sha256": validation.content_sha256,
        "filename_hash": validation.filename_hash,
        "file_extension": validation.file_extension,
        "image_format": validation.format,
        "width": validation.width,
        "height": validation.height,
        "byte_size": validation.byte_size,
        "validation_warnings": json.dumps(validation.warnings),
        "quality_brightness_mean": validation.quality.brightness_mean,
        "quality_contrast_std": validation.quality.contrast_std,
        "quality_focus_score": validation.quality.focus_score,
        "quality_saturation_mean": validation.quality.saturation_mean,
        "quality_passed": int(validation.quality.passed),
        "gradcam_layer": result.gradcam_layer,
        "xai_error": package.xai_error,
    }


def _ensure_sqlite_schema(connection: sqlite3.Connection) -> None:
    columns_sql = ", ".join(f"{column} TEXT" for column in LOG_COLUMNS)
    connection.execute(f"CREATE TABLE IF NOT EXISTS predictions ({columns_sql})")
    existing_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(predictions)").fetchall()
    }
    for column in LOG_COLUMNS:
        if column not in existing_columns:
            connection.execute(f"ALTER TABLE predictions ADD COLUMN {column} TEXT")
    connection.commit()


def _ensure_feedback_schema(connection: sqlite3.Connection) -> None:
    columns_sql = ", ".join(f"{column} TEXT" for column in FEEDBACK_COLUMNS)
    connection.execute(f"CREATE TABLE IF NOT EXISTS review_feedback ({columns_sql})")
    existing_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(review_feedback)").fetchall()
    }
    for column in FEEDBACK_COLUMNS:
        if column not in existing_columns:
            connection.execute(f"ALTER TABLE review_feedback ADD COLUMN {column} TEXT")
    connection.commit()


def _ensure_event_schema(connection: sqlite3.Connection) -> None:
    columns_sql = ", ".join(f"{column} TEXT" for column in EVENT_COLUMNS)
    connection.execute(f"CREATE TABLE IF NOT EXISTS system_events ({columns_sql})")
    existing_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(system_events)").fetchall()
    }
    for column in EVENT_COLUMNS:
        if column not in existing_columns:
            connection.execute(f"ALTER TABLE system_events ADD COLUMN {column} TEXT")
    connection.commit()


def _ensure_api_metric_schema(connection: sqlite3.Connection) -> None:
    columns_sql = ", ".join(f"{column} TEXT" for column in API_METRIC_COLUMNS)
    connection.execute(f"CREATE TABLE IF NOT EXISTS api_requests ({columns_sql})")
    existing_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(api_requests)").fetchall()
    }
    for column in API_METRIC_COLUMNS:
        if column not in existing_columns:
            connection.execute(f"ALTER TABLE api_requests ADD COLUMN {column} TEXT")
    connection.commit()


def _json_details(details: dict[str, Any] | list[Any] | None) -> str:
    if details is None:
        return "{}"
    return json.dumps(details, sort_keys=True, default=str)


def write_system_event(
    event_type: str,
    stage: str,
    status: str,
    message: str,
    *,
    severity: str = "info",
    correlation_id: str | None = None,
    request_id: str | None = None,
    details: dict[str, Any] | list[Any] | None = None,
    content_sha256: str | None = None,
    filename_hash: str | None = None,
) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event_id": str(uuid.uuid4()),
        "correlation_id": correlation_id,
        "request_id": request_id,
        "event_type": event_type,
        "severity": severity,
        "stage": stage,
        "status": status,
        "message": message,
        "details_json": _json_details(details),
        "model_version": MODEL_VERSION,
        "model_sha256": model_sha256(),
        "content_sha256": content_sha256,
        "filename_hash": filename_hash,
    }
    event_status = {
        "enabled": True,
        "event_id": record["event_id"],
        "sqlite_path": str(SQLITE_LOG_PATH),
        "csv_path": str(EVENT_CSV_PATH),
        "sqlite": "not_written",
        "csv": "not_written",
    }

    try:
        with sqlite3.connect(SQLITE_LOG_PATH) as connection:
            _ensure_event_schema(connection)
            placeholders = ", ".join("?" for _ in EVENT_COLUMNS)
            connection.execute(
                f"INSERT INTO system_events ({', '.join(EVENT_COLUMNS)}) VALUES ({placeholders})",
                [str(record.get(column, "")) for column in EVENT_COLUMNS],
            )
            connection.commit()
        event_status["sqlite"] = "ok"
    except Exception as exc:
        event_status["sqlite"] = f"failed: {exc}"

    try:
        write_header = not EVENT_CSV_PATH.exists()
        with EVENT_CSV_PATH.open("a", newline="", encoding="utf-8") as event_file:
            writer = csv.DictWriter(event_file, fieldnames=EVENT_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({column: record.get(column, "") for column in EVENT_COLUMNS})
        event_status["csv"] = "ok"
    except Exception as exc:
        event_status["csv"] = f"failed: {exc}"

    return event_status


def write_api_request_metric(
    *,
    correlation_id: str,
    method: str,
    path: str,
    status_code: int,
    elapsed_ms: float,
) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "correlation_id": correlation_id,
        "method": method,
        "path": path,
        "status_code": int(status_code),
        "elapsed_ms": round(float(elapsed_ms), 2),
    }
    status = {
        "enabled": True,
        "sqlite_path": str(SQLITE_LOG_PATH),
        "csv_path": str(API_METRIC_CSV_PATH),
        "sqlite": "not_written",
        "csv": "not_written",
    }

    try:
        with sqlite3.connect(SQLITE_LOG_PATH) as connection:
            _ensure_api_metric_schema(connection)
            placeholders = ", ".join("?" for _ in API_METRIC_COLUMNS)
            connection.execute(
                f"INSERT INTO api_requests ({', '.join(API_METRIC_COLUMNS)}) VALUES ({placeholders})",
                [str(record.get(column, "")) for column in API_METRIC_COLUMNS],
            )
            connection.commit()
        status["sqlite"] = "ok"
    except Exception as exc:
        status["sqlite"] = f"failed: {exc}"

    try:
        write_header = not API_METRIC_CSV_PATH.exists()
        with API_METRIC_CSV_PATH.open("a", newline="", encoding="utf-8") as metric_file:
            writer = csv.DictWriter(metric_file, fieldnames=API_METRIC_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({column: record.get(column, "") for column in API_METRIC_COLUMNS})
        status["csv"] = "ok"
    except Exception as exc:
        status["csv"] = f"failed: {exc}"

    return status


def write_prediction_log(package: DiagnosisPackage, source: str = "streamlit") -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = _log_record(package, source)
    status = {
        "enabled": True,
        "sqlite_path": str(SQLITE_LOG_PATH),
        "csv_path": str(CSV_LOG_PATH),
        "sqlite": "not_written",
        "csv": "not_written",
    }

    try:
        with sqlite3.connect(SQLITE_LOG_PATH) as connection:
            _ensure_sqlite_schema(connection)
            placeholders = ", ".join("?" for _ in LOG_COLUMNS)
            connection.execute(
                f"INSERT INTO predictions ({', '.join(LOG_COLUMNS)}) VALUES ({placeholders})",
                [str(record.get(column, "")) for column in LOG_COLUMNS],
            )
            connection.commit()
        status["sqlite"] = "ok"
    except Exception as exc:
        status["sqlite"] = f"failed: {exc}"

    try:
        write_header = not CSV_LOG_PATH.exists()
        with CSV_LOG_PATH.open("a", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=LOG_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({column: record.get(column, "") for column in LOG_COLUMNS})
        status["csv"] = "ok"
    except Exception as exc:
        status["csv"] = f"failed: {exc}"

    write_system_event(
        event_type="prediction.completed",
        stage="inference",
        status="ok",
        severity="info" if status["sqlite"] == "ok" or status["csv"] == "ok" else "warning",
        message="Prediction completed and logging attempted.",
        correlation_id=package.result.correlation_id,
        request_id=package.result.request_id,
        details={
            "predicted_class": package.result.predicted_class,
            "review_required": package.result.review_required,
            "review_reason": package.result.review_reason,
            "logging_status": status,
            "xai_error": package.xai_error,
        },
        content_sha256=package.validation.content_sha256,
        filename_hash=package.validation.filename_hash,
    )

    if package.xai_error:
        write_system_event(
            event_type="xai.failed",
            stage="explainability",
            status="failed",
            severity="warning",
            message="Grad-CAM or activation-map generation failed.",
            correlation_id=package.result.correlation_id,
            request_id=package.result.request_id,
            details={"xai_error": package.xai_error},
            content_sha256=package.validation.content_sha256,
            filename_hash=package.validation.filename_hash,
        )

    return status


def read_recent_logs(limit: int = 20) -> list[dict[str, Any]]:
    if not SQLITE_LOG_PATH.exists():
        return []

    with sqlite3.connect(SQLITE_LOG_PATH) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_sqlite_schema(connection)
        rows = connection.execute(
            "SELECT * FROM predictions ORDER BY timestamp_utc DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def read_prediction_by_request_id(request_id: str) -> dict[str, Any] | None:
    if not SQLITE_LOG_PATH.exists():
        return None

    with sqlite3.connect(SQLITE_LOG_PATH) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_sqlite_schema(connection)
        row = connection.execute(
            "SELECT * FROM predictions WHERE request_id = ? ORDER BY timestamp_utc DESC LIMIT 1",
            (request_id,),
        ).fetchone()
    return dict(row) if row else None


def read_events_for_trace(
    request_id: str | None = None,
    correlation_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if not SQLITE_LOG_PATH.exists():
        return []

    filters = []
    values: list[str | int] = []
    if request_id:
        filters.append("request_id = ?")
        values.append(request_id)
    if correlation_id:
        filters.append("correlation_id = ?")
        values.append(correlation_id)
    if not filters:
        return []

    where_clause = " OR ".join(filters)
    values.append(int(limit))
    with sqlite3.connect(SQLITE_LOG_PATH) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_event_schema(connection)
        rows = connection.execute(
            f"""
            SELECT *
            FROM system_events
            WHERE {where_clause}
            ORDER BY timestamp_utc DESC
            LIMIT ?
            """,
            tuple(values),
        ).fetchall()
    return [dict(row) for row in rows]


def read_recent_events(limit: int = 100) -> list[dict[str, Any]]:
    if not SQLITE_LOG_PATH.exists():
        return []

    with sqlite3.connect(SQLITE_LOG_PATH) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_event_schema(connection)
        rows = connection.execute(
            "SELECT * FROM system_events ORDER BY timestamp_utc DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def read_recent_api_metrics(limit: int = 200) -> list[dict[str, Any]]:
    if not SQLITE_LOG_PATH.exists():
        return []

    with sqlite3.connect(SQLITE_LOG_PATH) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_api_metric_schema(connection)
        rows = connection.execute(
            "SELECT * FROM api_requests ORDER BY timestamp_utc DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def summarize_api_metrics(limit: int = 200) -> dict[str, Any]:
    rows = read_recent_api_metrics(limit=limit)
    if not rows:
        return {
            "recent_request_count": 0,
            "avg_request_latency_ms": 0.0,
            "p95_request_latency_ms": 0.0,
            "max_request_latency_ms": 0.0,
            "api_error_rate": 0.0,
        }

    latencies: list[float] = []
    error_count = 0
    for row in rows:
        try:
            latencies.append(float(row.get("elapsed_ms") or 0.0))
        except (TypeError, ValueError):
            latencies.append(0.0)
        try:
            status_code = int(float(row.get("status_code") or 0))
        except (TypeError, ValueError):
            status_code = 0
        if status_code >= 500:
            error_count += 1

    latency_array = np.asarray(latencies, dtype=np.float32)
    return {
        "recent_request_count": len(rows),
        "avg_request_latency_ms": float(np.mean(latency_array)),
        "p95_request_latency_ms": float(np.percentile(latency_array, 95)),
        "max_request_latency_ms": float(np.max(latency_array)),
        "api_error_rate": error_count / len(rows),
    }


def _bucket_timestamp(timestamp: str | None, bucket: str) -> str:
    if not timestamp:
        return "unknown"
    bucket = bucket.lower()
    if bucket == "hour":
        return timestamp[:13] + ":00"
    return timestamp[:10]


def summarize_monitoring_history(limit: int = 500, bucket: str = "day") -> dict[str, Any]:
    if bucket not in {"day", "hour"}:
        raise ValueError("bucket must be one of ['day', 'hour']")

    prediction_rows = read_recent_logs(limit=limit)
    api_rows = read_recent_api_metrics(limit=limit)
    grouped: dict[str, dict[str, Any]] = {}

    def entry(bucket_key: str) -> dict[str, Any]:
        return grouped.setdefault(
            bucket_key,
            {
                "bucket": bucket_key,
                "total_predictions": 0,
                "review_count": 0,
                "validation_warning_count": 0,
                "quality_pass_count": 0,
                "api_request_count": 0,
                "api_error_count": 0,
                "latencies_ms": [],
                "class_counts": {},
            },
        )

    def safe_bool(value: Any) -> bool:
        return str(value).lower() in {"1", "true", "yes"}

    def safe_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    for row in prediction_rows:
        item = entry(_bucket_timestamp(row.get("timestamp_utc"), bucket))
        item["total_predictions"] += 1
        item["review_count"] += int(safe_bool(row.get("review_required")))
        item["validation_warning_count"] += int(row.get("validation_warnings") not in {"", "[]", None})
        item["quality_pass_count"] += int(safe_bool(row.get("quality_passed")))
        label = str(row.get("predicted_class") or "unknown")
        item["class_counts"][label] = item["class_counts"].get(label, 0) + 1

    for row in api_rows:
        item = entry(_bucket_timestamp(row.get("timestamp_utc"), bucket))
        item["api_request_count"] += 1
        try:
            status_code = int(float(row.get("status_code") or 0))
        except (TypeError, ValueError):
            status_code = 0
        item["api_error_count"] += int(status_code >= 500)
        item["latencies_ms"].append(safe_float(row.get("elapsed_ms")))

    buckets: list[dict[str, Any]] = []
    for item in grouped.values():
        total = item["total_predictions"]
        request_count = item["api_request_count"]
        latencies = np.asarray(item["latencies_ms"], dtype=np.float32)
        buckets.append(
            {
                "bucket": item["bucket"],
                "total_predictions": total,
                "review_rate": item["review_count"] / total if total else 0.0,
                "validation_warning_rate": (
                    item["validation_warning_count"] / total if total else 0.0
                ),
                "quality_pass_rate": item["quality_pass_count"] / total if total else 0.0,
                "class_counts": item["class_counts"],
                "api_request_count": request_count,
                "api_error_rate": item["api_error_count"] / request_count if request_count else 0.0,
                "p95_request_latency_ms": (
                    float(np.percentile(latencies, 95)) if len(latencies) else 0.0
                ),
            }
        )

    return {
        "bucket": bucket,
        "limit": int(limit),
        "items": sorted(buckets, key=lambda row: row["bucket"], reverse=True),
    }


def read_active_learning_queue(limit: int = 50) -> list[dict[str, Any]]:
    if not SQLITE_LOG_PATH.exists():
        return []

    with sqlite3.connect(SQLITE_LOG_PATH) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_sqlite_schema(connection)
        _ensure_feedback_schema(connection)
        rows = connection.execute(
            """
            SELECT p.*
            FROM predictions p
            LEFT JOIN review_feedback f
              ON p.request_id = f.request_id
             AND LOWER(COALESCE(f.reviewer_decision, '')) IN ('correct', 'incorrect', 'uncertain')
            WHERE f.request_id IS NULL
              AND (
                LOWER(COALESCE(p.review_required, '0')) IN ('1', 'true', 'yes')
                OR LOWER(COALESCE(p.quality_passed, '1')) IN ('0', 'false', 'no')
                OR COALESCE(p.validation_warnings, '') NOT IN ('', '[]')
              )
            ORDER BY p.timestamp_utc DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def write_review_feedback(
    prediction_row: dict[str, Any],
    reviewer_decision: str,
    reviewer_notes: str = "",
    reviewer_id: str = "anonymous",
    final_label: str = "unknown",
    follow_up_action: str = "none",
    review_status: str = "reviewed",
    assigned_to: str = "",
    priority: str = "routine",
) -> dict[str, Any]:
    decision = reviewer_decision.strip().lower()
    allowed_decisions = {"correct", "incorrect", "uncertain", "needs_follow_up"}
    if decision not in allowed_decisions:
        raise ValueError(f"reviewer_decision must be one of {sorted(allowed_decisions)}")
    final_label = final_label.strip().lower() if final_label else "unknown"
    allowed_labels = {"parasitized", "uninfected", "unknown", "not_assessable"}
    if final_label not in allowed_labels:
        raise ValueError(f"final_label must be one of {sorted(allowed_labels)}")
    follow_up_action = follow_up_action.strip().lower() if follow_up_action else "none"
    allowed_actions = {
        "none",
        "repeat_image",
        "senior_review",
        "add_to_retraining",
        "exclude_from_retraining",
    }
    if follow_up_action not in allowed_actions:
        raise ValueError(f"follow_up_action must be one of {sorted(allowed_actions)}")
    review_status = review_status.strip().lower() if review_status else "reviewed"
    allowed_statuses = {"pending", "assigned", "reviewed", "escalated", "closed"}
    if review_status not in allowed_statuses:
        raise ValueError(f"review_status must be one of {sorted(allowed_statuses)}")
    priority = priority.strip().lower() if priority else "routine"
    allowed_priorities = {"routine", "high", "urgent"}
    if priority not in allowed_priorities:
        raise ValueError(f"priority must be one of {sorted(allowed_priorities)}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feedback_id": str(uuid.uuid4()),
        "request_id": prediction_row.get("request_id"),
        "correlation_id": prediction_row.get("correlation_id"),
        "reviewer_id": reviewer_id.strip() or "anonymous",
        "reviewer_decision": decision,
        "final_label": final_label,
        "follow_up_action": follow_up_action,
        "review_status": review_status,
        "assigned_to": assigned_to.strip(),
        "priority": priority,
        "reviewer_notes": reviewer_notes.strip(),
        "model_predicted_class": prediction_row.get("predicted_class"),
        "model_parasitized_score": prediction_row.get("parasitized_score"),
        "model_threshold": prediction_row.get("threshold"),
        "review_required": prediction_row.get("review_required"),
        "quality_passed": prediction_row.get("quality_passed"),
        "content_sha256": prediction_row.get("content_sha256"),
    }

    with sqlite3.connect(SQLITE_LOG_PATH) as connection:
        _ensure_feedback_schema(connection)
        placeholders = ", ".join("?" for _ in FEEDBACK_COLUMNS)
        connection.execute(
            f"INSERT INTO review_feedback ({', '.join(FEEDBACK_COLUMNS)}) VALUES ({placeholders})",
            [str(record.get(column, "")) for column in FEEDBACK_COLUMNS],
        )
        connection.commit()

    write_system_event(
        event_type="review.feedback_created",
        stage="human_review",
        status=review_status,
        severity=(
            "warning"
            if decision in {"incorrect", "needs_follow_up"} or review_status == "escalated"
            else "info"
        ),
        message="Reviewer feedback was recorded.",
        correlation_id=record["correlation_id"],
        request_id=record["request_id"],
        details={
            "feedback_id": record["feedback_id"],
            "reviewer_id": record["reviewer_id"],
            "reviewer_decision": decision,
            "final_label": final_label,
            "follow_up_action": follow_up_action,
            "review_status": review_status,
            "assigned_to": record["assigned_to"],
            "priority": priority,
        },
        content_sha256=record["content_sha256"],
    )

    return {"status": "ok", "feedback_id": record["feedback_id"], "sqlite_path": str(SQLITE_LOG_PATH)}


def read_review_feedback(limit: int = 100) -> list[dict[str, Any]]:
    if not SQLITE_LOG_PATH.exists():
        return []

    with sqlite3.connect(SQLITE_LOG_PATH) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_feedback_schema(connection)
        rows = connection.execute(
            "SELECT * FROM review_feedback ORDER BY timestamp_utc DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def read_review_feedback_for_request(request_id: str) -> list[dict[str, Any]]:
    if not SQLITE_LOG_PATH.exists():
        return []

    with sqlite3.connect(SQLITE_LOG_PATH) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_feedback_schema(connection)
        rows = connection.execute(
            "SELECT * FROM review_feedback WHERE request_id = ? ORDER BY timestamp_utc DESC",
            (request_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def read_trace_bundle(request_id: str) -> dict[str, Any]:
    prediction = read_prediction_by_request_id(request_id)
    correlation_id = prediction.get("correlation_id") if prediction else None
    feedback = read_review_feedback_for_request(request_id)
    events = read_events_for_trace(request_id=request_id, correlation_id=correlation_id, limit=100)
    return {
        "request_id": request_id,
        "correlation_id": correlation_id,
        "prediction": prediction,
        "review_feedback": feedback,
        "events": events,
        "timeline": build_trace_timeline(prediction, feedback, events),
    }


def read_trace_bundle_by_correlation_id(correlation_id: str) -> dict[str, Any]:
    events = read_events_for_trace(correlation_id=correlation_id, limit=100)
    prediction = None
    if SQLITE_LOG_PATH.exists():
        with sqlite3.connect(SQLITE_LOG_PATH) as connection:
            connection.row_factory = sqlite3.Row
            _ensure_sqlite_schema(connection)
            row = connection.execute(
                "SELECT * FROM predictions WHERE correlation_id = ? ORDER BY timestamp_utc DESC LIMIT 1",
                (correlation_id,),
            ).fetchone()
            prediction = dict(row) if row else None
    request_id = prediction.get("request_id") if prediction else None
    feedback = read_review_feedback_for_request(request_id) if request_id else []
    return {
        "request_id": request_id,
        "correlation_id": correlation_id,
        "prediction": prediction,
        "review_feedback": feedback,
        "events": events,
        "timeline": build_trace_timeline(prediction, feedback, events),
    }


def _parse_event_details(details_json: str | None) -> dict[str, Any]:
    if not details_json:
        return {}
    try:
        parsed = json.loads(details_json)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"raw": details_json}


def build_trace_timeline(
    prediction: dict[str, Any] | None,
    feedback: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []

    if prediction:
        prediction_time = prediction.get("timestamp_utc")
        warnings = prediction.get("validation_warnings")
        timeline.extend(
            [
                {
                    "timestamp_utc": prediction_time,
                    "stage": "input_validation",
                    "status": "warning" if warnings not in {"", "[]", None} else "passed",
                    "severity": "warning" if warnings not in {"", "[]", None} else "info",
                    "title": "Input validation completed",
                    "message": "Image decoded, normalized, and checked before inference.",
                    "details": {
                        "format": prediction.get("image_format"),
                        "width": prediction.get("width"),
                        "height": prediction.get("height"),
                        "warnings": warnings,
                    },
                },
                {
                    "timestamp_utc": prediction_time,
                    "stage": "model_inference",
                    "status": "completed",
                    "severity": "info",
                    "title": "Model prediction generated",
                    "message": (
                        f"{prediction.get('predicted_class')} with parasitized score "
                        f"{prediction.get('parasitized_score')} at threshold {prediction.get('threshold')}."
                    ),
                    "details": {
                        "model_version": prediction.get("model_version"),
                        "model_sha256": prediction.get("model_sha256"),
                        "decision_margin": prediction.get("decision_margin"),
                    },
                },
                {
                    "timestamp_utc": prediction_time,
                    "stage": "review_routing",
                    "status": (
                        "review_required"
                        if str(prediction.get("review_required")).lower() in {"1", "true", "yes"}
                        else "not_required"
                    ),
                    "severity": (
                        "warning"
                        if str(prediction.get("review_required")).lower() in {"1", "true", "yes"}
                        else "info"
                    ),
                    "title": "Review policy evaluated",
                    "message": prediction.get("review_reason") or "No review reason recorded.",
                    "details": {
                        "review_margin": prediction.get("review_margin"),
                        "quality_passed": prediction.get("quality_passed"),
                    },
                },
            ]
        )

        if prediction.get("gradcam_layer") or prediction.get("xai_error"):
            timeline.append(
                {
                    "timestamp_utc": prediction_time,
                    "stage": "explainability",
                    "status": "failed" if prediction.get("xai_error") else "completed",
                    "severity": "warning" if prediction.get("xai_error") else "info",
                    "title": "Explainability artifact generated",
                    "message": (
                        prediction.get("xai_error")
                        or f"Grad-CAM layer: {prediction.get('gradcam_layer')}."
                    ),
                    "details": {"gradcam_layer": prediction.get("gradcam_layer")},
                }
            )

    for row in feedback:
        review_status = row.get("review_status") or row.get("reviewer_decision") or "recorded"
        timeline.append(
            {
                "timestamp_utc": row.get("timestamp_utc"),
                "stage": "human_review",
                "status": review_status,
                "severity": (
                    "warning"
                    if row.get("reviewer_decision") in {"incorrect", "needs_follow_up"}
                    or review_status == "escalated"
                    else "info"
                ),
                "title": "Reviewer feedback recorded",
                "message": row.get("reviewer_notes") or "Reviewer submitted a decision.",
                "details": {
                    "reviewer_id": row.get("reviewer_id"),
                    "reviewer_decision": row.get("reviewer_decision"),
                    "final_label": row.get("final_label"),
                    "follow_up_action": row.get("follow_up_action"),
                    "assigned_to": row.get("assigned_to"),
                    "priority": row.get("priority"),
                },
            }
        )

    for row in events:
        timeline.append(
            {
                "timestamp_utc": row.get("timestamp_utc"),
                "stage": row.get("stage"),
                "status": row.get("status"),
                "severity": row.get("severity"),
                "title": row.get("event_type") or "system.event",
                "message": row.get("message") or "",
                "details": _parse_event_details(row.get("details_json")),
            }
        )

    return sorted(
        timeline,
        key=lambda item: str(item.get("timestamp_utc") or ""),
    )


def export_review_feedback_csv(output_path: Path = FEEDBACK_CSV_EXPORT_PATH) -> str:
    rows = read_review_feedback(limit=100_000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FEEDBACK_COLUMNS)
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in FEEDBACK_COLUMNS} for row in rows])
    return str(output_path)


def summarize_review_feedback(limit: int = 500) -> dict[str, Any]:
    rows = read_review_feedback(limit=limit)
    if not rows:
        return {
            "total_reviews": 0,
            "decision_counts": {},
            "status_counts": {},
            "priority_counts": {},
            "model_disagreement_rate": 0.0,
            "pending_follow_up_count": 0,
        }

    decision_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    disagreement_count = 0
    follow_up_count = 0
    for row in rows:
        decision = str(row.get("reviewer_decision") or "unknown")
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        review_status = str(row.get("review_status") or "unknown")
        status_counts[review_status] = status_counts.get(review_status, 0) + 1
        priority = str(row.get("priority") or "unknown")
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
        if decision == "incorrect":
            disagreement_count += 1
        if decision == "needs_follow_up":
            follow_up_count += 1

    return {
        "total_reviews": len(rows),
        "decision_counts": decision_counts,
        "status_counts": status_counts,
        "priority_counts": priority_counts,
        "model_disagreement_rate": disagreement_count / len(rows),
        "pending_follow_up_count": follow_up_count,
    }


def summarize_logs(limit: int = 200) -> dict[str, Any]:
    rows = read_recent_logs(limit=limit)
    api_metrics = summarize_api_metrics(limit=limit)
    if not rows:
        recent_events = read_recent_events(limit=limit)
        failure_events = [
            row for row in recent_events
            if str(row.get("severity")).lower() in {"warning", "error"}
            or str(row.get("status")).lower() == "failed"
        ]
        return {
            "total_predictions": 0,
            "review_rate": 0.0,
            "validation_warning_rate": 0.0,
            "quality_pass_rate": 0.0,
            "avg_focus_score": 0.0,
            "avg_brightness": 0.0,
            "class_counts": {},
            "failure_event_count": len(failure_events),
            "last_failure_stage": failure_events[0].get("stage") if failure_events else None,
            **api_metrics,
        }

    def safe_float(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def safe_bool(row: dict[str, Any], key: str) -> bool:
        value = row.get(key)
        return str(value).lower() in {"1", "true", "yes"}

    total = len(rows)
    review_count = sum(1 for row in rows if safe_bool(row, "review_required"))
    warning_count = sum(1 for row in rows if row.get("validation_warnings") not in {"", "[]", None})
    quality_pass_count = sum(1 for row in rows if safe_bool(row, "quality_passed"))
    class_counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("predicted_class") or "unknown")
        class_counts[label] = class_counts.get(label, 0) + 1
    recent_events = read_recent_events(limit=limit)
    failure_events = [
        row for row in recent_events
        if str(row.get("severity")).lower() in {"warning", "error"}
        or str(row.get("status")).lower() == "failed"
    ]

    return {
        "total_predictions": total,
        "review_rate": review_count / total,
        "validation_warning_rate": warning_count / total,
        "quality_pass_rate": quality_pass_count / total,
        "avg_focus_score": sum(safe_float(row, "quality_focus_score") for row in rows) / total,
        "avg_brightness": sum(safe_float(row, "quality_brightness_mean") for row in rows) / total,
        "class_counts": class_counts,
        "failure_event_count": len(failure_events),
        "last_failure_stage": failure_events[0].get("stage") if failure_events else None,
        **api_metrics,
    }
