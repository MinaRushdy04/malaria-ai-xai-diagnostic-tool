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
        filename_hash = _sha256_hex(path.name.lower().encode("utf-8"))[:16]
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

    return status


def read_recent_logs(limit: int = 20) -> list[dict[str, Any]]:
    if not SQLITE_LOG_PATH.exists():
        return []

    with sqlite3.connect(SQLITE_LOG_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM predictions ORDER BY timestamp_utc DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def summarize_logs(limit: int = 200) -> dict[str, Any]:
    rows = read_recent_logs(limit=limit)
    if not rows:
        return {
            "total_predictions": 0,
            "review_rate": 0.0,
            "validation_warning_rate": 0.0,
            "quality_pass_rate": 0.0,
            "avg_focus_score": 0.0,
            "avg_brightness": 0.0,
            "class_counts": {},
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

    return {
        "total_predictions": total,
        "review_rate": review_count / total,
        "validation_warning_rate": warning_count / total,
        "quality_pass_rate": quality_pass_count / total,
        "avg_focus_score": sum(safe_float(row, "quality_focus_score") for row in rows) / total,
        "avg_brightness": sum(safe_float(row, "quality_brightness_mean") for row in rows) / total,
        "class_counts": class_counts,
    }
