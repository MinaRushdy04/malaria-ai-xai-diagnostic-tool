from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from malaria_App.diagnostic_core import (  # noqa: E402
    DEFAULT_REVIEW_MARGIN,
    PARASITIZED_THRESHOLD,
    IMG_SIZE,
    assess_image_quality,
    build_review_decision,
    load_keras_model,
    preprocess_image,
)


DATASET_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
ZIP_PATH = ROOT / "data" / "raw" / "cell_images.zip"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "robustness"
POSITIVE_CLASS = "Parasitized"
NEGATIVE_CLASS = "Uninfected"


def find_dataset_zip() -> Path | None:
    candidates = [
        ZIP_PATH,
        ROOT.parent / "AI-X-ray-diagnosis-project-xai" / "data" / "raw" / "cell_images.zip",
        ROOT.parent / "AI-X-ray-diagnosis-project" / "data" / "raw" / "cell_images.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def download_dataset_zip() -> Path:
    existing = find_dataset_zip()
    if existing:
        ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
        if existing != ZIP_PATH and not ZIP_PATH.exists():
            shutil.copyfile(existing, ZIP_PATH)
        return ZIP_PATH if ZIP_PATH.exists() else existing

    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    return ZIP_PATH


def collect_samples(zip_path: Path) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            lower = member.lower()
            if not lower.endswith((".png", ".jpg", ".jpeg")):
                continue
            parts = member.replace("\\", "/").split("/")
            if POSITIVE_CLASS in parts:
                samples.append((member, 1))
            elif NEGATIVE_CLASS in parts:
                samples.append((member, 0))
    if not samples:
        raise ValueError(f"No class-labeled images found in {zip_path}")
    return samples


def test_split(samples: list[tuple[str, int]], seed: int) -> list[tuple[str, int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    shuffled = [samples[index] for index in indices]
    train_count = int(len(shuffled) * 0.8)
    val_count = int(len(shuffled) * 0.1)
    return shuffled[train_count + val_count :]


def balanced_subset(samples: list[tuple[str, int]], max_samples: int, seed: int) -> list[tuple[str, int]]:
    if max_samples <= 0 or max_samples >= len(samples):
        return samples
    rng = np.random.default_rng(seed)
    positives = [sample for sample in samples if sample[1] == 1]
    negatives = [sample for sample in samples if sample[1] == 0]
    per_class = max(1, max_samples // 2)
    selected = []
    for group in (positives, negatives):
        count = min(per_class, len(group))
        selected.extend([group[index] for index in rng.choice(len(group), size=count, replace=False)])
    rng.shuffle(selected)
    return selected


def load_image(zf: zipfile.ZipFile, member: str) -> Image.Image:
    with zf.open(member) as image_file:
        return Image.open(image_file).convert("RGB")


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def add_noise(image: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    array = np.asarray(image, dtype=np.float32) / 255.0
    noisy = array + rng.normal(0, sigma, array.shape)
    return Image.fromarray(np.clip(noisy * 255, 0, 255).astype(np.uint8))


def corruption_plan() -> dict[str, tuple[str, float]]:
    return {
        "clean": ("clean", 0.0),
        "gaussian_blur": ("blur", 2.0),
        "low_contrast": ("contrast", 0.45),
        "underexposed": ("brightness", 0.45),
        "overexposed": ("brightness", 1.65),
        "gaussian_noise": ("noise", 0.08),
        "jpeg_compression": ("jpeg", 18),
    }


def apply_corruption(image: Image.Image, kind: str, value: float, rng: np.random.Generator) -> Image.Image:
    if kind == "clean":
        return image.copy()
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=value))
    if kind == "contrast":
        return ImageEnhance.Contrast(image).enhance(value)
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(value)
    if kind == "noise":
        return add_noise(image, value, rng)
    if kind == "jpeg":
        return jpeg_compress(image, quality=int(value))
    raise ValueError(f"Unknown corruption kind: {kind}")


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
    }


def evaluate_corruptions(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    zip_path = args.zip_path or download_dataset_zip()
    samples = balanced_subset(test_split(collect_samples(zip_path), args.seed), args.max_samples, args.seed)
    model, model_error = load_keras_model()
    if model_error:
        raise RuntimeError(model_error)

    plan = corruption_plan()
    rng = np.random.default_rng(args.seed)
    rows = []

    with zipfile.ZipFile(zip_path) as zf:
        for image_index, (member, true_positive) in enumerate(samples):
            original = load_image(zf, member)
            for corruption_name, (kind, value) in plan.items():
                corrupted = apply_corruption(original, kind, value, rng)
                batch = preprocess_image(corrupted.resize(IMG_SIZE))
                raw_uninfected = float(model.predict(batch, verbose=0)[0][0])
                parasitized_score = 1.0 - raw_uninfected
                quality = assess_image_quality(corrupted)
                review_required, review_reason = build_review_decision(
                    parasitized_score,
                    args.threshold,
                    args.review_margin,
                    quality.warnings,
                    route_warnings_to_review=True,
                )
                rows.append(
                    {
                        "image_index": image_index,
                        "member": member,
                        "true_positive_parasitized": int(true_positive),
                        "corruption": corruption_name,
                        "parasitized_score": parasitized_score,
                        "predicted_positive": int(parasitized_score >= args.threshold),
                        "review_required": int(review_required),
                        "review_reason": review_reason,
                        "quality_passed": int(quality.passed),
                        "brightness_mean": quality.brightness_mean,
                        "contrast_std": quality.contrast_std,
                        "focus_score": quality.focus_score,
                        "saturation_mean": quality.saturation_mean,
                    }
                )

    predictions = pd.DataFrame(rows)
    metrics_rows = []
    for corruption_name, group in predictions.groupby("corruption"):
        metrics = metrics_at_threshold(
            group["true_positive_parasitized"].to_numpy(dtype=int),
            group["parasitized_score"].to_numpy(dtype=float),
            args.threshold,
        )
        metrics_rows.append(
            {
                "corruption": corruption_name,
                "sample_count": int(len(group)),
                "review_rate": float(group["review_required"].mean()),
                "quality_pass_rate": float(group["quality_passed"].mean()),
                "mean_score": float(group["parasitized_score"].mean()),
                **metrics,
            }
        )
    metrics_frame = pd.DataFrame(metrics_rows).sort_values("corruption")
    return predictions, metrics_frame


def plot_performance(metrics_frame: pd.DataFrame, output_path: Path) -> None:
    ordered = metrics_frame.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ordered["corruption"], ordered["accuracy"], marker="o", label="Accuracy")
    ax.plot(ordered["corruption"], ordered["sensitivity"], marker="o", label="Sensitivity")
    ax.plot(ordered["corruption"], ordered["specificity"], marker="o", label="Specificity")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Metric")
    ax.set_title("Robustness Under Synthetic Image Degradation")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_review_rates(metrics_frame: pd.DataFrame, output_path: Path) -> None:
    ordered = metrics_frame.sort_values("review_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ordered["corruption"], ordered["review_rate"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Review-required rate")
    ax.set_title("Review Routing By Corruption")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def metrics_markdown_table(metrics_frame: pd.DataFrame) -> str:
    columns = [
        "corruption",
        "sample_count",
        "accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "review_rate",
        "quality_pass_rate",
        "fn",
        "fp",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in metrics_frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_report(output_dir: Path, metrics_frame: pd.DataFrame, args) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    clean = metrics_frame[metrics_frame["corruption"] == "clean"].iloc[0]
    worst = metrics_frame.sort_values("accuracy").iloc[0]
    markdown_table = metrics_markdown_table(metrics_frame)
    content = f"""# Robustness Analysis

Generated: {generated_at}

## Purpose

This report stress-tests the classifier under synthetic image degradation: blur, low contrast,
exposure changes, noise, and JPEG compression. The goal is to measure how model behavior changes
when image acquisition quality deteriorates.

## Configuration

- Sample count per corruption: {int(clean['sample_count'])}
- Threshold: `{args.threshold:.3f}`
- Review margin: `{args.review_margin:.3f}`
- Seed: `{args.seed}`

## Key Findings

- Clean accuracy on sampled images: `{clean['accuracy']:.3f}`
- Worst corruption by accuracy: `{worst['corruption']}` with accuracy `{worst['accuracy']:.3f}`
- Worst corruption review rate: `{worst['review_rate']:.3f}`

## Metrics

{markdown_table}

## Figures

- [Performance by corruption](performance_by_corruption.png)
- [Review rate by corruption](review_rate_by_corruption.png)

## Interpretation

The robustness report demonstrates whether the pre-inference quality gate and expert-review
routing catch cases where acquisition artifacts may make predictions less reliable. These
synthetic degradations do not replace external validation, but they are useful for engineering
stress tests and portfolio-grade failure analysis.
"""
    (output_dir / "robustness_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate robustness under synthetic image degradation.")
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=PARASITIZED_THRESHOLD)
    parser.add_argument("--review-margin", type=float, default=DEFAULT_REVIEW_MARGIN)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions, metrics_frame = evaluate_corruptions(args)
    predictions.to_csv(args.output_dir / "robustness_predictions.csv", index=False)
    metrics_frame.to_csv(args.output_dir / "robustness_metrics.csv", index=False)
    plot_performance(metrics_frame, args.output_dir / "performance_by_corruption.png")
    plot_review_rates(metrics_frame, args.output_dir / "review_rate_by_corruption.png")
    write_report(args.output_dir, metrics_frame, args)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": int(len(predictions))}, indent=2))


if __name__ == "__main__":
    main()
