import argparse
import gc
import json
import os
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from PIL import Image
from sklearn.metrics import auc, confusion_matrix, roc_curve


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "malaria_App" / "malaria_cell_parasite_prediction_model.h5"
RAW_DATA_DIR = ROOT / "data" / "raw"
ZIP_PATH = RAW_DATA_DIR / "cell_images.zip"
REPORT_DIR = ROOT / "reports" / "evaluation"
DATASET_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
IMG_SIZE = (224, 224)
POSITIVE_CLASS = "Parasitized"
NEGATIVE_CLASS = "Uninfected"


def load_model():
    try:
        return tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception:
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
        with h5py.File(str(MODEL_PATH), "r") as f:
            weights = f["model_weights"]
            kernel = np.array(weights["dense"]["sequential"]["dense"]["kernel"])
            bias = np.array(weights["dense"]["sequential"]["dense"]["bias"])
        model.layers[-1].set_weights([kernel, bias])
        return model


def find_existing_dataset_zip():
    if ZIP_PATH.exists():
        return ZIP_PATH

    candidates = [
        *ROOT.glob("data/tfds/downloads/malaria/*.zip"),
        *ROOT.glob("data/tfds/downloads/**/*.zip"),
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 300_000_000:
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate, ZIP_PATH)
            return ZIP_PATH
    return None


def download_dataset_zip():
    existing = find_existing_dataset_zip()
    if existing:
        return existing

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    return ZIP_PATH


def collect_samples(zip_path):
    samples = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            lower_member = member.lower()
            if not lower_member.endswith((".png", ".jpg", ".jpeg")):
                continue
            path_parts = member.replace("\\", "/").split("/")
            if POSITIVE_CLASS in path_parts:
                samples.append((member, 0))
            elif NEGATIVE_CLASS in path_parts:
                samples.append((member, 1))

    if not samples:
        raise ValueError(f"No malaria images found inside {zip_path}")
    return samples


def split_samples(samples, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    shuffled = [samples[index] for index in indices]

    total = len(shuffled)
    train_count = int(total * 0.8)
    val_count = int(total * 0.1)
    test_count = total - train_count - val_count
    val_samples = shuffled[train_count : train_count + val_count]
    test_samples = shuffled[train_count + val_count :]

    split_info = {
        "dataset_url": DATASET_URL,
        "zip_path": str(ZIP_PATH),
        "label_names": ["parasitized", "uninfected"],
        "total": total,
        "train_count": train_count,
        "validation_count": val_count,
        "test_count": test_count,
        "seed": seed,
    }
    return val_samples, test_samples, split_info


def load_image_from_zip(zf, member):
    with zf.open(member) as image_file:
        image = Image.open(image_file).convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def predict_samples(model, zip_path, samples, batch_size, split_name, cache_path, max_predictions_this_run):
    y_true = []
    raw_uninfected_scores = []
    processed = 0
    processed_this_run = 0

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_count = int(cached["processed"])
        if cached_count <= len(samples):
            processed = cached_count
            y_true = cached["y_true"].astype(int).tolist()
            raw_uninfected_scores = cached["raw_uninfected_scores"].astype(np.float32).tolist()
            print(f"{split_name}: resuming from {processed}/{len(samples)} cached predictions")

    with zipfile.ZipFile(zip_path) as zf:
        for start in range(processed, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            batch_images = np.empty((len(batch), IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
            for row, (member, _) in enumerate(batch):
                batch_images[row] = load_image_from_zip(zf, member).astype(np.float32) / 255.0

            batch_labels = np.asarray([label for _, label in batch], dtype=int)
            raw_scores = model(batch_images, training=False).numpy().reshape(-1)
            raw_uninfected_scores.extend(raw_scores.tolist())
            y_true.extend(batch_labels.tolist())
            del batch_images
            gc.collect()

            batch_number = start // batch_size + 1
            total_batches = int(np.ceil(len(samples) / batch_size))
            if batch_number == 1 or batch_number == total_batches or batch_number % 10 == 0:
                print(f"{split_name}: predicted batch {batch_number}/{total_batches}")
            if batch_number % 10 == 0 or batch_number == total_batches:
                np.savez(
                    cache_path,
                    processed=start + len(batch),
                    y_true=np.asarray(y_true, dtype=np.int8),
                    raw_uninfected_scores=np.asarray(raw_uninfected_scores, dtype=np.float32),
                )
            processed_this_run += len(batch)
            if (
                max_predictions_this_run
                and processed_this_run >= max_predictions_this_run
                and start + len(batch) < len(samples)
            ):
                print(f"{split_name}: checkpointed {start + len(batch)}/{len(samples)} predictions")
                return None, None, None, False

    y_true = np.asarray(y_true, dtype=int)
    raw_uninfected_scores = np.asarray(raw_uninfected_scores, dtype=np.float32)
    y_true_positive = (y_true == 0).astype(int)
    parasitized_scores = 1.0 - raw_uninfected_scores
    return y_true_positive, parasitized_scores, raw_uninfected_scores, True


def metrics_at_threshold(y_true_positive, parasitized_scores, threshold):
    y_pred_positive = (parasitized_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_positive, y_pred_positive, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    youden_j = sensitivity + specificity - 1

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "negative_predictive_value": float(npv),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "youden_j": float(youden_j),
    }


def choose_threshold(validation_metrics, min_specificity):
    feasible = validation_metrics[validation_metrics["specificity"] >= min_specificity]
    if not feasible.empty:
        chosen = feasible.sort_values(
            ["sensitivity", "f1", "specificity"],
            ascending=[False, False, False],
        ).iloc[0]
        rationale = (
            f"Selected the validation threshold that maximizes sensitivity for {POSITIVE_CLASS} "
            f"while keeping specificity at or above {min_specificity:.2f}."
        )
    else:
        chosen = validation_metrics.sort_values(["youden_j", "f1"], ascending=[False, False]).iloc[0]
        rationale = (
            f"No threshold reached specificity >= {min_specificity:.2f}; selected the threshold "
            "with the best Youden J statistic instead."
        )
    return float(chosen["threshold"]), rationale


def plot_confusion(metrics, title, output_path):
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=[NEGATIVE_CLASS, POSITIVE_CLASS],
        yticklabels=[NEGATIVE_CLASS, POSITIVE_CLASS],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    for row in range(2):
        for col in range(2):
            ax.text(col, row, matrix[row, col], ha="center", va="center", color="black", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_threshold_sweep(metrics_frame, selected_threshold, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics_frame["threshold"], metrics_frame["sensitivity"], label="Sensitivity")
    ax.plot(metrics_frame["threshold"], metrics_frame["specificity"], label="Specificity")
    ax.plot(metrics_frame["threshold"], metrics_frame["f1"], label="F1")
    ax.axvline(selected_threshold, color="black", linestyle="--", label=f"Selected t={selected_threshold:.3f}")
    ax.set_xlabel(f"{POSITIVE_CLASS} threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.02)
    ax.set_title("Validation Threshold Sweep")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_roc(y_true_positive, parasitized_scores, output_path):
    fpr, tpr, _ = roc_curve(y_true_positive, parasitized_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate / sensitivity")
    ax.set_title(f"ROC Curve: {POSITIVE_CLASS} Detection")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return float(roc_auc)


def write_report(split_info, rationale, selected_threshold, validation_selected, validation_default, test_selected, test_default, test_auc):
    report_path = REPORT_DIR / "threshold_rationale.md"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def metric_table(metrics):
        return (
            "| Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | TP | FP | TN | FN |\n"
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            f"| {metrics['threshold']:.3f} | {metrics['accuracy']:.3f} | "
            f"{metrics['sensitivity']:.3f} | {metrics['specificity']:.3f} | "
            f"{metrics['precision']:.3f} | {metrics['f1']:.3f} | "
            f"{metrics['tp']} | {metrics['fp']} | {metrics['tn']} | {metrics['fn']} |\n"
        )

    content = f"""# Threshold Rationale and Evaluation

Generated: {generated_at}

## Dataset

- Source ZIP: `{split_info['dataset_url']}`
- Local ZIP cache: `{split_info['zip_path']}`
- Label mapping: `Parasitized = positive class`, `Uninfected = negative class`
- Deterministic split seed: `{split_info['seed']}`
- Training slice: {split_info['train_count']} images
- Validation slice: {split_info['validation_count']} images
- Test slice: {split_info['test_count']} images

Note: this evaluation rebuilds an 80/10/10 split from the fetched ZIP dataset. It is suitable
for a reproducible academic report, but it may not exactly match the original Kaggle file split
used in the training notebook.

## Threshold Policy

Positive class: `{POSITIVE_CLASS}`.

The model outputs a sigmoid score for `{NEGATIVE_CLASS}`. For thresholding, this report converts it
to a `{POSITIVE_CLASS}` score using:

```text
parasitized_score = 1 - raw_uninfected_sigmoid
```

{rationale}

Selected threshold: `{selected_threshold:.3f}`.

This is a better healthcare-AI framing than using accuracy alone: the threshold is chosen from
validation data using a stated clinical preference, then evaluated separately on the test set.

## Validation Metrics

Default threshold:

{metric_table(validation_default)}

Selected threshold:

{metric_table(validation_selected)}

## Test Metrics

Default threshold:

{metric_table(test_default)}

Selected threshold:

{metric_table(test_selected)}

Test ROC-AUC for `{POSITIVE_CLASS}` detection: `{test_auc:.3f}`.

## Figures

- [Validation threshold sweep](threshold_sweep.png)
- [Test confusion matrix at selected threshold](confusion_matrix_selected_threshold.png)
- [Test confusion matrix at default threshold](confusion_matrix_default_threshold.png)
- [Test ROC curve](roc_curve.png)

## Interpretation Notes

- Sensitivity answers: of truly parasitized cells, how many did the model catch?
- Specificity answers: of truly uninfected cells, how many did the model leave unflagged?
- False negatives are especially important in a screening context because infected cells are missed.
- The selected threshold is not clinically validated; it is an academic demonstration of explicit
  threshold selection and should be documented as such.
"""
    report_path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Download malaria data and evaluate threshold rationale.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-specificity", type=float, default=0.90)
    parser.add_argument("--max-predictions-per-run", type=int, default=600)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = download_dataset_zip()
    samples = collect_samples(zip_path)
    val_samples, test_samples, split_info = split_samples(samples, seed=args.seed)

    model = load_model()
    val_true, val_scores, _, val_complete = predict_samples(
        model,
        zip_path,
        val_samples,
        args.batch_size,
        "validation",
        REPORT_DIR / "validation_predictions_cache.npz",
        args.max_predictions_per_run,
    )
    if not val_complete:
        print(json.dumps({"status": "partial", "split": "validation"}))
        return

    test_true, test_scores, test_raw, test_complete = predict_samples(
        model,
        zip_path,
        test_samples,
        args.batch_size,
        "test",
        REPORT_DIR / "test_predictions_cache.npz",
        args.max_predictions_per_run,
    )
    if not test_complete:
        print(json.dumps({"status": "partial", "split": "test"}))
        return

    thresholds = np.linspace(0.01, 0.99, 197)
    val_metrics = pd.DataFrame([metrics_at_threshold(val_true, val_scores, t) for t in thresholds])
    selected_threshold, rationale = choose_threshold(val_metrics, min_specificity=args.min_specificity)

    validation_selected = metrics_at_threshold(val_true, val_scores, selected_threshold)
    validation_default = metrics_at_threshold(val_true, val_scores, 0.5)
    test_selected = metrics_at_threshold(test_true, test_scores, selected_threshold)
    test_default = metrics_at_threshold(test_true, test_scores, 0.5)

    val_metrics.to_csv(REPORT_DIR / "validation_threshold_sweep.csv", index=False)
    pd.DataFrame(
        [
            {"split": "validation", "policy": "default_0.5", **validation_default},
            {"split": "validation", "policy": "selected", **validation_selected},
            {"split": "test", "policy": "default_0.5", **test_default},
            {"split": "test", "policy": "selected", **test_selected},
        ]
    ).to_csv(REPORT_DIR / "metrics_summary.csv", index=False)
    pd.DataFrame(
        {
            "true_label_positive_parasitized": test_true,
            "parasitized_score": test_scores,
            "raw_uninfected_sigmoid": test_raw,
        }
    ).to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    plot_threshold_sweep(val_metrics, selected_threshold, REPORT_DIR / "threshold_sweep.png")
    plot_confusion(
        test_selected,
        f"Test Confusion Matrix, Selected Threshold ({selected_threshold:.3f})",
        REPORT_DIR / "confusion_matrix_selected_threshold.png",
    )
    plot_confusion(
        test_default,
        "Test Confusion Matrix, Default Threshold (0.500)",
        REPORT_DIR / "confusion_matrix_default_threshold.png",
    )
    test_auc = plot_roc(test_true, test_scores, REPORT_DIR / "roc_curve.png")

    write_report(
        split_info,
        rationale,
        selected_threshold,
        validation_selected,
        validation_default,
        test_selected,
        test_default,
        test_auc,
    )

    summary = {
        "selected_threshold": selected_threshold,
        "validation_selected": validation_selected,
        "test_selected": test_selected,
        "test_auc": test_auc,
        "report": str(REPORT_DIR / "threshold_rationale.md"),
    }
    (REPORT_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
