from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

from scripts.experiment_tracking import log_experiment


DATASET_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
RAW_ZIP_PATH = ROOT / "data" / "raw" / "cell_images.zip"
EXTRACT_DIR = ROOT / "data" / "processed"
DEFAULT_DATA_DIR = EXTRACT_DIR / "cell_images"
DEFAULT_OUTPUT = ROOT / "models" / "malaria_mobilenetv2.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Parasitized", "Uninfected"]


def download_dataset(zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"Downloading dataset: {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
    return zip_path


def extract_dataset(zip_path: Path, extract_dir: Path) -> Path:
    data_dir = extract_dir / "cell_images"
    if (data_dir / "Parasitized").exists() and (data_dir / "Uninfected").exists():
        return data_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting dataset to {extract_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return data_dir


def resolve_data_dir(data_dir: Path | None, zip_path: Path) -> Path:
    if data_dir:
        return data_dir
    zip_path = download_dataset(zip_path)
    return extract_dataset(zip_path, EXTRACT_DIR)


def build_datasets(data_dir: Path, batch_size: int, seed: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=IMG_SIZE,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=IMG_SIZE,
        batch_size=batch_size,
    )

    autotune = tf.data.AUTOTUNE

    def normalize(images, labels):
        return tf.cast(images, tf.float32) / 255.0, labels

    return (
        train_ds.map(normalize, num_parallel_calls=autotune).prefetch(autotune),
        val_ds.map(normalize, num_parallel_calls=autotune).prefetch(autotune),
    )


def build_model(dropout: float, learning_rate: float):
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
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation="sigmoid", name="uninfected_score"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MobileNetV2 malaria cell classifier.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--zip-path", type=Path, default=RAW_ZIP_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracking-backend", choices=["none", "local", "mlflow", "wandb"], default="local")
    parser.add_argument("--experiment-name", default="malaria-cell-classifier")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    data_dir = resolve_data_dir(args.data_dir, args.zip_path)
    train_ds, val_ds = build_datasets(data_dir, args.batch_size, args.seed)
    model = build_model(args.dropout, args.learning_rate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.output,
            monitor="val_roc_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_roc_auc",
            mode="max",
            patience=2,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.save(args.output)
    final_metrics = {key: values[-1] for key, values in history.history.items() if values}
    tracking_result = log_experiment(
        backend=args.tracking_backend,
        experiment_name=args.experiment_name,
        run_name=args.run_name or f"mobilenetv2-seed-{args.seed}",
        params={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "model_output": str(args.output),
        },
        metrics={key: float(value) for key, value in final_metrics.items()},
        artifacts=[args.output],
    )
    print(f"Saved model to {args.output}")
    print(f"Tracking: {tracking_result}")


if __name__ == "__main__":
    main()
