from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from robustness_analysis import collect_samples, download_dataset_zip, test_split


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "reports" / "evaluation" / "test_predictions.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "error_analysis"
DEFAULT_THRESHOLD = 0.285


def export_image(zf: zipfile.ZipFile, member: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as source, output_path.open("wb") as target:
        shutil.copyfileobj(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build false-positive/false-negative error gallery.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--limit-per-type", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    zip_path = args.zip_path or download_dataset_zip()
    samples = test_split(collect_samples(zip_path), args.seed)
    predictions = pd.read_csv(args.predictions)
    if len(predictions) != len(samples):
        raise ValueError(
            f"Prediction rows ({len(predictions)}) do not match test split size ({len(samples)}). "
            "Regenerate reports/evaluation/test_predictions.csv with scripts/evaluate_threshold.py."
        )

    frame = predictions.copy()
    frame["member"] = [member for member, _ in samples]
    frame["predicted_positive"] = (frame["parasitized_score"] >= args.threshold).astype(int)
    frame["error_type"] = "correct"
    frame.loc[
        (frame["true_label_positive_parasitized"] == 1) & (frame["predicted_positive"] == 0),
        "error_type",
    ] = "false_negative"
    frame.loc[
        (frame["true_label_positive_parasitized"] == 0) & (frame["predicted_positive"] == 1),
        "error_type",
    ] = "false_positive"

    false_negatives = frame[frame["error_type"] == "false_negative"].sort_values(
        "parasitized_score",
        ascending=False,
    ).head(args.limit_per_type)
    false_positives = frame[frame["error_type"] == "false_positive"].sort_values(
        "parasitized_score",
        ascending=False,
    ).head(args.limit_per_type)
    selected = pd.concat([false_negatives, false_positives], ignore_index=True)

    image_dir = args.output_dir / "images"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for index, row in selected.iterrows():
            source_member = str(row["member"])
            suffix = Path(source_member).suffix or ".png"
            filename = f"{row['error_type']}_{index:02d}{suffix}"
            output_path = image_dir / filename
            export_image(zf, source_member, output_path)
            rows.append(
                {
                    "error_type": row["error_type"],
                    "true_label_positive_parasitized": int(row["true_label_positive_parasitized"]),
                    "parasitized_score": float(row["parasitized_score"]),
                    "raw_uninfected_sigmoid": float(row["raw_uninfected_sigmoid"]),
                    "member": source_member,
                    "exported_image": str(output_path.relative_to(args.output_dir)),
                }
            )

    index_path = args.output_dir / "error_gallery_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    content = f"""# Error Analysis Gallery

Generated: {generated_at}

Threshold: `{args.threshold:.3f}`

This gallery exports the highest-risk false negatives and false positives from the test split.
False negatives are sorted by highest Parasitized score below threshold; false positives are
sorted by highest Parasitized score above threshold.

## Counts

- False negatives exported: {len(false_negatives)}
- False positives exported: {len(false_positives)}

## Files

- [Gallery index](error_gallery_index.csv)
- Images folder: `images/`

## Why This Matters

Healthcare AI projects should inspect failure modes directly. Aggregate metrics are necessary,
but reviewing false positives and false negatives helps reveal whether errors are related to
image quality, morphology, threshold placement, or model attention patterns.
"""
    (args.output_dir / "error_gallery.md").write_text(content, encoding="utf-8")
    print(f"Wrote {len(rows)} gallery items to {args.output_dir}")


if __name__ == "__main__":
    main()
