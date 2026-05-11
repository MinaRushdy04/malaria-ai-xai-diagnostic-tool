from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports" / "evaluation" / "test_predictions.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "calibration"


def compute_calibration_bins(y_true: np.ndarray, scores: np.ndarray, bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        left = edges[index]
        right = edges[index + 1]
        if index == bins - 1:
            mask = (scores >= left) & (scores <= right)
        else:
            mask = (scores >= left) & (scores < right)

        count = int(mask.sum())
        if count:
            mean_score = float(scores[mask].mean())
            observed_rate = float(y_true[mask].mean())
            abs_gap = abs(observed_rate - mean_score)
        else:
            mean_score = float((left + right) / 2)
            observed_rate = np.nan
            abs_gap = np.nan

        rows.append(
            {
                "bin": index + 1,
                "left": float(left),
                "right": float(right),
                "count": count,
                "mean_predicted_score": mean_score,
                "observed_positive_rate": observed_rate,
                "absolute_gap": abs_gap,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(bins_frame: pd.DataFrame, total_count: int) -> float:
    non_empty = bins_frame[bins_frame["count"] > 0].copy()
    weighted = non_empty["count"] / total_count * non_empty["absolute_gap"]
    return float(weighted.sum())


def plot_reliability_curve(bins_frame: pd.DataFrame, output_path: Path) -> None:
    non_empty = bins_frame[bins_frame["count"] > 0]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(
        non_empty["mean_predicted_score"],
        non_empty["observed_positive_rate"],
        marker="o",
        label="Model bins",
    )
    ax.set_xlabel("Mean predicted Parasitized score")
    ax.set_ylabel("Observed Parasitized rate")
    ax.set_title("Reliability Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_score_histogram(y_true: np.ndarray, scores: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(scores[y_true == 1], bins=30, alpha=0.65, label="True Parasitized")
    ax.hist(scores[y_true == 0], bins=30, alpha=0.65, label="True Uninfected")
    ax.set_xlabel("Parasitized score")
    ax.set_ylabel("Image count")
    ax.set_title("Score Distribution By True Class")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    summary: dict,
    bins_frame: pd.DataFrame,
) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    non_empty_bins = bins_frame[bins_frame["count"] > 0]
    worst_bin = non_empty_bins.sort_values("absolute_gap", ascending=False).iloc[0]

    content = f"""# Calibration Analysis

Generated: {generated_at}

## Purpose

This report evaluates whether the model's `Parasitized` score behaves like a calibrated
probability on the test split. Calibration is different from accuracy: a model can be accurate
while still being overconfident or underconfident.

## Summary

- Test samples: {summary['sample_count']}
- Brier score: `{summary['brier_score']:.4f}`
- Expected calibration error (ECE): `{summary['expected_calibration_error']:.4f}`
- Maximum calibration error (MCE): `{summary['maximum_calibration_error']:.4f}`
- Mean Parasitized score: `{summary['mean_score']:.4f}`
- Observed Parasitized rate: `{summary['observed_positive_rate']:.4f}`

Worst non-empty bin:

- Score range: `{worst_bin['left']:.2f}` to `{worst_bin['right']:.2f}`
- Count: `{int(worst_bin['count'])}`
- Mean predicted score: `{worst_bin['mean_predicted_score']:.3f}`
- Observed positive rate: `{worst_bin['observed_positive_rate']:.3f}`
- Absolute gap: `{worst_bin['absolute_gap']:.3f}`

## Figures

- [Reliability curve](reliability_curve.png)
- [Score histogram](score_histogram.png)

## Interpretation

The model score should not be treated as a clinically calibrated probability unless calibration
is explicitly validated. This report makes that limitation visible and gives a baseline for
future calibration methods such as temperature scaling or isotonic regression.
"""
    (output_dir / "calibration_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate calibration analysis from test predictions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    y_true = frame["true_label_positive_parasitized"].to_numpy(dtype=int)
    scores = frame["parasitized_score"].to_numpy(dtype=float)

    bins_frame = compute_calibration_bins(y_true, scores, args.bins)
    brier_score = float(np.mean((scores - y_true) ** 2))
    ece = expected_calibration_error(bins_frame, len(scores))
    mce = float(bins_frame["absolute_gap"].dropna().max())
    summary = {
        "sample_count": int(len(scores)),
        "brier_score": brier_score,
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
        "mean_score": float(scores.mean()),
        "observed_positive_rate": float(y_true.mean()),
        "input": str(args.input),
        "bins": int(args.bins),
    }

    bins_frame.to_csv(args.output_dir / "calibration_bins.csv", index=False)
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    plot_reliability_curve(bins_frame, args.output_dir / "reliability_curve.png")
    plot_score_histogram(y_true, scores, args.output_dir / "score_histogram.png")
    write_report(args.output_dir, summary, bins_frame)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
