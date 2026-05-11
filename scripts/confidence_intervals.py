from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports" / "evaluation" / "test_predictions.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "confidence_intervals"
DEFAULT_THRESHOLD = 0.285


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positive_indices = np.flatnonzero(y_true == 1)
    negative_indices = np.flatnonzero(y_true == 0)
    sampled_positive = rng.choice(positive_indices, size=len(positive_indices), replace=True)
    sampled_negative = rng.choice(negative_indices, size=len(negative_indices), replace=True)
    sampled = np.concatenate([sampled_positive, sampled_negative])
    rng.shuffle(sampled)
    return sampled


def bootstrap_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_bootstrap):
        sampled = stratified_bootstrap_indices(y_true, rng)
        rows.append(threshold_metrics(y_true[sampled], scores[sampled], threshold))
    return pd.DataFrame(rows)


def summarize_intervals(point_metrics: dict[str, float], bootstrap_frame: pd.DataFrame) -> pd.DataFrame:
    metric_names = ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc", "pr_auc"]
    rows = []
    for metric in metric_names:
        values = bootstrap_frame[metric].to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric,
                "point_estimate": point_metrics[metric],
                "ci_lower_95": float(np.quantile(values, 0.025)),
                "ci_upper_95": float(np.quantile(values, 0.975)),
                "bootstrap_std": float(np.std(values, ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def plot_intervals(summary_frame: pd.DataFrame, output_path: Path) -> None:
    plot_frame = summary_frame.iloc[::-1].copy()
    y_positions = np.arange(len(plot_frame))
    lower_error = plot_frame["point_estimate"] - plot_frame["ci_lower_95"]
    upper_error = plot_frame["ci_upper_95"] - plot_frame["point_estimate"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        plot_frame["point_estimate"],
        y_positions,
        xerr=[lower_error, upper_error],
        fmt="o",
        color="#1f77b4",
        ecolor="#64748b",
        capsize=4,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_frame["metric"])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Metric value")
    ax.set_title("95% Bootstrap Confidence Intervals")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    summary_frame: pd.DataFrame,
    point_metrics: dict[str, float],
    args,
) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    input_display = str(args.input)
    try:
        input_display = args.input.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        input_display = str(args.input)

    table_rows = []
    for _, row in summary_frame.iterrows():
        table_rows.append(
            f"| {row['metric']} | {row['point_estimate']:.3f} | "
            f"{row['ci_lower_95']:.3f} | {row['ci_upper_95']:.3f} |"
        )
    table = "\n".join(
        [
            "| Metric | Point estimate | 95% CI lower | 95% CI upper |",
            "|---|---:|---:|---:|",
            *table_rows,
        ]
    )

    content = f"""# Confidence Interval Analysis

Generated: {generated_at}

## Purpose

Point metrics are incomplete in healthcare AI. This report estimates uncertainty around the
test-set metrics using stratified bootstrap resampling.

## Configuration

- Input predictions: `{input_display}`
- Threshold: `{args.threshold:.3f}`
- Bootstrap iterations: `{args.n_bootstrap}`
- Seed: `{args.seed}`
- Bootstrap type: stratified resampling by true class

## Test Confusion Matrix

- True negatives: {int(point_metrics['tn'])}
- False positives: {int(point_metrics['fp'])}
- False negatives: {int(point_metrics['fn'])}
- True positives: {int(point_metrics['tp'])}

## 95% Confidence Intervals

{table}

## Figure

- [Metric confidence intervals](metric_confidence_intervals.png)

## Interpretation

These intervals describe uncertainty from the finite test split. They do not account for dataset
shift, new microscope hardware, different staining protocols, or patient-level deployment
conditions. External validation is still required before any serious clinical interpretation.
"""
    (output_dir / "confidence_interval_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bootstrap confidence intervals for test metrics.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    y_true = frame["true_label_positive_parasitized"].to_numpy(dtype=int)
    scores = frame["parasitized_score"].to_numpy(dtype=float)

    point_metrics = threshold_metrics(y_true, scores, args.threshold)
    bootstrap_frame = bootstrap_metrics(
        y_true,
        scores,
        args.threshold,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    summary_frame = summarize_intervals(point_metrics, bootstrap_frame)

    summary_frame.to_csv(args.output_dir / "confidence_interval_summary.csv", index=False)
    bootstrap_frame.to_csv(args.output_dir / "bootstrap_metric_samples.csv", index=False)
    (args.output_dir / "confidence_interval_summary.json").write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
                "point_metrics": point_metrics,
                "intervals": summary_frame.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_intervals(summary_frame, args.output_dir / "metric_confidence_intervals.png")
    write_report(args.output_dir, summary_frame, point_metrics, args)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
