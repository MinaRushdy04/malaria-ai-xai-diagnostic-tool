from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "reports" / "evaluation" / "test_predictions.csv"
DEFAULT_CURRENT = ROOT / "logs" / "predictions.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "monitoring"


def find_score_column(frame: pd.DataFrame) -> str:
    for candidate in ["parasitized_score", "model_parasitized_score"]:
        if candidate in frame.columns:
            return candidate
    raise ValueError("Could not find a parasitized score column.")


def find_class_column(frame: pd.DataFrame) -> str | None:
    for candidate in ["predicted_class", "label", "true_label_positive_parasitized"]:
        if candidate in frame.columns:
            return candidate
    return None


def numeric_summary(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    scores = pd.to_numeric(frame[score_column], errors="coerce").dropna()
    if scores.empty:
        return {"count": 0, "mean": 0.0, "std": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "count": int(scores.shape[0]),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=0)),
        "p05": float(scores.quantile(0.05)),
        "p50": float(scores.quantile(0.50)),
        "p95": float(scores.quantile(0.95)),
    }


def population_stability_index(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    bins: int = 10,
) -> float:
    baseline_scores = baseline_scores[np.isfinite(baseline_scores)]
    current_scores = current_scores[np.isfinite(current_scores)]
    if baseline_scores.size == 0 or current_scores.size == 0:
        return 0.0

    edges = np.quantile(baseline_scores, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.linspace(0, 1, bins + 1)

    baseline_counts, _ = np.histogram(baseline_scores, bins=edges)
    current_counts, _ = np.histogram(current_scores, bins=edges)
    baseline_pct = np.clip(baseline_counts / max(baseline_counts.sum(), 1), 1e-6, None)
    current_pct = np.clip(current_counts / max(current_counts.sum(), 1), 1e-6, None)
    return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))


def class_distribution(frame: pd.DataFrame) -> dict[str, float]:
    class_column = find_class_column(frame)
    if not class_column:
        return {}
    values = frame[class_column].astype(str)
    counts = values.value_counts(normalize=True)
    return {str(label): float(value) for label, value in counts.items()}


def drift_status(psi: float, mean_shift: float, review_rate: float | None) -> str:
    if psi >= 0.25 or abs(mean_shift) >= 0.15:
        return "alert"
    if psi >= 0.10 or abs(mean_shift) >= 0.08 or (review_rate is not None and review_rate >= 0.35):
        return "watch"
    return "ok"


def bool_rate(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or frame.empty:
        return None
    values = frame[column].astype(str).str.lower().isin(["1", "true", "yes"])
    return float(values.mean())


def build_report(baseline: pd.DataFrame, current: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    baseline_score_column = find_score_column(baseline)
    current_score_column = find_score_column(current)
    baseline_scores = pd.to_numeric(baseline[baseline_score_column], errors="coerce").to_numpy(dtype=float)
    current_scores = pd.to_numeric(current[current_score_column], errors="coerce").to_numpy(dtype=float)

    baseline_summary = numeric_summary(baseline, baseline_score_column)
    current_summary = numeric_summary(current, current_score_column)
    psi = population_stability_index(baseline_scores, current_scores, bins=args.bins)
    mean_shift = current_summary["mean"] - baseline_summary["mean"]
    review_rate = bool_rate(current, "review_required")
    quality_pass_rate = bool_rate(current, "quality_passed")
    status = drift_status(psi, mean_shift, review_rate)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "baseline_path": args.baseline.as_posix(),
        "current_path": args.current.as_posix(),
        "score_column": {
            "baseline": baseline_score_column,
            "current": current_score_column,
        },
        "baseline_summary": baseline_summary,
        "current_summary": current_summary,
        "score_psi": psi,
        "mean_score_shift": mean_shift,
        "review_rate": review_rate,
        "quality_pass_rate": quality_pass_rate,
        "baseline_class_distribution": class_distribution(baseline),
        "current_class_distribution": class_distribution(current),
        "thresholds": {
            "psi_watch": 0.10,
            "psi_alert": 0.25,
            "mean_shift_watch": 0.08,
            "mean_shift_alert": 0.15,
            "review_rate_watch": 0.35,
        },
        "interpretation": (
            "Drift status is a monitoring signal, not proof of clinical failure. "
            "Investigate data collection, staining, microscope source, and recent review feedback."
        ),
    }


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    review_rate = report["review_rate"]
    quality_pass_rate = report["quality_pass_rate"]
    review_display = "not available" if review_rate is None else f"{review_rate:.4f}"
    quality_display = "not available" if quality_pass_rate is None else f"{quality_pass_rate:.4f}"
    content = f"""# Drift Monitoring Report

Generated: {report['generated_at_utc']}

Status: `{report['status']}`

## Score Distribution

| Metric | Baseline | Current |
|---|---:|---:|
| Count | {report['baseline_summary']['count']} | {report['current_summary']['count']} |
| Mean score | {report['baseline_summary']['mean']:.4f} | {report['current_summary']['mean']:.4f} |
| Std score | {report['baseline_summary']['std']:.4f} | {report['current_summary']['std']:.4f} |
| P05 | {report['baseline_summary']['p05']:.4f} | {report['current_summary']['p05']:.4f} |
| P50 | {report['baseline_summary']['p50']:.4f} | {report['current_summary']['p50']:.4f} |
| P95 | {report['baseline_summary']['p95']:.4f} | {report['current_summary']['p95']:.4f} |

## Drift Signals

- Score PSI: `{report['score_psi']:.4f}`
- Mean score shift: `{report['mean_score_shift']:.4f}`
- Review rate: `{review_display}`
- Quality pass rate: `{quality_display}`

## Interpretation

{report['interpretation']}
"""
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline predictions with recent logged predictions.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    if not args.current.exists():
        raise FileNotFoundError(
            f"Current prediction log not found: {args.current}. Run API/dashboard predictions first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(args.baseline)
    current = pd.read_csv(args.current)
    report = build_report(baseline, current, args)
    (args.output_dir / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.output_dir / "drift_report.md")
    print(json.dumps({"status": report["status"], "score_psi": report["score_psi"]}, indent=2))


if __name__ == "__main__":
    main()
