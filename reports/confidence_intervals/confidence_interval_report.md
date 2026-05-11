# Confidence Interval Analysis

Generated: 2026-05-11 15:37 UTC

## Purpose

Point metrics are incomplete in healthcare AI. This report estimates uncertainty around the
test-set metrics using stratified bootstrap resampling.

## Configuration

- Input predictions: `reports/evaluation/test_predictions.csv`
- Threshold: `0.285`
- Bootstrap iterations: `2000`
- Seed: `42`
- Bootstrap type: stratified resampling by true class

## Test Confusion Matrix

- True negatives: 1306
- False positives: 106
- False negatives: 54
- True positives: 1291

## 95% Confidence Intervals

| Metric | Point estimate | 95% CI lower | 95% CI upper |
|---|---:|---:|---:|
| accuracy | 0.942 | 0.934 | 0.951 |
| sensitivity | 0.960 | 0.949 | 0.970 |
| specificity | 0.925 | 0.911 | 0.938 |
| precision | 0.924 | 0.911 | 0.937 |
| f1 | 0.942 | 0.933 | 0.950 |
| roc_auc | 0.982 | 0.978 | 0.986 |
| pr_auc | 0.980 | 0.973 | 0.986 |

## Figure

- [Metric confidence intervals](metric_confidence_intervals.png)

## Interpretation

These intervals describe uncertainty from the finite test split. They do not account for dataset
shift, new microscope hardware, different staining protocols, or patient-level deployment
conditions. External validation is still required before any serious clinical interpretation.
