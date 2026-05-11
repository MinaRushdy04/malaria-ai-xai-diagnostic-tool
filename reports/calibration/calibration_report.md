# Calibration Analysis

Generated: 2026-05-11 15:05 UTC

## Purpose

This report evaluates whether the model's `Parasitized` score behaves like a calibrated
probability on the test split. Calibration is different from accuracy: a model can be accurate
while still being overconfident or underconfident.

## Summary

- Test samples: 2757
- Brier score: `0.0460`
- Expected calibration error (ECE): `0.0266`
- Maximum calibration error (MCE): `0.2119`
- Mean Parasitized score: `0.4815`
- Observed Parasitized rate: `0.4878`

Worst non-empty bin:

- Score range: `0.50` to `0.60`
- Count: `46`
- Mean predicted score: `0.549`
- Observed positive rate: `0.761`
- Absolute gap: `0.212`

## Figures

- [Reliability curve](reliability_curve.png)
- [Score histogram](score_histogram.png)

## Interpretation

The model score should not be treated as a clinically calibrated probability unless calibration
is explicitly validated. This report makes that limitation visible and gives a baseline for
future calibration methods such as temperature scaling or isotonic regression.
