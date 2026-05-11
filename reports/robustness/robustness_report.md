# Robustness Analysis

Generated: 2026-05-11 15:10 UTC

## Purpose

This report stress-tests the classifier under synthetic image degradation: blur, low contrast,
exposure changes, noise, and JPEG compression. The goal is to measure how model behavior changes
when image acquisition quality deteriorates.

## Configuration

- Sample count per corruption: 60
- Threshold: `0.285`
- Review margin: `0.075`
- Seed: `42`

## Key Findings

- Clean accuracy on sampled images: `0.950`
- Worst corruption by accuracy: `gaussian_blur` with accuracy `0.633`
- Worst corruption review rate: `1.000`

## Metrics

| corruption | sample_count | accuracy | sensitivity | specificity | f1 | review_rate | quality_pass_rate | fn | fp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 60 | 0.950 | 0.967 | 0.933 | 0.951 | 0.050 | 1.000 | 1 | 2 |
| gaussian_blur | 60 | 0.633 | 1.000 | 0.267 | 0.732 | 1.000 | 0.000 | 0 | 22 |
| gaussian_noise | 60 | 0.833 | 0.700 | 0.967 | 0.808 | 0.117 | 1.000 | 9 | 1 |
| jpeg_compression | 60 | 0.817 | 1.000 | 0.633 | 0.845 | 0.067 | 1.000 | 0 | 11 |
| low_contrast | 60 | 0.900 | 0.867 | 0.933 | 0.897 | 1.000 | 0.000 | 4 | 2 |
| overexposed | 60 | 0.933 | 1.000 | 0.867 | 0.938 | 0.050 | 1.000 | 0 | 4 |
| underexposed | 60 | 0.917 | 0.867 | 0.967 | 0.912 | 1.000 | 0.000 | 4 | 1 |

## Figures

- [Performance by corruption](performance_by_corruption.png)
- [Review rate by corruption](review_rate_by_corruption.png)

## Interpretation

The robustness report demonstrates whether the pre-inference quality gate and expert-review
routing catch cases where acquisition artifacts may make predictions less reliable. These
synthetic degradations do not replace external validation, but they are useful for engineering
stress tests and portfolio-grade failure analysis.
