# Dataset Scope

This project currently targets cropped single-cell malaria classification. That choice matters:
calibration, robustness stress tests, Grad-CAM, thresholding, and review routing are valid for
the current system, while full-slide diagnosis, parasite localization, and parasitemia estimation
require different labels and model architecture.

## Current Dataset

The current classifier uses the public NIH/NLM malaria cell image dataset:

```text
https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip
```

This dataset contains cropped thin-smear cell images labeled as `Parasitized` or `Uninfected`.
It is suitable for:

- Binary cell classification
- Threshold analysis
- Calibration analysis
- Robustness testing under synthetic image degradation
- Explainability inspection with Grad-CAM
- Input-quality and expert-review routing experiments

It is not suitable for:

- Full-slide diagnosis
- Parasitemia calculation
- Object detection across a field of view
- Cell counting across many microscope fields
- Species/stage identification beyond the binary label

## Candidate Datasets For Future Work

### BBBC041

BBBC041 contains P. vivax infected blood-smear images with bounding-box annotations for multiple
cell and parasite-stage classes. It is better aligned with an object-detection pipeline than the
current cropped-cell classifier.

Potential use:

- Cell detection
- Infected-cell localization
- Multi-class parasite-stage experiments
- A future YOLO/RT-DETR style detection system

### NIH Thick Smears

The NIH/NLM thick-smear collection contains patient smear images that are closer to parasite
counting workflows. This is more relevant to parasitemia-style questions, but it requires a
different pipeline and annotation assumptions.

Potential use:

- Parasite detection in thicker smear fields
- Counting-style workflows
- Clinical workflow discussion around parasite burden

## Current Engineering Decision

For this repository, the honest scope is:

> Robust, explainable, threshold-aware single-cell malaria classification.

The next technically valid extension inside the current scope is not full-slide parasitemia.
It is robustness, calibration, failure analysis, and safer inference behavior around the existing
classifier.
