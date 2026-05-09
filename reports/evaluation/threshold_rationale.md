# Threshold Rationale and Evaluation

Generated: 2026-05-09 07:06 UTC

## Dataset

- Source ZIP: `https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip`
- Local ZIP cache: `D:\Projects\AI in Healtchare\AI-X-ray-diagnosis-project-xai\data\raw\cell_images.zip`
- Label mapping: `Parasitized = positive class`, `Uninfected = negative class`
- Deterministic split seed: `42`
- Training slice: 22046 images
- Validation slice: 2755 images
- Test slice: 2757 images

Note: this evaluation rebuilds an 80/10/10 split from the fetched ZIP dataset. It is suitable
for a reproducible academic report, but it may not exactly match the original Kaggle file split
used in the training notebook.

## Threshold Policy

Positive class: `Parasitized`.

The model outputs a sigmoid score for `Uninfected`. For thresholding, this report converts it
to a `Parasitized` score using:

```text
parasitized_score = 1 - raw_uninfected_sigmoid
```

Selected the validation threshold that maximizes sensitivity for Parasitized while keeping specificity at or above 0.90.

Selected threshold: `0.285`.

This is a better healthcare-AI framing than using accuracy alone: the threshold is chosen from
validation data using a stated clinical preference, then evaluated separately on the test set.

## Validation Metrics

Default threshold:

| Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.500 | 0.936 | 0.916 | 0.956 | 0.955 | 0.935 | 1264 | 60 | 1315 | 116 |


Selected threshold:

| Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.285 | 0.932 | 0.961 | 0.903 | 0.908 | 0.934 | 1326 | 134 | 1241 | 54 |


## Test Metrics

Default threshold:

| Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.500 | 0.943 | 0.920 | 0.965 | 0.962 | 0.940 | 1237 | 49 | 1363 | 108 |


Selected threshold:

| Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.285 | 0.942 | 0.960 | 0.925 | 0.924 | 0.942 | 1291 | 106 | 1306 | 54 |


Test ROC-AUC for `Parasitized` detection: `0.982`.

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
