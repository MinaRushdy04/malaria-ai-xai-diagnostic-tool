# Model Card: Malaria Cell-Smear Classifier

## Model Details

- Model type: binary image classifier
- Architecture: MobileNetV2 feature extractor with a sigmoid classification head
- Input: 224 x 224 RGB microscopy cell image
- Output: probability-like sigmoid score
- Class mapping: `0 = Parasitized`, `1 = Uninfected`

## Intended Use

This model is intended for academic demonstration of AI-assisted microscopy classification,
with emphasis on explainability and responsible communication of limitations.

It may be used to:

- Demonstrate transfer learning for medical-image classification
- Demonstrate Grad-CAM as an explainability method
- Discuss risks of using model confidence as clinical certainty
- Discuss why healthcare AI evaluation needs sensitivity, specificity, calibration, and failure analysis

## Out-of-Scope Use

This model must not be used for:

- Clinical diagnosis
- Patient triage
- Automated treatment decisions
- Replacing microscopy review by qualified professionals
- Processing identifiable patient data through temporary public tunnels

## Data

The notebook uses a public malaria cell image dataset with cropped single-cell microscopy
images. The dataset is useful for education, but it does not cover the full variability of
clinical slide preparation, microscope hardware, acquisition settings, artifacts, or patient
populations.

## Explainability

The Streamlit app provides Grad-CAM overlays for the predicted class.

Grad-CAM can help users inspect whether the model attends to plausible image regions, but it
does not prove that the model has learned medically valid causal features. Heatmaps should be
reviewed alongside prediction errors and confidence calibration.

## Known Limitations

- The current app uses the validation-selected `0.285` threshold for the Parasitized score.
- The displayed confidence is the model's sigmoid output, not a calibrated probability.
- The repository includes a reproducible threshold report, but the model still needs external
  validation beyond this public dataset.
- The model was trained on cropped cells, not full slides or patient-level cases.
- Grad-CAM is spatially coarse and may not precisely outline parasite structures.

## Recommended Evaluation Before Any Serious Use

- Confusion matrix
- Sensitivity and specificity
- ROC-AUC and PR-AUC
- Calibration curve and Brier score
- Threshold analysis
- False-positive and false-negative review
- Grad-CAM review across correct and incorrect predictions
