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
- Demonstrate how an inference model can be wrapped with validation, review routing, and logging

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

The Streamlit app and FastAPI inference service can provide Grad-CAM overlays for the predicted
class.

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
- Input validation checks file safety and structural suitability, not clinical appropriateness.
- Prediction logs are for reflection and debugging; they are not a regulated audit system.
- The image-quality gate uses simple brightness, contrast, and focus heuristics; it is not a replacement
  for clinical specimen-quality review.
- Monitoring summaries are local observability signals, not validated drift-detection infrastructure.

## Runtime Safeguards

- Images are validated before inference for file type, decodability, size, dimensions, and RGB conversion.
- Image quality is scored using brightness, contrast, focus/detail, and saturation heuristics.
- Near-threshold predictions can be routed to expert review using a configurable review band.
- Validation warnings can also trigger review routing.
- Predictions can be logged to local SQLite and CSV files without storing the raw uploaded image.
- Correlation IDs, model version, model hash, and request timing make API predictions easier to trace.

## Recommended Evaluation Before Any Serious Use

- Confusion matrix
- Sensitivity and specificity
- ROC-AUC and PR-AUC
- Calibration curve and Brier score
- Threshold analysis
- False-positive and false-negative review
- Grad-CAM review across correct and incorrect predictions
