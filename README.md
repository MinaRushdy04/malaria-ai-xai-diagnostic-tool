# Malaria Cell-Smear AI Diagnostic Prototype

An academic AI-in-healthcare project for classifying thin-smear microscopy cell images as
`Parasitized` or `Uninfected`, with Grad-CAM explainability and explicit threshold evaluation.

This is not a production medical device. The project is designed to demonstrate responsible
AI integration in healthcare: prediction, confidence, explainability, threshold choice, and
limitations are presented together instead of treating accuracy as the only success measure.

## Project Highlights

- Transfer-learning classifier built with TensorFlow/Keras and MobileNetV2.
- Streamlit web app for uploading microscopy cell images.
- Grad-CAM overlay showing image regions that contributed most to the prediction.
- Optional activation-map debug view for inspecting MobileNetV2 feature channels.
- Validation-based threshold rationale for the `Parasitized` class.
- Committed evaluation artifacts: confusion matrices, ROC curve, threshold sweep, metrics CSV, and markdown report.
- Model card documenting intended use, limitations, and responsible-use boundaries.

## Repository Structure

```text
.
├── Diagnosis_model.ipynb
├── MODEL_CARD.md
├── README.md
├── malaria_App/
│   ├── app.py
│   ├── deploy_cloudflare.py
│   └── malaria_cell_parasite_prediction_model.h5
├── reports/
│   └── evaluation/
│       ├── confusion_matrix_default_threshold.png
│       ├── confusion_matrix_selected_threshold.png
│       ├── metrics_summary.csv
│       ├── roc_curve.png
│       ├── threshold_rationale.md
│       └── threshold_sweep.png
├── requirements.txt
└── scripts/
    └── evaluate_threshold.py
```

## Dataset

The evaluation script fetches the public NIH/NLM malaria cell image ZIP:

```text
https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip
```

The raw dataset is cached locally under `data/raw/` when the evaluation script runs. This folder
is ignored by Git because the dataset is large.

Dataset summary:

- Task: binary image classification
- Classes: `Parasitized`, `Uninfected`
- Approximate size: 27,558 cropped cell images
- Evaluation split: deterministic 80/10/10 split with seed `42`

Important limitation: these are cropped single-cell images. They do not represent full-slide
diagnosis, patient-level aggregation, microscope variability, slide preparation quality, or
clinical deployment conditions.

## Model

- Base model: MobileNetV2 pretrained on ImageNet
- Classification head: GlobalAveragePooling2D, Dropout, Dense sigmoid output
- Input size: 224 x 224 RGB
- Raw output: sigmoid score for `Uninfected`
- Derived positive score: `parasitized_score = 1 - raw_uninfected_sigmoid`

The saved model is included at:

```text
malaria_App/malaria_cell_parasite_prediction_model.h5
```

## Explainability

The Streamlit app generates a Grad-CAM heatmap for the predicted class and overlays it on the
uploaded image.

Interpretation guidance:

- Bright regions contributed more strongly to the model's prediction.
- Grad-CAM shows model attention, not medical causality.
- A plausible heatmap does not prove the diagnosis is correct.
- Low-confidence outputs or unclear explanations should be treated as expert-review cases.

The app also includes an activation-map debug view. This is useful for technical inspection,
but it should not be treated as clinician-facing evidence.

## Evaluation Results

The selected threshold is `0.285` for the `Parasitized` score. It was chosen on validation data
to maximize sensitivity for parasitized cells while keeping specificity at or above `0.90`, then
evaluated separately on the test split.

| Split | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | False Negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.285 | 0.932 | 0.961 | 0.903 | 0.908 | 0.934 | 54 |
| Test | 0.285 | 0.942 | 0.960 | 0.925 | 0.924 | 0.942 | 54 |

Test ROC-AUC for `Parasitized` detection: `0.982`.

Evaluation artifacts:

- [Threshold rationale report](reports/evaluation/threshold_rationale.md)
- [Validation threshold sweep](reports/evaluation/threshold_sweep.png)
- [Selected-threshold confusion matrix](reports/evaluation/confusion_matrix_selected_threshold.png)
- [Default-threshold confusion matrix](reports/evaluation/confusion_matrix_default_threshold.png)
- [ROC curve](reports/evaluation/roc_curve.png)
- [Metrics summary CSV](reports/evaluation/metrics_summary.csv)

## Run The App

From the repository root:

```bash
pip install -r requirements.txt
streamlit run malaria_App/app.py
```

Then upload a `.jpg`, `.jpeg`, or `.png` thin-smear cell image.

## Reproduce Evaluation

From the repository root:

```bash
python scripts/evaluate_threshold.py
```

The script downloads the dataset ZIP if needed, caches predictions in ignored local files, and
regenerates the evaluation report and figures under `reports/evaluation/`.

## Responsible Use

This project is for education and research demonstration only. It is not medical advice, not a
clinical diagnostic system, and not intended for patient care. Any real diagnosis must be made
by qualified healthcare professionals using validated clinical workflows.

If deploying through a temporary public tunnel, do not upload patient-identifiable or sensitive
medical data.

## Future Work

- Add false-positive and false-negative case galleries.
- Add calibration curve, Brier score, and expected calibration error.
- Add PR-AUC and confidence intervals for core metrics.
- Add Grad-CAM examples for true positives, true negatives, false positives, and false negatives.
- Evaluate on an external dataset or institutionally separate holdout set.
