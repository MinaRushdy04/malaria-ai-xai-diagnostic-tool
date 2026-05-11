# Malaria Cell-Smear AI Diagnostic Prototype

An academic AI-in-healthcare project for classifying cropped thin-smear microscopy cell images
as `Parasitized` or `Uninfected`. The project emphasizes responsible integration: prediction,
threshold selection, explainability, validation, expert-review routing, and logging are treated
as part of the system rather than as afterthoughts.

This is not a production medical device. It is a research and education prototype designed to
show how an AI model can be wrapped in safer decision-support behavior instead of presenting
accuracy as the only measure of success.

The notebook is kept as supporting experimental history under `notebooks/`. The main project
workflow is implemented through Python modules, CLI scripts, tests, a FastAPI service, a
Streamlit dashboard, and committed evaluation reports.

## Project Highlights

- Transfer-learning classifier built with TensorFlow/Keras and MobileNetV2.
- Streamlit dashboard with local-model mode and FastAPI-service mode.
- FastAPI inference layer with `/health` and `/predict` endpoints.
- Grad-CAM overlay for model attention, plus optional activation-map debug view.
- Validation-based threshold rationale for the `Parasitized` class.
- Flexible expert-review routing for near-threshold, validation-warning, or quality-warning cases.
- Input validation before inference: file type, decodability, size, dimensions, static image check, RGB normalization, and image-quality scoring.
- Prediction logging to SQLite and CSV for audit-style reflection.
- Correlation IDs, model version/hash logging, and request timing headers for API traceability.
- Lightweight monitoring summary for review rate, validation warnings, quality pass rate, and class mix.
- Calibration analysis with Brier score, expected calibration error, reliability curve, and score histogram.
- Robustness analysis under blur, noise, contrast, exposure, and compression degradation.
- Error-analysis gallery for false positives and false negatives.
- Docker and Docker Compose setup for dashboard/API execution.
- Script-first utilities for training, threshold evaluation, calibration, robustness, error analysis, and single-image CLI prediction.
- Automated tests for validation, quality gate, review routing, and API middleware behavior.
- Committed evaluation artifacts: confusion matrices, ROC curve, threshold sweep, metrics CSV, and markdown report.
- Model card documenting intended use, limitations, and responsible-use boundaries.

## Repository Structure

```text
.
|-- Dockerfile
|-- MODEL_CARD.md
|-- README.md
|-- docker-compose.yml
|-- docs/
|   |-- ARCHITECTURE.md
|   `-- DATASETS.md
|-- malaria_App/
|   |-- __init__.py
|   |-- api.py
|   |-- app.py
|   |-- diagnostic_core.py
|   |-- deploy_cloudflare.py
|   |-- middleware.py
|   |-- schemas.py
|   `-- malaria_cell_parasite_prediction_model.h5
|-- reports/
|   |-- calibration/
|   |-- error_analysis/
|   |-- evaluation/
|   `-- robustness/
|-- requirements.txt
|-- notebooks/
|   `-- 01_model_exploration.ipynb
|-- scripts/
|   |-- calibration_analysis.py
|   |-- error_analysis.py
|   |-- evaluate_threshold.py
|   |-- predict_image.py
|   |-- robustness_analysis.py
|   `-- train_model.py
|-- tests/
|   `-- test_core_safety.py
```

Runtime logs are written under `logs/`, which is ignored by Git.

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

## System Design

The project has two inference paths that share the same core logic:

```text
Streamlit UI
    |-- Local model mode --> diagnostic_core.py --> TensorFlow model
    `-- API mode ---------> FastAPI /predict --> diagnostic_core.py --> TensorFlow model
```

Shared behavior lives in `malaria_App/diagnostic_core.py`, including:

- Model loading
- Input validation
- Preprocessing
- Thresholding
- Expert-review routing
- Image-quality scoring
- Grad-CAM generation
- Activation-map generation
- Prediction logging
- Monitoring summaries

This keeps the UI and API consistent: the same image should receive the same validation,
threshold policy, review decision, and logging structure in both modes.

See [Architecture](docs/ARCHITECTURE.md) for the module-level structure.
See [Dataset Scope](docs/DATASETS.md) for why this repo focuses on cropped-cell robustness
instead of overclaiming full-slide parasitemia.

## Input Validation

The validation layer is a practical safety layer before model inference. It does not prove that
an image is clinically appropriate, but it prevents unsafe or unsuitable files from being silently
processed.

It checks:

- File is not empty.
- File size is below the configured upload limit.
- Extension is one of `.jpg`, `.jpeg`, or `.png`.
- The file can actually be decoded as an image.
- Encoding is static JPEG or PNG, not an animated image.
- Image dimensions are not too small for meaningful resizing.
- Image dimensions are not dangerously large.
- Image can be normalized into RGB format.
- Unusual aspect ratio or small images are flagged as warnings.
- Brightness, contrast, focus/detail, and saturation are measured.
- Underexposed, overexposed, low-contrast, or low-detail images are flagged as warnings.

Warnings can be routed into expert review through the Streamlit sidebar or API parameter.

## Expert-Review Routing

The default decision threshold is `0.285` for the `Parasitized` score. It was selected from the
validation threshold sweep, not guessed from a generic `0.5` cutoff.

The app also defines a configurable review band around the threshold. By default, cases within
`0.075` of the threshold are marked as requiring expert review.

Example:

```text
threshold = 0.285
review_margin = 0.075
review range = 0.210 to 0.360
```

Any case in that range is treated as uncertain enough for human review. This is intentionally
flexible: the user can widen or narrow the review band in Streamlit, and API callers can pass
`review_margin` per request.

## Logging

If logging is enabled, each prediction writes a row to:

```text
logs/predictions.sqlite3
logs/predictions.csv
```

Logged fields include:

- UTC timestamp
- Request ID
- Correlation ID
- Model version and model file hash
- Inference source: Streamlit local or FastAPI
- Predicted class
- Raw uninfected score
- Derived parasitized score
- Threshold and review margin
- Review-required flag and reason
- Image hash and filename hash
- Image dimensions and format
- Validation warnings
- Quality metrics: brightness, contrast, focus, saturation, and quality-pass flag
- Grad-CAM layer

The logger does not store raw images. Filenames are hashed rather than written directly.

## Monitoring Snapshot

The app and API expose a lightweight monitoring summary from recent local logs. This is not a
production drift detector, but it demonstrates the observability hooks a healthcare AI system
would need before serious deployment.

Tracked summary fields include:

- Total logged predictions
- Expert-review rate
- Validation-warning rate
- Image-quality pass rate
- Average focus score
- Average brightness
- Predicted class mix

FastAPI endpoint:

```bash
curl http://127.0.0.1:8000/monitoring/summary
```

## Explainability

The app generates a Grad-CAM heatmap for the predicted class and overlays it on the uploaded
image.

Interpretation guidance:

- Bright regions contributed more strongly to the model's prediction.
- Grad-CAM shows model attention, not medical causality.
- A plausible heatmap does not prove the diagnosis is correct.
- Low-confidence outputs, near-threshold outputs, or unclear explanations should be treated as expert-review cases.

The Streamlit app also includes an optional activation-map debug view. This is useful for
technical inspection, but it should not be treated as clinician-facing evidence.

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

Additional analysis artifacts:

- [Calibration report](reports/calibration/calibration_report.md)
- [Reliability curve](reports/calibration/reliability_curve.png)
- [Robustness report](reports/robustness/robustness_report.md)
- [Robustness performance plot](reports/robustness/performance_by_corruption.png)
- [Error analysis gallery](reports/error_analysis/error_gallery.md)

Calibration summary on the test predictions:

- Brier score: `0.0460`
- Expected calibration error: `0.0266`
- Maximum calibration error: `0.2119`

Robustness subset summary:

- Clean sampled accuracy: `0.950`
- Gaussian blur accuracy: `0.633`, review-routing rate: `1.000`
- Low contrast accuracy: `0.900`, review-routing rate: `1.000`
- Underexposure accuracy: `0.917`, review-routing rate: `1.000`
- Gaussian noise accuracy: `0.833`, sensitivity dropped to `0.700`

The noise result is intentionally documented as a remaining weakness: the current quality gate
catches blur, low contrast, and underexposure more clearly than synthetic high-frequency noise.

## Run The Streamlit Dashboard

From the repository root:

```bash
pip install -r requirements.txt
streamlit run malaria_App/app.py
```

The sidebar lets you choose:

- Local model inference
- FastAPI service inference
- Parasitized threshold
- Expert-review band
- Whether validation warnings should trigger review
- Whether Grad-CAM and activation maps should run
- Whether predictions should be logged

The dashboard has four main views:

- `Analysis Workbench`: upload a cell image, inspect quality checks, run inference, review Grad-CAM, and view telemetry.
- `Monitoring`: inspect recent review rate, validation-warning rate, quality-pass rate, and class mix.
- `Audit Log`: review recent local prediction records with correlation IDs and quality metrics.
- `System Notes`: summarize what the system demonstrates and where its clinical boundaries are.

## Run With Docker

Build and run the Streamlit dashboard:

```bash
docker build -t malaria-ai-xai .
docker run --rm -p 8501:8501 -v "%cd%/logs:/app/logs" malaria-ai-xai
```

Run both the dashboard and FastAPI service:

```bash
docker compose up --build
```

Then open:

- Dashboard: `http://127.0.0.1:8501`
- FastAPI docs: `http://127.0.0.1:8000/docs`

## Run The FastAPI Service

From the repository root:

```bash
pip install -r requirements.txt
uvicorn malaria_App.api:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@sample_cell.png" \
  -F "threshold=0.285" \
  -F "review_margin=0.075" \
  -F "include_xai=true" \
  -F "route_warnings_to_review=true" \
  -F "enable_logging=true"
```

The API returns prediction details, validation metadata, review routing, optional Grad-CAM images
as base64 PNG data URIs, model metadata, and logging status. The API also returns
`X-Correlation-ID` and `X-Process-Time-Ms` headers for traceability.

To pass your own correlation ID:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "X-Correlation-ID: demo-case-001" \
  -F "file=@sample_cell.png"
```

## Reproduce Evaluation

From the repository root:

```bash
python scripts/evaluate_threshold.py
```

The script downloads the dataset ZIP if needed, caches predictions in ignored local files, and
regenerates the evaluation report and figures under `reports/evaluation/`.

## Script-First Workflows

Train a new MobileNetV2 classifier from the public dataset:

```bash
python scripts/train_model.py --epochs 5 --batch-size 32
```

The training script downloads and extracts the dataset if needed, then saves the best model under
`models/`, which is ignored by Git. The committed app model remains at
`malaria_App/malaria_cell_parasite_prediction_model.h5`.

Run a single-image prediction from the command line:

```bash
python scripts/predict_image.py path/to/cell.png --threshold 0.285 --review-margin 0.075 --log
```

The CLI uses the same validation, quality scoring, thresholding, review routing, and logging
logic as the dashboard and API.

Generate calibration artifacts:

```bash
python scripts/calibration_analysis.py
```

Generate robustness artifacts from a deterministic test subset:

```bash
python scripts/robustness_analysis.py --max-samples 300
```

Generate false-positive and false-negative gallery items:

```bash
python scripts/error_analysis.py --limit-per-type 12
```

## Run Tests

```bash
python -m pytest tests -q
```

The current tests cover input validation, image-quality warnings, near-threshold review routing,
and API correlation-ID middleware.

## Responsible Use

This project is for education and research demonstration only. It is not medical advice, not a
clinical diagnostic system, and not intended for patient care. Any real diagnosis must be made
by qualified healthcare professionals using validated clinical workflows.

If deploying through a temporary public tunnel, do not upload patient-identifiable or sensitive
medical data.

## Future Work

- Add PR-AUC and confidence intervals for core metrics.
- Add Grad-CAM examples for true positives, true negatives, false positives, and false negatives.
- Add authenticated reviewer feedback capture for human-in-the-loop case review.
- Improve image-quality heuristics for noise and compression artifacts.
- Evaluate on an external dataset or institutionally separate holdout set.
