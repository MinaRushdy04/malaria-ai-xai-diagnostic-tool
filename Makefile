.PHONY: install test compile app api calibration robustness error-gallery docker-build docker-up

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

compile:
	python -m py_compile malaria_App/app.py malaria_App/diagnostic_core.py malaria_App/api.py malaria_App/middleware.py malaria_App/schemas.py scripts/evaluate_threshold.py scripts/predict_image.py scripts/train_model.py scripts/calibration_analysis.py scripts/robustness_analysis.py scripts/error_analysis.py tests/test_core_safety.py

test: compile
	python -m pytest tests -q

app:
	streamlit run malaria_App/app.py

api:
	uvicorn malaria_App.api:app --reload

calibration:
	python scripts/calibration_analysis.py

robustness:
	python scripts/robustness_analysis.py --max-samples 300

error-gallery:
	python scripts/error_analysis.py --limit-per-type 12

docker-build:
	docker build -t malaria-ai-xai .

docker-up:
	docker compose up --build
