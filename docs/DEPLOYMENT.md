# Deployment

The primary deployment path is now FastAPI-first. Streamlit is still available as a research UI,
but the default container starts the API and serves a lightweight browser dashboard from the same
service.

## Runtime Shape

```text
Browser
  `-- /dashboard/ static assets
        `-- fetch /predict, /health, /monitoring/summary, /review/queue

FastAPI container
  |-- validation and quality gate
  |-- TensorFlow model inference
  |-- Grad-CAM generation
  |-- review routing
  |-- SQLite and CSV logs mounted at /app/logs
  `-- reviewer feedback API
```

This keeps the deployment closer to a normal service architecture:

- one backend service owns inference and safety policy,
- the browser UI is replaceable,
- external clients can call the same endpoints as the dashboard,
- Docker Compose can run the API without requiring Streamlit,
- Streamlit can still be launched through the `research-ui` profile for local inspection.

## Local API Deployment

```bash
uvicorn malaria_App.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/dashboard/
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Docker Deployment

```bash
docker compose up --build
```

Open:

```text
http://127.0.0.1:8000/dashboard/
```

Run the optional Streamlit research UI:

```bash
docker compose --profile research-ui up --build
```

## Health Check

The Compose API service includes a health check:

```text
GET /health
```

The health payload includes model load status, model version, model hash, default threshold, and
default review margin.

## Deployment Boundary

This is still an academic prototype. The deployment demonstrates separation of concerns,
containerization, API contracts, traceability, and human-review workflow. It does not provide
authentication, PHI-grade storage, regulated audit controls, hospital integration, or clinical
validation.
