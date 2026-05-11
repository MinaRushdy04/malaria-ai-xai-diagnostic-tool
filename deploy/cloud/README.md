# Cloud Deployment Notes

This repo is ready for an API-first container deployment. The practical cloud path is:

1. Build the Docker image.
2. Push it to a registry such as GHCR.
3. Run the container on a managed platform.
4. Set `MALARIA_API_KEY` as a secret.
5. Mount or externalize logs.
6. Put TLS and authentication in front of the service.

## Reasonable Targets

- Render, Railway, Fly.io, or Azure Container Apps for a simple managed container demo.
- AWS ECS/Fargate or Google Cloud Run for serverless container deployment.
- AKS/EKS/GKE if demonstrating Kubernetes orchestration.

## Required Runtime Configuration

```text
PORT=8000
MALARIA_API_KEY=<secret>
MALARIA_LOG_DIR=/app/logs
```

## Important Boundary

The repo does not include real patient-data controls, PHI storage, IAM policies, TLS certificates,
or a managed database. For a serious cloud deployment, replace local SQLite/CSV logs with a secure
database, add authentication/authorization, and define a retention policy.
