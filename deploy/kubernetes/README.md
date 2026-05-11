# Kubernetes Deployment

These manifests are a portfolio-grade orchestration path for the FastAPI inference service.

```bash
kubectl apply -k deploy/kubernetes
```

The deployment exposes:

- FastAPI inference service
- `/dashboard/` browser dashboard
- `/health` readiness/liveness checks
- optional API-key protection through `MALARIA_API_KEY`
- resource requests/limits
- service and ingress manifests
- HPA scaffold

## Security

Copy `secret.example.yaml` into an environment-specific secret file and replace the placeholder:

```bash
kubectl create secret generic malaria-ai-secret \
  --namespace malaria-ai \
  --from-literal=MALARIA_API_KEY="<long-random-value>"
```

Do not commit real secrets.

## Scaling Boundary

The API container can be replicated for stateless inference, but the current audit/review storage
uses local SQLite/CSV logs. For real multi-replica serving, move logs and review feedback to a
shared database such as Postgres and store images/artifacts in object storage. Until then, keep
`replicas: 1` for traceable local demo behavior.
