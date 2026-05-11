from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import time
from pathlib import Path
from typing import Any

import requests


def timed_request(base_url: str, target: str, image_path: Path | None, api_key: str | None) -> dict[str, Any]:
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    start = time.perf_counter()
    try:
        if target == "health":
            response = requests.get(f"{base_url.rstrip('/')}/health", timeout=30)
        else:
            if image_path is None:
                raise ValueError("--image is required for predict load tests")
            with image_path.open("rb") as image_file:
                response = requests.post(
                    f"{base_url.rstrip('/')}/predict",
                    files={"file": (image_path.name, image_file, "image/png")},
                    data={"include_xai": "false", "enable_logging": "false"},
                    headers=headers,
                    timeout=60,
                )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"ok": response.ok, "status_code": response.status_code, "elapsed_ms": elapsed_ms}
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"ok": False, "status_code": "error", "elapsed_ms": elapsed_ms, "error": str(exc)}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight API load/scalability smoke test.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--target", choices=["health", "predict"], default="health")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(timed_request, args.base_url, args.target, args.image, args.api_key)
            for _ in range(args.requests)
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    latencies = [float(result["elapsed_ms"]) for result in results]
    successes = sum(1 for result in results if result["ok"])
    summary = {
        "target": args.target,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "successes": successes,
        "failures": args.requests - successes,
        "success_rate": successes / args.requests if args.requests else 0.0,
        "latency_ms": {
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": max(latencies) if latencies else 0.0,
        },
    }
    print(summary)


if __name__ == "__main__":
    main()
