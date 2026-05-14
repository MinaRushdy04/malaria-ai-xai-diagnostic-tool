from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware

try:
    from .diagnostic_core import write_api_request_metric
except ImportError:
    from diagnostic_core import write_api_request_metric


CORRELATION_ID_HEADER = "X-Correlation-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if self._should_record_metric(request.url.path):
                self._write_metric(
                    correlation_id=correlation_id,
                    method=request.method,
                    path=request.url.path,
                    status_code=500,
                    elapsed_ms=elapsed_ms,
                )
            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        response.headers[PROCESS_TIME_HEADER] = f"{elapsed_ms:.2f}"
        if self._should_record_metric(request.url.path):
            self._write_metric(
                correlation_id=correlation_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                elapsed_ms=elapsed_ms,
            )
        return response

    @staticmethod
    def _should_record_metric(path: str) -> bool:
        return not (path.startswith("/dashboard") or path == "/favicon.ico")

    @staticmethod
    def _write_metric(
        *,
        correlation_id: str,
        method: str,
        path: str,
        status_code: int,
        elapsed_ms: float,
    ) -> None:
        try:
            write_api_request_metric(
                correlation_id=correlation_id,
                method=method,
                path=path,
                status_code=status_code,
                elapsed_ms=elapsed_ms,
            )
        except Exception:
            pass
