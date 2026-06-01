"""OpenTelemetry observability bootstrap.

Mirrors the Go side (utils/observability):
  - Traces  : OTLP-HTTP exporter → Jaeger (env `OTEL_EXPORTER_OTLP_ENDPOINT`, e.g.
              `jaeger.observability.svc.cluster.local:4318`). Accepts host:port or
              full URL with scheme.
  - Metrics : OpenTelemetry MeterProvider with a Prometheus pull exporter; long-
              running services pass `prometheus_port=…` to start a sidecar HTTP
              server, or `fastapi_app=…` to mount /metrics on the existing app.
  - Logs    : JSON to stdout with trace_id/span_id injected from the current span,
              so log lines correlate with traces in Jaeger.

No-op when `OTEL_EXPORTER_OTLP_ENDPOINT` is empty — matches the Go convention so
local dev without Jaeger keeps working.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import socket
import sys
from typing import Callable, Iterable, Optional

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

# Standard LogRecord attributes — anything else on the record gets emitted as
# a top-level JSON field (so `logger.info("x", extra={"k": v})` shows up).
_STD_LOG_FIELDS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
})


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        span = trace.get_current_span()
        ctx = span.get_span_context() if span else None
        if ctx is not None and ctx.is_valid:
            out["trace_id"] = format(ctx.trace_id, "032x")
            out["span_id"] = format(ctx.span_id, "016x")
        for k, v in record.__dict__.items():
            if k not in _STD_LOG_FIELDS and not k.startswith("_"):
                out[k] = v
        return json.dumps(out, default=str)


def _setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


def _normalize_otlp_endpoint(raw: str) -> str:
    """Allow `host:port` (Go-side convention) or full URL with scheme."""
    raw = raw.strip()
    if not raw:
        return ""
    if raw.startswith(("http://", "https://")):
        return raw
    return f"http://{raw}"


_initialized = False
_shutdown_callbacks: list[Callable[[], None]] = []


def init_observability(
    service_name: str,
    *,
    prometheus_port: Optional[int] = None,
    fastapi_app=None,
    instrumentors: Optional[Iterable[object]] = None,
) -> Callable[[], None]:
    """Initialize traces, metrics, structured logging. Returns a shutdown function.

    Idempotent: a second call returns the shutdown registered by the first.

    Args:
        service_name: `service.name` resource attribute (overridden by
            `OTEL_SERVICE_NAME` env if set).
        prometheus_port: long-running consumers/jobs pass this to spin up an
            HTTP server exposing `/metrics`. Skip for FastAPI (use
            `fastapi_app=`) and for short-lived batch jobs.
        fastapi_app: FastAPI app instance — when provided, a `/metrics` route
            is registered on it (no separate port).
        instrumentors: already-constructed instrumentor instances to apply
            (e.g. [FastAPIInstrumentor(), HTTPXClientInstrumentor()]).
    """
    global _initialized
    _setup_logging()

    if _initialized:
        return _shutdown_all

    raw_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    endpoint = _normalize_otlp_endpoint(raw_endpoint)

    if not endpoint:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set; observability disabled")
        _initialized = True
        return _shutdown_all

    # Put the normalized URL back so any downstream SDK reading the env
    # variable directly (e.g. opentelemetry-instrument bootstrap) sees it.
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint

    resource = Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME", service_name),
        "service.instance.id": socket.gethostname(),
    })

    tracer_provider = TracerProvider(resource=resource)
    # OTLPSpanExporter auto-appends `/v1/traces` to the env-derived endpoint.
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    prom_reader = PrometheusMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[prom_reader])
    metrics.set_meter_provider(meter_provider)

    if fastapi_app is not None:
        _mount_fastapi_metrics(fastapi_app)
    if prometheus_port is not None:
        from prometheus_client import start_http_server
        start_http_server(prometheus_port)
        logger.info("Prometheus metrics on :%d/metrics", prometheus_port)

    for inst in instrumentors or []:
        try:
            inst.instrument()
        except Exception:
            logger.exception("Failed to apply instrumentor %s", type(inst).__name__)

    _shutdown_callbacks.append(tracer_provider.shutdown)
    _shutdown_callbacks.append(meter_provider.shutdown)
    atexit.register(_shutdown_all)
    _initialized = True
    logger.info("Observability initialized (service=%s, endpoint=%s)", service_name, endpoint)
    return _shutdown_all


def _mount_fastapi_metrics(app) -> None:
    """Register a /metrics route serving the Prometheus exposition format."""
    from fastapi import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    @app.get("/metrics", include_in_schema=False)
    def _metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _shutdown_all() -> None:
    for cb in _shutdown_callbacks:
        try:
            cb()
        except Exception:
            logger.exception("observability shutdown step failed")
    _shutdown_callbacks.clear()


def get_tracer(name: str):
    return trace.get_tracer(name)


def get_meter(name: str):
    return metrics.get_meter(name)
