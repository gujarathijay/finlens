"""
Prometheus metrics for FinLens.

Tracks: request count, latency, guardrail pass/fail, extraction counts.
Exposes a /metrics endpoint that Prometheus scrapes.

Usage:
    from src.finlens.monitoring.metrics import track_request, metrics_app
"""

from prometheus_client import Counter, Histogram, make_asgi_app

# ── Counters (things that only go up) ──

REQUEST_COUNT = Counter(
    "finlens_requests_total",
    "Total extraction requests",
    ["status"],  # labels: success, failed, flagged
)

GUARDRAIL_FAILURES = Counter(
    "finlens_guardrail_failures_total",
    "Total guardrail failures",
    ["check"],  # labels: json_parse, schema, hallucination, pii
)

# ── Histograms (distributions of values) ──

REQUEST_LATENCY = Histogram(
    "finlens_request_latency_seconds",
    "Request latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

EXTRACTION_COUNT = Histogram(
    "finlens_extraction_items",
    "Number of items extracted per request",
    buckets=[0, 1, 2, 3, 5, 8, 10, 15, 20],
)


def track_request(status: str, latency_ms: float, num_items: int, guardrail_failures: list[str]):
    """Record metrics for one request."""
    REQUEST_COUNT.labels(status=status).inc()
    REQUEST_LATENCY.observe(latency_ms / 1000)  # convert ms to seconds
    EXTRACTION_COUNT.observe(num_items)

    for failure in guardrail_failures:
        GUARDRAIL_FAILURES.labels(check=failure).inc()


# ASGI app that serves /metrics endpoint
metrics_app = make_asgi_app()