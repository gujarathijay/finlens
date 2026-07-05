"""OpenTelemetry tracing for FinLens."""

import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

resource = Resource.create({"service.name": "finlens-api"})
provider = TracerProvider(resource=resource)

# Only export traces if OTLP endpoint is reachable
otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
if otlp_endpoint:
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    except Exception:
        pass  # no collector running, skip silently

trace.set_tracer_provider(provider)
tracer = trace.get_tracer("finlens")