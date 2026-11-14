"""OpenTelemetry tracing. Spans follow the GenAI semantic conventions where they exist.

Without configuration the OpenTelemetry API is a no-op. Set
APPLICATIONINSIGHTS_CONNECTION_STRING (and install the `monitor` extra) to export traces,
including FastAPI and HTTP client spans, to Azure Monitor / Application Insights.
Customer messages are not recorded, only their length.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

log = logging.getLogger(__name__)
tracer = trace.get_tracer("retail_rag")

GEN_AI_SYSTEM = "az.ai.openai"
_configured = False


def configure_telemetry(connection_string: str | None = None) -> bool:
    """Enable Azure Monitor export once, if a connection string is present.

    Pass it from Settings: a value that lives only in .env is not in os.environ.
    """
    global _configured
    connection_string = connection_string or os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if _configured or not connection_string:
        return _configured
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
    except ImportError:
        log.warning(
            "APPLICATIONINSIGHTS_CONNECTION_STRING is set but the 'monitor' extra "
            "is not installed; traces are not exported"
        )
        return False
    configure_azure_monitor(connection_string=connection_string, logger_name="retail_rag")
    _configured = True
    return True


@contextmanager
def span(name: str, **attributes):
    """Start a span; exceptions are recorded and re-raised."""
    with tracer.start_as_current_span(name) as s:
        for k, v in attributes.items():
            if v is not None:
                s.set_attribute(k, v)
        try:
            yield s
        except Exception as exc:
            s.record_exception(exc)
            s.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise


def record_usage(s, usage) -> None:
    if usage is None:
        return
    s.set_attribute("gen_ai.usage.input_tokens", getattr(usage, "prompt_tokens", 0) or 0)
    s.set_attribute("gen_ai.usage.output_tokens", getattr(usage, "completion_tokens", 0) or 0)
