"""Distributed tracing.

A2A runs over HTTP, so it slots into standard OpenTelemetry tracing: the SDK
already instruments its own client/server/task paths, and this adds a span for
a2acode's own protocol-mapping layer so a trace shows where time went inside
the executor, not just in the SDK.

OpenTelemetry is an optional dependency (install ``a2acode[telemetry]``). When
it is absent, ``span`` is a no-op context manager, so the core install and the
hot path stay free of the dependency. Trace context propagation over HTTP
headers is handled by the standard OTel instrumentation, not here.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:
    from opentelemetry import trace

    _tracer: Any | None = trace.get_tracer("a2acode")
except ImportError:  # opentelemetry not installed: tracing is a no-op
    _tracer = None


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """Run the block inside a span named ``name`` if tracing is available.

    ``None``-valued attributes are dropped so they do not clutter the span.
    """
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield
