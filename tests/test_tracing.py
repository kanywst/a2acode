"""Tracing.

Runs the executor under an in-memory span exporter so the span and its
attributes are checked without an OTLP collector. opentelemetry-sdk is a dev
dependency, so this exercises the real (non-no-op) path.
"""

from __future__ import annotations

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.types import Message, Part, Role, SendMessageRequest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from a2acode.backends import make_backend
from a2acode.executor import ClaudeCodeExecutor


@pytest.fixture
def exporter():
    # A global tracer provider can only be set once per process; clear any
    # existing one so this test gets its own and does not depend on order, and
    # restore the previous one afterwards so other tests are unaffected.
    prev_provider = getattr(trace, "_TRACER_PROVIDER", None)
    trace._TRACER_PROVIDER = None
    provider = TracerProvider()
    exp = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    # The module-level tracer is a proxy until a provider is set; setting it
    # here makes the spans land in our exporter.
    trace.set_tracer_provider(provider)
    yield exp
    trace._TRACER_PROVIDER = prev_provider


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_execute_emits_a_span_with_attributes(exporter):
    executor = ClaudeCodeExecutor(make_backend("echo"))
    message = Message(
        message_id="m1",
        role=Role.ROLE_USER,
        parts=[Part(text="hello")],
        task_id="task-1",
        context_id="ctx-1",
    )
    context = RequestContext(
        ServerCallContext(),
        request=SendMessageRequest(message=message),
        task_id="task-1",
        context_id="ctx-1",
    )
    await executor.execute(context, EventQueue())

    spans = exporter.get_finished_spans()
    execute_spans = [s for s in spans if s.name == "a2acode.execute"]
    assert len(execute_spans) == 1
    attrs = execute_spans[0].attributes
    assert attrs["a2a.task_id"] == "task-1"
    assert attrs["a2a.context_id"] == "ctx-1"
    assert attrs["a2acode.backend"] == "echo"
