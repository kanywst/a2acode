"""Server assembly.

Wires a backend through the executor into a standards-compliant A2A Starlette
app: JSON-RPC and REST bindings, the well-known agent card, and push
notifications so callers can register a webhook for long-running tasks instead
of holding a stream open.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from starlette.applications import Starlette

from .backends.base import Backend
from .card import build_card
from .executor import ClaudeCodeExecutor


def build_app(backend: Backend, *, url: str) -> Starlette:
    card = build_card(url, streaming=True, push_notifications=True)

    push_config_store = InMemoryPushNotificationConfigStore()
    push_client = httpx.AsyncClient()
    push_sender = BasePushNotificationSender(
        httpx_client=push_client,
        config_store=push_config_store,
        context=ServerCallContext(),
    )

    handler = DefaultRequestHandler(
        agent_executor=ClaudeCodeExecutor(backend),
        task_store=InMemoryTaskStore(),
        agent_card=card,
        push_config_store=push_config_store,
        push_sender=push_sender,
    )

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await push_client.aclose()

    routes = [
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, rpc_url="/", enable_v0_3_compat=True),
        *create_rest_routes(handler, enable_v0_3_compat=True),
    ]
    return Starlette(routes=routes, lifespan=lifespan)
