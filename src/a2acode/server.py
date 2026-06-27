"""Server assembly.

Wires a backend through the executor into a standards-compliant A2A Starlette
app: JSON-RPC and REST bindings, the well-known agent card, and push
notifications so callers can register a webhook for long-running tasks instead
of holding a stream open.
"""

from __future__ import annotations

from collections.abc import Callable
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
from a2a.types import AgentCard
from starlette.applications import Starlette
from starlette.middleware import Middleware

from .auth import BearerAuthMiddleware
from .backends.base import Backend
from .card import build_card
from .executor import ClaudeCodeExecutor


def build_app(
    backend: Backend,
    *,
    url: str,
    card_name: str | None = None,
    card_description: str | None = None,
    card_signer: Callable[[AgentCard], AgentCard] | None = None,
    auth_token: str | None = None,
) -> Starlette:
    card = build_card(
        url,
        name=card_name or "Claude Code",
        description=card_description,
        streaming=True,
        push_notifications=True,
        require_auth=auth_token is not None,
    )
    if card_signer is not None:
        card = card_signer(card)

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
    middleware = (
        [Middleware(BearerAuthMiddleware, token=auth_token)]
        if auth_token is not None
        else None
    )
    return Starlette(routes=routes, lifespan=lifespan, middleware=middleware)
