"""Server assembly.

Wires a backend through the executor into a standards-compliant A2A Starlette
app: JSON-RPC and REST bindings, the well-known agent card, and push
notifications so callers can register a webhook for long-running tasks instead
of holding a stream open.

Task storage is in memory unless ``task_db`` names a database. Persisting it
covers task history, artifacts, and push-notification registrations — not live
agent sessions, which are processes and die with the server either way.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:  # only installed with the persistence extra
    from sqlalchemy.ext.asyncio import AsyncEngine

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
    PushNotificationConfigStore,
    TaskStore,
)
from a2a.types import AgentCard
from starlette.applications import Starlette
from starlette.middleware import Middleware

from .auth import BearerAuthMiddleware
from .backends.base import Backend, ClosableBackend
from .card import build_card
from .executor import ClaudeCodeExecutor


class _Initializable(Protocol):
    """A store that creates its schema before first use."""

    async def initialize(self) -> None: ...


def _stores(
    task_db: str | None,
) -> tuple[AsyncEngine | None, TaskStore, PushNotificationConfigStore]:
    """Pick the task and push-config stores, plus the engine backing them.

    Imported lazily: the database stores pull SQLAlchemy and a driver, which
    only a deployment that asked for persistence should have to install.
    """
    if not task_db:
        return None, InMemoryTaskStore(), InMemoryPushNotificationConfigStore()

    from a2a.server.tasks import (
        DatabasePushNotificationConfigStore,
        DatabaseTaskStore,
    )
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(task_db)
    return (
        engine,
        DatabaseTaskStore(engine),
        DatabasePushNotificationConfigStore(engine),
    )


def build_app(
    backend: Backend,
    *,
    url: str,
    card_name: str | None = None,
    card_description: str | None = None,
    card_signer: Callable[[AgentCard], AgentCard] | None = None,
    auth_token: str | None = None,
    task_db: str | None = None,
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

    engine, task_store, push_config_store = _stores(task_db)
    push_client = httpx.AsyncClient()
    push_sender = BasePushNotificationSender(
        httpx_client=push_client,
        config_store=push_config_store,
        context=ServerCallContext(),
    )

    handler = DefaultRequestHandler(
        agent_executor=ClaudeCodeExecutor(backend),
        task_store=task_store,
        agent_card=card,
        push_config_store=push_config_store,
        push_sender=push_sender,
    )

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        if engine is not None:
            # Creates the tables on first run, and fails startup rather than the
            # first request if the database is unreachable. Only the database
            # stores have this; engine is not None exactly when they are in use.
            await cast(_Initializable, task_store).initialize()
            await cast(_Initializable, push_config_store).initialize()
        try:
            yield
        finally:
            # Nested so one failing close cannot strand the others, and in a
            # finally so a crash during serving still releases everything. A
            # backend may hold agent subprocesses; they must not outlive us.
            try:
                if isinstance(backend, ClosableBackend):
                    with suppress(Exception):
                        await backend.aclose()
            finally:
                with suppress(Exception):
                    await push_client.aclose()
                if engine is not None:
                    with suppress(Exception):
                        await engine.dispose()

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
