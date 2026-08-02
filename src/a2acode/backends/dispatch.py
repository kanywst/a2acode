"""Ordered dispatch of an ACP agent's notifications.

The library's default dispatcher spawns a task per incoming message and marks
the queue item done as soon as that task exists. For a request that is right —
a permission request parks until the A2A caller answers, and running it inline
would stall every message behind it. For a ``session/update`` it costs the two
guarantees a turn depends on:

* **Order.** Text chunks are only an answer in the order they were sent.
* **Completion.** A prompt's reply is handled inline by the receive loop, so it
  can overtake notification tasks that have been created but not run. The turn
  then ends while the agent's last words are still in flight, and they are lost
  behind the end-of-stream sentinel.

Handling notifications inline restores both, and makes the queue's ``join()``
mean what the backend needs it to mean: everything the agent said has been
turned into events.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from acp.task import RpcTaskKind

logger = logging.getLogger(__name__)


class OrderedDispatcher:
    """Await notifications in arrival order; keep requests concurrent."""

    def __init__(
        self,
        queue: Any,
        supervisor: Any,
        store: Any,
        run_request: Callable[[dict[str, Any]], Awaitable[Any]],
        run_notification: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        self._queue = queue
        self._supervisor = supervisor
        self._store = store
        self._run_request = run_request
        self._run_notification = run_notification
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = self._supervisor.create(self._run(), name="a2acode.dispatch")

    async def stop(self) -> None:
        await self._queue.close()
        if self._task is not None:
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        try:
            async for task in self._queue:
                try:
                    if task.kind is RpcTaskKind.REQUEST:
                        self._spawn_request(task.message)
                    else:
                        await self._run_notification(task.message)
                except Exception:
                    # Handling notifications inline means one that raises would
                    # otherwise end this loop, and the connection would go
                    # silent for the rest of its life.
                    logger.exception("dropping an ACP notification that failed")
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            return

    def _spawn_request(self, message: dict[str, Any]) -> None:
        record = self._store.begin_incoming(
            message.get("method", ""), message.get("params")
        )

        async def runner() -> None:
            try:
                result = await self._run_request(message)
            except Exception as exc:
                self._store.fail_incoming(record, exc)
                raise
            else:
                self._store.complete_incoming(record, result)

        self._supervisor.create(runner(), name="a2acode.dispatch.request")
