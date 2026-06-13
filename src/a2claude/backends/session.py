"""Session machinery shared by all backends.

A backend's ``drive`` coroutine runs in a background task and pushes events onto
the session. The consumer (the executor) reads them with ``drain``, which stops
either when the run finishes or when it surfaces a permission request — at which
point the background task is parked inside ``request_permission`` waiting for a
decision. A later ``resolve`` un-parks it. This decoupling is what allows the
A2A ``input-required`` round trip to span two separate ``execute`` calls while
the Claude session stays alive in between.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .base import BackendEvent, PermissionDecision, PermissionRequest


@dataclass(slots=True)
class _Error:
    exc: BaseException


_DONE = object()

# The event loop holds only a weak reference to tasks created with
# create_task; keep a strong one so a runner can't be garbage-collected
# mid-flight (e.g. after its session is dropped on a client disconnect).
_RUNNERS: set[asyncio.Task[None]] = set()


class BackendSession:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._pending: dict[str, asyncio.Future[PermissionDecision]] = {}
        self._runner: asyncio.Task[None] | None = None
        self.last_request_id: str | None = None
        self.done = False

    @property
    def is_parked(self) -> bool:
        """Whether the run is currently waiting for a permission decision."""
        return (
            self.last_request_id is not None and self.last_request_id in self._pending
        )

    def start(self, driver: Callable[[BackendSession], Awaitable[None]]) -> None:
        async def runner() -> None:
            try:
                await driver(self)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001 — relayed to consumer
                # put_nowait (the queue is unbounded) so a pending cancellation
                # cannot stop the sentinel from reaching a blocked drain().
                self._queue.put_nowait(_Error(exc))
            finally:
                self._queue.put_nowait(_DONE)

        self._runner = asyncio.create_task(runner())
        _RUNNERS.add(self._runner)
        self._runner.add_done_callback(_RUNNERS.discard)

    async def emit(self, event: BackendEvent) -> None:
        await self._queue.put(event)

    async def request_permission(
        self, tool_name: str, tool_input: dict[str, Any], description: str = ""
    ) -> PermissionDecision:
        request_id = uuid4().hex
        future: asyncio.Future[PermissionDecision] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[request_id] = future
        await self._queue.put(
            PermissionRequest(request_id, tool_name, dict(tool_input), description)
        )
        return await future

    def resolve(self, decision: PermissionDecision) -> None:
        future = self._pending.pop(decision.request_id, None)
        if future is None:
            # Fail fast: a no-op would leave the driver parked and hang drain().
            raise ValueError(f"no pending permission request {decision.request_id!r}")
        if not future.done():
            future.set_result(decision)

    async def drain(self) -> AsyncIterator[BackendEvent]:
        """Yield events until the run ends or a permission request is surfaced.

        A permission request is yielded and then stops the iteration, leaving the
        background task parked until ``resolve`` is called and ``drain`` resumes.
        """
        while True:
            item = await self._queue.get()
            if item is _DONE:
                self.done = True
                return
            if isinstance(item, _Error):
                self.done = True
                raise item.exc
            yield item
            if isinstance(item, PermissionRequest):
                self.last_request_id = item.request_id
                return

    def abort(self) -> None:
        """Cancel the background runner without awaiting — safe to call from a
        cancelled context (e.g. an interrupted ``execute``)."""
        if self._runner is not None and not self._runner.done():
            self._runner.cancel()

    async def close(self) -> None:
        try:
            if self._runner is not None and not self._runner.done():
                self._runner.cancel()
                with suppress(asyncio.CancelledError):
                    await self._runner
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.cancel()
            self._pending.clear()
