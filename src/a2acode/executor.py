"""Protocol mapping.

Translates a backend's normalized event stream into A2A task lifecycle events:

    text                -> a streamed artifact (append / last_chunk)
    tool use            -> a working-state status update describing the action
    file change         -> a named artifact carrying the diff
    permission request  -> an input-required pause the caller answers
    result              -> run metadata on the completion message + continuity

A task that pauses on a permission request keeps its backend session alive in a
registry; the caller's follow-up message (same task id) carries the decision and
resumes the same session. Session ids are mapped to the A2A ``context_id`` so a
new task in the same context resumes the same Claude conversation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus

from .backends.base import (
    Backend,
    FileChange,
    PermissionDecision,
    PermissionRequest,
    Result,
    RunRequest,
    TextDelta,
    ToolUse,
)
from .backends.session import BackendSession
from .tracing import span

logger = logging.getLogger(__name__)

_ALLOW_WORDS = {"allow", "yes", "y", "approve", "ok", "accept", "grant"}

# Bound the in-memory maps so a long-running server cannot grow without limit
# (e.g. from many contexts, or tasks left paused on a permission and never
# answered). The continuity cache (_MAX_CONTEXTS) evicts its least-recently-used
# entry; the live-session map (_MAX_LIVE) evicts a parked session first, else the
# oldest entry.
_MAX_CONTEXTS = 4096
_MAX_LIVE = 256


@dataclass
class _Stream:
    """Response-stream state for a task, persisted across permission pauses."""

    artifact_id: str
    chunks: list[str] = field(default_factory=list)
    pending: str | None = None
    sent_first: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


def _build_prompt(context: RequestContext) -> str:
    """Assemble the prompt from every part of the incoming message.

    Text parts are concatenated; file, URL, and structured-data parts are
    surfaced to Claude as labelled references so an attachment is not silently
    dropped on the floor.
    """
    message = getattr(context, "message", None)
    parts = getattr(message, "parts", None) if message is not None else None
    if not parts:
        return context.get_user_input() or ""

    texts: list[str] = []
    refs: list[str] = []
    for part in parts:
        which = part.WhichOneof("content")
        if which == "text":
            if part.text:
                texts.append(part.text)
        elif which == "url":
            label = part.filename or part.url
            refs.append(f"[attached {label} ({part.media_type or 'file'}): {part.url}]")
        elif which == "raw":
            name = part.filename or "unnamed"
            kind = part.media_type or "application/octet-stream"
            refs.append(f"[attached file {name} ({kind}, {len(part.raw)} bytes)]")
        elif which == "data":
            refs.append(f"[attached data ({part.media_type or 'application/json'})]")

    prompt = "\n".join(texts)
    if refs:
        prompt = f"{prompt}\n\n" + "\n".join(refs)
    return prompt.strip()


def _describe_tool(event: ToolUse) -> str:
    """A short, human-readable line for a tool invocation."""
    i = event.tool_input
    if event.name == "Bash":
        return f"$ {str(i.get('command', '')).strip()[:120]}"
    path = i.get("file_path") or i.get("path") or i.get("pattern")
    if path:
        return f"{event.name} {path}"
    return event.name


class ClaudeCodeExecutor(AgentExecutor):
    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        # context_id -> claude session id, for resuming a new task in a context.
        self._session_ids: dict[str, str] = {}
        # task_id -> live session, for resuming a task paused on a permission.
        self._live: dict[str, BackendSession] = {}
        # task_id -> response-stream state, kept across permission pauses.
        self._streams: dict[str, _Stream] = {}
        # Serializes capacity eviction with new-session registration so a burst
        # of concurrent first-turn requests cannot each see a slot freed by one
        # eviction and collectively overshoot _MAX_LIVE. Uncontended below
        # capacity, where _evict_if_full returns without awaiting.
        self._admit_lock = asyncio.Lock()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # span() drops None-valued attributes, so no fallbacks are needed; the
        # ids are populated by the SDK before execute is called.
        with span(
            "a2acode.execute",
            **{
                "a2a.task_id": context.task_id,
                "a2a.context_id": context.context_id,
                "a2acode.backend": self._backend.name,
            },
        ):
            await self._execute(context, event_queue)

    async def _execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id = context.task_id, context.context_id
        assert task_id is not None and context_id is not None
        updater = TaskUpdater(event_queue, task_id, context_id)
        session = self._live.get(task_id)

        if session is None:
            # The stream MUST open with a Task object before any status update.
            await event_queue.enqueue_event(
                Task(
                    id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                )
            )
            await updater.start_work()
            request = RunRequest(
                prompt=_build_prompt(context),
                context_id=context_id,
                resume=self._session_ids.get(context_id),
            )
            async with self._admit_lock:
                await self._evict_if_full()
                session = BackendSession()
                session.start(lambda s: self._backend.drive(s, request))
                self._live[task_id] = session
        else:
            # Follow-up to an input-required pause: the message is the decision.
            # Guard against a concurrent message arriving while the task is still
            # running; resolving and pumping a non-parked session would put two
            # consumers on the same queue and lose events.
            if not session.is_parked:
                raise RuntimeError(
                    f"task {task_id} is already running and not awaiting input"
                )
            await updater.start_work()
            session.resolve(self._decision(context, session))

        await self._pump(updater, task_id, context_id, session)

    async def _pump(
        self,
        updater: TaskUpdater,
        task_id: str,
        context_id: str,
        session: BackendSession,
    ) -> None:
        # One stream of artifacts/text per task, kept across permission pauses so
        # the response stays a single artifact and the completion text is whole.
        stream = self._streams.setdefault(task_id, _Stream(artifact_id=uuid4().hex))

        async def flush(text: str, *, last: bool) -> None:
            await updater.add_artifact(
                [Part(text=text)],
                artifact_id=stream.artifact_id,
                name="response",
                append=stream.sent_first,
                last_chunk=last,
            )
            stream.sent_first = True

        try:
            async for event in session.drain():
                if isinstance(event, TextDelta):
                    if stream.pending is not None:
                        stream.chunks.append(stream.pending)
                        await flush(stream.pending, last=False)
                    stream.pending = event.text
                elif isinstance(event, ToolUse):
                    await updater.update_status(
                        TaskState.TASK_STATE_WORKING,
                        message=updater.new_agent_message(
                            [Part(text=_describe_tool(event))]
                        ),
                    )
                elif isinstance(event, FileChange):
                    await updater.add_artifact(
                        [Part(text=event.diff, media_type="text/x-diff")],
                        name=event.path,
                    )
                elif isinstance(event, PermissionRequest):
                    if stream.pending is not None:
                        stream.chunks.append(stream.pending)
                        await flush(stream.pending, last=False)
                        stream.pending = None
                    await self._request_input(updater, event)
                elif isinstance(event, Result):
                    stream.metadata = self._result_metadata(event)
                    if event.session_id:
                        self._remember_session(context_id, event.session_id)
        except asyncio.CancelledError:
            # Client disconnected / timed out: drop the session and its runner
            # instead of leaking them. Synchronous cleanup since we are cancelled.
            self._live.pop(task_id, None)
            self._streams.pop(task_id, None)
            session.abort()
            raise
        except Exception:  # noqa: BLE001 (surface failure without leaking details)
            logger.exception("backend run failed for task %s", task_id)
            await self._discard(task_id, session)
            await updater.failed(
                message=updater.new_agent_message(
                    [Part(text="Claude Code run failed; see server logs.")]
                )
            )
            return

        if not session.done:
            # Paused on a permission request; keep the stream for the follow-up.
            return

        if session.evicted:
            # The session was dropped to free a capacity slot while still
            # running, so its drain ended on the cancellation sentinel rather
            # than a real result. Fail the task instead of presenting the partial
            # buffer as a completed run.
            await updater.failed(
                message=updater.new_agent_message(
                    [Part(text="Task evicted to free server capacity.")]
                )
            )
            return

        if stream.pending is not None:
            stream.chunks.append(stream.pending)
            await flush(stream.pending, last=True)
        elif stream.sent_first:
            # No new text this turn, but earlier chunks went out, so close the
            # artifact so it is not left without a final chunk.
            await flush("", last=True)

        await self._discard(task_id, session)
        full_text = "".join(stream.chunks) or "(no text output)"
        await updater.complete(
            message=updater.new_agent_message(
                [Part(text=full_text)], metadata=stream.metadata or None
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id = context.task_id, context.context_id
        assert task_id is not None and context_id is not None
        self._streams.pop(task_id, None)
        session = self._live.pop(task_id, None)
        if session is not None:
            await session.close()
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()

    @staticmethod
    async def _request_input(updater: TaskUpdater, event: PermissionRequest) -> None:
        line = event.description or event.tool_name
        await updater.requires_input(
            message=updater.new_agent_message(
                [Part(text=f"Permission requested for {event.tool_name}: {line}")],
                metadata={
                    "a2acode_permission": {
                        "request_id": event.request_id,
                        "tool": event.tool_name,
                        "input": event.tool_input,
                    }
                },
            )
        )

    @staticmethod
    def _decision(
        context: RequestContext, session: BackendSession
    ) -> PermissionDecision:
        text = (context.get_user_input() or "").strip().lower()
        allow = text in _ALLOW_WORDS or text.startswith("allow")
        return PermissionDecision(
            request_id=session.last_request_id or "",
            allow=allow,
            message="" if allow else "Denied by A2A caller",
        )

    def _remember_session(self, context_id: str, session_id: str) -> None:
        # Re-insert so the most recently used context moves to the end of the
        # dict: a plain reassignment keeps an existing key in its original
        # position, which would let an actively reused context be evicted before
        # an idle, more-recently-created one. Pop-then-set makes eviction LRU.
        self._session_ids.pop(context_id, None)
        self._session_ids[context_id] = session_id
        while len(self._session_ids) > _MAX_CONTEXTS:
            oldest = next(iter(self._session_ids))
            del self._session_ids[oldest]

    async def _evict_if_full(self) -> None:
        """Make room when at capacity.

        Prefer evicting a parked session (an input-required task the caller
        abandoned without answering) over one still actively running: a parked
        task's ``_pump`` has already returned, so dropping it just closes a
        stalled session. Only when nothing is parked do we fall back to the
        oldest entry, which is a running task; mark it evicted so its ``_pump``
        fails the task rather than completing it with partial output.
        """
        while len(self._live) >= _MAX_LIVE:
            # Fall back lazily to the oldest entry: passing next(iter(...)) as the
            # default would evaluate it even when a parked session is found.
            victim = next((tid for tid, s in self._live.items() if s.is_parked), None)
            if victim is None:
                victim = next(iter(self._live))
            logger.warning("evicting live task %s at capacity", victim)
            session = self._live[victim]
            session.evicted = True
            await self._discard(victim, session)

    async def _discard(self, task_id: str, session: BackendSession) -> None:
        self._live.pop(task_id, None)
        self._streams.pop(task_id, None)
        await session.close()

    @staticmethod
    def _result_metadata(event: Result) -> dict[str, object]:
        meta: dict[str, object] = {}
        if event.session_id is not None:
            meta["claude_session_id"] = event.session_id
        if event.cost_usd is not None:
            meta["cost_usd"] = event.cost_usd
        if event.num_turns is not None:
            meta["num_turns"] = event.num_turns
        if event.usage is not None:
            meta["usage"] = event.usage
        return meta
