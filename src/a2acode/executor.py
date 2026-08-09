"""Protocol mapping.

Translates a backend's normalized event stream into A2A task lifecycle events:

    text                -> a streamed artifact (append / last_chunk)
    thought             -> a separate "thinking" artifact, never the answer
    tool use            -> a working-state status update describing the action
    tool result         -> a working-state status update carrying the outcome
    file change         -> a named artifact carrying the diff
    plan                -> a "plan" artifact, replaced on every update
    notice              -> a working-state status update about the run itself
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
from google.protobuf import json_format

from .backends.base import (
    Attachment,
    Backend,
    FileChange,
    Notice,
    PermissionDecision,
    PermissionRequest,
    Plan,
    Result,
    RunRequest,
    TextDelta,
    Thought,
    ToolResult,
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

# Checklist markers for a plan step's status; anything else renders as open.
_PLAN_MARKS = {"completed": "x", "in_progress": ">"}

# Caps on what a caller can attach to one turn. Text is inlined into the
# prompt, so it competes with the work for the context window; binary travels
# as its own content block and gets the headroom a screenshot needs.
_MAX_TEXT_ATTACHED = 64 * 1024
_MAX_TEXT_TOTAL = 256 * 1024
_MAX_BINARY_ATTACHED = 4 * 1024 * 1024
_MAX_BINARY_TOTAL = 8 * 1024 * 1024

# Media types that are text despite not being under text/*.
_TEXTUAL_TYPES = {
    "application/json",
    "application/x-ndjson",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
    "application/javascript",
    "application/x-sh",
    "application/sql",
    "application/x-patch",
    "application/toml",
}


@dataclass
class _Stream:
    """Response-stream state for a task, persisted across permission pauses."""

    artifact_id: str
    chunks: list[str] = field(default_factory=list)
    pending: str | None = None
    sent_first: bool = False
    metadata: dict[str, object] = field(default_factory=dict)
    # tool_use_id -> name, for a ToolResult that omits the title.
    tool_names: dict[str, str] = field(default_factory=dict)
    # Reused so each plan update replaces the last instead of stacking a copy.
    plan_artifact_id: str = ""
    # Reasoning streams into its own artifact, never into the answer.
    thinking_artifact_id: str = ""
    sent_thought: bool = False


def _is_textual(media_type: str) -> bool:
    return media_type.startswith("text/") or media_type in _TEXTUAL_TYPES


def _build_input(context: RequestContext) -> tuple[str, list[Attachment]]:
    """Split the incoming message into prompt text and attachments.

    A caller sending a log, a patch, or a screenshot gets it read, rather than
    reduced to a note that something was attached.
    """
    message = getattr(context, "message", None)
    parts = getattr(message, "parts", None) if message is not None else None
    if not parts:
        return context.get_user_input() or "", []

    texts: list[str] = []
    attachments: list[Attachment] = []
    budget = _Budget(text=_MAX_TEXT_TOTAL, binary=_MAX_BINARY_TOTAL)
    for part in parts:
        which = part.WhichOneof("content")
        if which == "text":
            if part.text:
                texts.append(part.text)
            continue
        if which == "url":
            attachments.append(
                Attachment(
                    name=part.filename or part.url,
                    media_type=part.media_type or "",
                    uri=part.url,
                )
            )
            continue
        if which == "raw":
            attachment = _from_raw(part)
        elif which == "data":
            attachment = _from_data(part)
        else:
            continue
        budget.fit(attachment)
        attachments.append(attachment)

    return "\n".join(texts).strip(), attachments


def _from_raw(part: Part) -> Attachment:
    """An inline file part: decoded when it is text, kept as bytes otherwise."""
    media_type = part.media_type or "application/octet-stream"
    text: str | None = None
    if _is_textual(media_type):
        try:
            text = part.raw.decode("utf-8")
        except UnicodeDecodeError:
            # Declared as text but is not: treat it as the bytes it actually is
            # rather than handing the agent replacement characters.
            text = None
    return Attachment(
        name=part.filename or "unnamed",
        media_type=media_type,
        text=text,
        data=None if text is not None else bytes(part.raw),
    )


def _from_data(part: Part) -> Attachment:
    """A structured-data part, rendered as the JSON the caller meant it to be."""
    return Attachment(
        name=part.filename or "data",
        media_type=part.media_type or "application/json",
        text=json_format.MessageToJson(part.data),
    )


@dataclass
class _Budget:
    """What is left of a message's attachment allowance."""

    text: int
    binary: int

    def fit(self, attachment: Attachment) -> None:
        """Trim an attachment to what remains, recording that it was trimmed."""
        if attachment.text is not None:
            cap = min(self.text, _MAX_TEXT_ATTACHED)
            if len(attachment.text) > cap:
                attachment.text = attachment.text[:cap]
                attachment.truncated = True
            self.text -= len(attachment.text)
        elif attachment.data is not None:
            cap = min(self.binary, _MAX_BINARY_ATTACHED)
            if len(attachment.data) > cap:
                # Bytes cannot be usefully cut in half, so an oversized one is
                # dropped whole and rendered as a note that it was.
                attachment.data = None
                attachment.truncated = True
            else:
                self.binary -= len(attachment.data)


def _describe_tool(event: ToolUse) -> str:
    """A short, human-readable line for a tool invocation."""
    i = event.tool_input
    if event.name == "Bash":
        return f"$ {str(i.get('command', '')).strip()[:120]}"
    path = i.get("file_path") or i.get("path") or i.get("pattern")
    # ACP names a tool call with a human title that often already says the path
    # ("Write calc.py"); appending it again reads as a stutter.
    if path and str(path) not in event.name:
        return f"{event.name} {path}"
    return event.name


def _describe_result(event: ToolResult, name: str) -> str:
    """A short line for a tool's outcome.

    Successful output is left out as noise; a failure carries its first line,
    which is the part that explains it.
    """
    if not event.failed:
        return f"✓ {name}"
    reason = event.output.strip().splitlines()[0][:200] if event.output.strip() else ""
    return f"✗ {name}: {reason}" if reason else f"✗ {name}"


def _render_plan(plan: Plan) -> str:
    """Render a plan as a markdown checklist.

    ``in_progress`` gets its own marker: which step the agent is on is the
    reason to watch a plan at all.
    """
    if plan.markdown:
        return plan.markdown
    if plan.uri:
        return f"The agent keeps its plan in {plan.uri}\n"
    lines = []
    for step in plan.steps:
        mark = _PLAN_MARKS.get(step.status, " ")
        prefix = "(high) " if step.priority == "high" else ""
        lines.append(f"- [{mark}] {prefix}{step.content}")
    return "\n".join(lines) + "\n" if lines else ""


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
            prompt, attachments = _build_input(context)
            request = RunRequest(
                prompt=prompt,
                context_id=context_id,
                resume=self._session_ids.get(context_id),
                attachments=attachments,
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
                    stream.tool_names[event.tool_use_id] = event.name
                    await updater.update_status(
                        TaskState.TASK_STATE_WORKING,
                        message=updater.new_agent_message(
                            [Part(text=_describe_tool(event))]
                        ),
                    )
                elif isinstance(event, ToolResult):
                    name = (
                        event.name or stream.tool_names.get(event.tool_use_id) or "tool"
                    )
                    await updater.update_status(
                        TaskState.TASK_STATE_WORKING,
                        message=updater.new_agent_message(
                            [Part(text=_describe_result(event, name))]
                        ),
                    )
                elif isinstance(event, FileChange):
                    await updater.add_artifact(
                        [Part(text=event.diff, media_type="text/x-diff")],
                        name=event.path,
                    )
                elif isinstance(event, Notice):
                    await updater.update_status(
                        TaskState.TASK_STATE_WORKING,
                        message=updater.new_agent_message([Part(text=event.text)]),
                    )
                elif isinstance(event, Thought):
                    if not stream.thinking_artifact_id:
                        stream.thinking_artifact_id = uuid4().hex
                    await updater.add_artifact(
                        [Part(text=event.text)],
                        artifact_id=stream.thinking_artifact_id,
                        name="thinking",
                        append=stream.sent_thought,
                        last_chunk=False,
                    )
                    stream.sent_thought = True
                elif isinstance(event, Plan):
                    body = _render_plan(event)
                    # An emptied plan still replaces the artifact, or the caller
                    # would keep seeing a checklist the agent has abandoned.
                    if body or stream.plan_artifact_id:
                        if not stream.plan_artifact_id:
                            stream.plan_artifact_id = uuid4().hex
                        await updater.add_artifact(
                            [Part(text=body, media_type="text/markdown")],
                            artifact_id=stream.plan_artifact_id,
                            name="plan",
                            append=False,
                            last_chunk=True,
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
            # A deliberate tasks/cancel, or server shutdown: drop the session and
            # its runner instead of leaking them.
            self._live.pop(task_id, None)
            self._streams.pop(task_id, None)
            session.abort()
            # The terminal state has to go out from here. We are cancelled before
            # AgentExecutor.cancel is called, and unwinding closes the event
            # queue, so the status it enqueues is dropped and the task stays
            # `working`. Awaited rather than shielded: the enqueue does not
            # suspend while the queue has room, so it lands ahead of that close.
            await self._report_cancel(updater, task_id)
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

        if stream.sent_thought:
            # Close it too, or a consumer waiting for a final chunk holds the
            # thinking artifact open past the end of the task.
            await updater.add_artifact(
                [Part(text="")],
                artifact_id=stream.thinking_artifact_id,
                name="thinking",
                append=True,
                last_chunk=True,
            )

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
        updater = TaskUpdater(event_queue, task_id, context_id)
        # A task paused on a permission has no _pump left to write its terminal
        # state, and closing the session below awaits, which is long enough for
        # the cancelled producer to close the queue. So emit here, before the
        # first await - but only when no _pump will, or the caller sees two.
        if session is None or session.is_parked:
            await self._report_cancel(updater, task_id)
        if session is not None:
            await session.close()

    @staticmethod
    async def _report_cancel(updater: TaskUpdater, task_id: str) -> None:
        """Write the terminal state, logged rather than raised if it cannot.

        Both callers are already unwinding, and a queue closed under us drops the
        event without raising, so a failure here is this very bug coming back.
        """
        try:
            await updater.cancel()
        except Exception:
            logger.debug("could not cancel task %s", task_id, exc_info=True)

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
        if event.stop_reason is not None:
            meta["stop_reason"] = event.stop_reason
        return meta
