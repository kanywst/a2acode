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

import logging
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

logger = logging.getLogger(__name__)

_ALLOW_WORDS = {"allow", "yes", "y", "approve", "ok", "accept", "grant"}


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

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
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
            session = BackendSession()
            session.start(lambda s: self._backend.drive(s, request))
            self._live[task_id] = session
        else:
            # Follow-up to an input-required pause: the message is the decision.
            # Guard against a concurrent message arriving while the task is still
            # running — resolving and pumping a non-parked session would put two
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
        artifact_id = uuid4().hex
        pending_text: str | None = None
        text_chunks: list[str] = []
        metadata: dict[str, object] = {}

        async def flush(text: str, *, last: bool) -> None:
            await updater.add_artifact(
                [Part(text=text)],
                artifact_id=artifact_id,
                name="response",
                append=len(text_chunks) > 1,
                last_chunk=last,
            )

        try:
            async for event in session.drain():
                if isinstance(event, TextDelta):
                    if pending_text is not None:
                        text_chunks.append(pending_text)
                        await flush(pending_text, last=False)
                    pending_text = event.text
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
                    if pending_text is not None:
                        text_chunks.append(pending_text)
                        await flush(pending_text, last=True)
                        pending_text = None
                    await self._request_input(updater, event)
                elif isinstance(event, Result):
                    metadata = self._result_metadata(event)
                    if event.session_id:
                        self._session_ids[context_id] = event.session_id
        except Exception as exc:  # noqa: BLE001 — surface any failure to the client
            logger.exception("backend run failed for task %s", task_id)
            await self._discard(task_id, session)
            await updater.failed(
                message=updater.new_agent_message(
                    [Part(text=f"Claude Code run failed: {exc}")]
                )
            )
            return

        if pending_text is not None:
            text_chunks.append(pending_text)
            await flush(pending_text, last=True)

        if not session.done:
            # Paused on a permission request; stay registered for the follow-up.
            return

        await self._discard(task_id, session)
        full_text = "".join(text_chunks) or "(no text output)"
        await updater.complete(
            message=updater.new_agent_message(
                [Part(text=full_text)], metadata=metadata or None
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id = context.task_id, context.context_id
        assert task_id is not None and context_id is not None
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
                    "a2claude_permission": {
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

    async def _discard(self, task_id: str, session: BackendSession) -> None:
        self._live.pop(task_id, None)
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
