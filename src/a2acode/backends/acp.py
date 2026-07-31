"""ACP backend.

Drives any agent that speaks Zed's Agent Client Protocol (ACP) — Claude Code,
Gemini CLI, Codex, OpenHands, ... — as a subprocess, and normalizes its
``session/update`` stream into backend events. This is the seam that makes the
server vendor-neutral: one ACP client backend instead of one SDK adapter per
agent. Swapping the underlying coding agent becomes a launch-command change, not
a new backend.

ACP maps almost one-to-one onto the backend event vocabulary:

    agent_message_chunk          -> TextDelta
    tool_call / tool_call_update -> ToolUse (+ FileChange for diff content)
    a terminal tool_call status  -> ToolResult (completed / failed, with output)
    plan / plan_content_update   -> Plan (the agent's task list, by replacement)
    session/request_permission   -> PermissionRequest (the input-required pause)
    PromptResponse usage + cost  -> Result

The permission round trip lands exactly on the session seam: the agent calls
back into the client's ``request_permission``, which awaits
``session.request_permission`` and parks until the A2A caller answers — the same
parked-across-two-execute-calls behavior the Claude backend gets through
``can_use_tool``. Cancellation runs the same seam in reverse: the session's
canceller sends ``session/cancel`` so the agent ends the turn itself.

``events_from_update`` and ``select_option`` are pure and side-effect free so the
protocol translation is unit-testable without launching an agent subprocess.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, Client, spawn_agent_process, text_block
from acp import schema as s

from .base import (
    BackendEvent,
    FileChange,
    Notice,
    Plan,
    PlanStep,
    Result,
    RunRequest,
    TextDelta,
    ToolResult,
    ToolUse,
)
from .diff import unified_diff
from .session import BackendSession

# Tool output is relayed as a short excerpt: it arrives on every status update
# and can be arbitrarily large (a whole test run, a file dump).
_MAX_TOOL_OUTPUT = 2000

# How to launch each known ACP agent adapter as a subprocess. A preset is just a
# default command; pass an explicit ``command``/``args`` to drive any other ACP
# agent (or a pinned/locally installed adapter).
_AGENTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "claude": ("npx", ("-y", "@zed-industries/claude-agent-acp")),
    "gemini": ("gemini", ("--experimental-acp",)),
    "codex": ("codex-acp", ()),
}


def events_from_update(update: object) -> Iterator[BackendEvent]:
    """Map one ACP ``session/update`` to normalized backend events.

    Pure and side-effect free so the translation can be unit tested without a
    live agent subprocess. ``usage_update`` yields nothing here; cost/usage is
    folded into the terminal ``Result`` by the backend.
    """
    if isinstance(update, s.AgentMessageChunk):
        text = getattr(update.content, "text", None)
        if text:
            yield TextDelta(text=text)
    elif isinstance(update, s.ToolCallStart):
        yield ToolUse(
            name=update.title or (update.kind or "tool"),
            tool_input=_as_dict(update.raw_input),
            tool_use_id=update.tool_call_id,
        )
        yield from _file_changes(update.content)
        yield from _tool_results(update)
    elif isinstance(update, s.ToolCallProgress):
        # A diff is often not ready when the tool call opens; later progress
        # updates carry it. The ToolUse was already emitted on the start event.
        yield from _file_changes(update.content)
        yield from _tool_results(update)
    elif isinstance(update, s.AgentPlanUpdate):
        yield _plan(update.entries)
    elif isinstance(update, s.AgentPlanContentUpdate):
        # The newer plan surface is a union; only the ``items`` variant carries
        # per-entry status. The markdown and file variants are prose, so they
        # are left alone rather than flattened into steps with invented states.
        if isinstance(update.plan, s.PlanUpdateItems):
            yield _plan(update.plan.entries)


def select_option(options: Sequence[s.PermissionOption], *, allow: bool) -> str | None:
    """Pick the option id that matches the caller's allow/deny decision.

    ACP returns the binding choice as an ``optionId``; ``kind`` is only a UI
    hint. Prefer a one-shot option (allow_once / reject_once) over a sticky one,
    then fall back to any option of the right polarity. ``None`` means the agent
    offered no option of that polarity.
    """
    preferred = (
        ("allow_once", "allow_always") if allow else ("reject_once", "reject_always")
    )
    for kind in preferred:
        for opt in options:
            if opt.kind == kind:
                return opt.option_id
    prefix = "allow" if allow else "reject"
    for opt in options:
        if (opt.kind or "").startswith(prefix):
            return opt.option_id
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _plan(entries: Sequence[s.PlanEntry]) -> Plan:
    return Plan(
        steps=[
            PlanStep(
                content=entry.content,
                status=entry.status,
                priority=entry.priority or "",
            )
            for entry in entries
        ]
    )


def _tool_results(update: s.ToolCallStart | s.ToolCallProgress) -> Iterator[ToolResult]:
    """Emit a tool's outcome once its status is terminal.

    ACP reports a tool call as a series of updates; ``pending`` and
    ``in_progress`` say nothing a caller can act on, so only ``completed`` and
    ``failed`` become an event. An agent that never reports a status simply
    produces no ToolResult, which is why the executor treats it as optional.
    """
    if update.status not in ("completed", "failed"):
        return
    yield ToolResult(
        tool_use_id=update.tool_call_id,
        name=update.title or "",
        failed=update.status == "failed",
        output=_tool_output(update.content),
    )


def _tool_output(content: Sequence[object] | None) -> str:
    """Collect the text an agent attached to a tool call, capped.

    Tool output is unbounded (a full test run, a file dump), and it travels on
    every status update, so it is truncated rather than relayed whole.
    """
    texts = [
        item.content.text
        for item in content or []
        if isinstance(item, s.ContentToolCallContent)
        and isinstance(item.content, s.TextContentBlock)
        and item.content.text
    ]
    out = "\n".join(texts)
    return out if len(out) <= _MAX_TOOL_OUTPUT else out[:_MAX_TOOL_OUTPUT] + " …"


def _file_changes(content: Sequence[object] | None) -> Iterator[FileChange]:
    for item in content or []:
        if isinstance(item, s.FileEditToolCallContent):
            yield FileChange(
                path=item.path,
                diff=unified_diff(item.path, item.old_text or "", item.new_text or ""),
            )


class _BridgeClient(Client):
    """ACP client that forwards agent output onto a BackendSession.

    The agent's notifications and permission requests arrive on the ACP
    connection's reader task; this translates each onto the session queue, and
    parks a permission request on ``session.request_permission`` until the A2A
    caller answers.
    """

    def __init__(self, session: BackendSession, cwd: str = ".") -> None:
        self._session = session
        # Resolved workspace root: every fs read/write is confined under it so a
        # buggy or hostile agent can't reach arbitrary files via the capability
        # we advertise. ACP paths are absolute, but we still contain them.
        self._cwd = Path(cwd).resolve()
        self.cost_usd: float | None = None

    def _safe_path(self, path: str) -> Path:
        target = Path(path)
        if not target.is_absolute():
            target = self._cwd / target
        target = target.resolve()
        if not target.is_relative_to(self._cwd):
            raise PermissionError(f"path escapes workspace {self._cwd}: {path!r}")
        return target

    async def session_update(self, session_id: str, update: Any, **_: Any) -> None:
        if isinstance(update, s.UsageUpdate) and update.cost is not None:
            self.cost_usd = update.cost.amount
        for event in events_from_update(update):
            await self._session.emit(event)

    async def request_permission(
        self,
        session_id: str,
        tool_call: s.ToolCallUpdate,
        options: list[s.PermissionOption],
        **_: Any,
    ) -> s.RequestPermissionResponse:
        name = tool_call.title or (tool_call.kind or "tool")
        decision = await self._session.request_permission(
            name, _as_dict(tool_call.raw_input), name
        )
        option_id = select_option(options, allow=decision.allow)
        if option_id is None:
            # The agent offered no option of the requested polarity; cancelling
            # is the only safe answer (selecting the wrong one could run a tool
            # the caller denied).
            return s.RequestPermissionResponse(
                outcome=s.DeniedOutcome(outcome="cancelled")
            )
        return s.RequestPermissionResponse(
            outcome=s.AllowedOutcome(outcome="selected", option_id=option_id)
        )

    async def read_text_file(
        self,
        session_id: str,
        path: str,
        line: int | None = None,
        limit: int | None = None,
        **_: Any,
    ) -> s.ReadTextFileResponse:
        # We advertise fs.readTextFile, so serve reads from disk. There are no
        # unsaved editor buffers on a server; the file on disk is the truth.
        target = self._safe_path(path)
        if limit is not None and limit <= 0:
            return s.ReadTextFileResponse(content="")
        # A non-positive line number reads from the top.
        start = (line - 1) if (line and line > 0) else 0

        def _read() -> str:
            if line is None and limit is None:
                return target.read_text(encoding="utf-8")
            # Stream so a small windowed read doesn't pull a huge file into
            # memory just to slice a few lines out of it.
            end = (start + limit) if limit is not None else None
            out: list[str] = []
            with target.open(encoding="utf-8") as f:
                for i, text_line in enumerate(f):
                    if i >= start:
                        out.append(text_line)
                    if end is not None and i >= end - 1:
                        break
            return "".join(out)

        # Offloaded to a thread so the synchronous read can't stall the event
        # loop the ACP connection runs on.
        text = await asyncio.to_thread(_read)
        return s.ReadTextFileResponse(content=text)

    async def write_text_file(
        self, session_id: str, path: str, content: str, **_: Any
    ) -> s.WriteTextFileResponse | None:
        target = self._safe_path(path)

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        # Offloaded so the blocking mkdir/write can't stall the event loop.
        await asyncio.to_thread(_write)
        return None


class ACPBackend:
    name = "acp"

    def __init__(
        self,
        *,
        agent: str = "claude",
        command: str | None = None,
        args: Sequence[str] | None = None,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if command is None:
            preset = _AGENTS.get(agent)
            if preset is None:
                known = ", ".join(sorted(_AGENTS))
                raise ValueError(
                    f"unknown ACP agent {agent!r} (known: {known}); "
                    "pass command=... to launch any other ACP agent"
                )
            command, default_args = preset
            args = default_args if args is None else args
        self.agent = agent
        self.command = command
        self.args = list(args or [])
        self.cwd = os.path.abspath(cwd or os.getcwd())
        # Overrides layered onto the server's own environment so the adapter
        # still inherits PATH and any provider credentials (ANTHROPIC_API_KEY,
        # GEMINI_API_KEY, ...) it needs to authenticate.
        self.env = {**os.environ, **(env or {})}

    async def drive(self, session: BackendSession, request: RunRequest) -> None:
        # The ACP Client base declares terminal/* and ext_* with empty bodies as
        # optional overrides; we advertise no terminal capability, so the agent
        # never calls them. mypy reads the empty bodies as abstract, hence the
        # scoped ignore.
        client = _BridgeClient(session, self.cwd)  # type: ignore[abstract]
        async with spawn_agent_process(
            client, self.command, *self.args, env=self.env, cwd=self.cwd
        ) as (conn, _process):
            init = await conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=s.ClientCapabilities(
                    fs=s.FileSystemCapabilities(
                        read_text_file=True, write_text_file=True
                    )
                ),
            )
            session_id = await self._open_session(conn, init, request, session)
            # An A2A cancel now reaches the agent as session/cancel, which ends
            # the turn cleanly, instead of only killing its process from under
            # whatever tool was mid-flight.
            session.set_canceller(lambda: conn.cancel(session_id=session_id))
            response = await conn.prompt(
                prompt=[text_block(request.prompt)], session_id=session_id
            )
            usage = response.usage.model_dump() if response.usage else None
            await session.emit(
                Result(
                    session_id=session_id,
                    cost_usd=client.cost_usd,
                    num_turns=None,
                    usage=usage,
                )
            )

    async def _open_session(
        self,
        conn: Any,
        init: s.InitializeResponse,
        request: RunRequest,
        session: BackendSession,
    ) -> str:
        if request.resume:
            if getattr(init.agent_capabilities, "load_session", False):
                await conn.load_session(
                    cwd=self.cwd, session_id=request.resume, mcp_servers=[]
                )
                return request.resume
            # Continuity was asked for and cannot be delivered. Say so: a caller
            # that sent a follow-up turn would otherwise get a confident answer
            # from an agent that has never seen the conversation it refers to.
            await session.emit(
                Notice(
                    f"the {self.agent} agent does not support session/load, so "
                    "this turn starts a fresh session; earlier turns in this "
                    "context are not in its view"
                )
            )
        # No resume, or the agent can't reload one: start fresh. The executor
        # learns the new session id from the Result and maps the A2A context
        # onto it for the next turn.
        response = await conn.new_session(cwd=self.cwd, mcp_servers=[])
        return response.session_id
