"""ACP backend.

Drives any agent that speaks Zed's Agent Client Protocol (ACP) — Claude Code,
Gemini CLI, Codex, OpenHands, ... — as a subprocess, and normalizes its
``session/update`` stream into backend events. This is the seam that makes the
server vendor-neutral: one ACP client backend instead of one SDK adapter per
agent. Swapping the underlying coding agent becomes a launch-command change, not
a new backend.

ACP maps almost one-to-one onto the backend event vocabulary:

    agent_message_chunk          -> TextDelta
    agent_thought_chunk          -> Thought (kept out of the answer text)
    tool_call / tool_call_update -> ToolUse (+ FileChange for diff content)
    a terminal tool_call status  -> ToolResult (completed / failed, with output)
    plan / plan_update           -> Plan (the agent's task list, by replacement)
    current_mode / session_info  -> Notice
    session/request_permission   -> PermissionRequest (the input-required pause)
    PromptResponse + cost        -> Result (usage, cost, stop reason)

The permission round trip lands exactly on the session seam: the agent calls
back into the client's ``request_permission``, which awaits
``session.request_permission`` and parks until the A2A caller answers — the same
parked-across-two-execute-calls behavior the Claude backend gets through
``can_use_tool``. Cancellation runs the same seam in reverse: the session's
canceller sends ``session/cancel`` so the agent ends the turn itself.

An agent subprocess is kept alive per A2A context rather than per turn. Spawning
one costs a process launch (``npx ...``) plus an ACP handshake plus a
``session/load``, all of which a follow-up turn in the same conversation can
skip entirely by talking to the process that already holds it.

``events_from_update``, ``select_option``, and ``prompt_blocks`` are pure and
side-effect free so the protocol translation is unit-testable without launching
an agent subprocess.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import shlex
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    Client,
    image_block,
    spawn_agent_process,
    text_block,
)
from acp import schema as s
from acp.exceptions import RequestError
from acp.task import InMemoryMessageQueue

from .attach import append_to_prompt
from .base import (
    Attachment,
    BackendEvent,
    FileChange,
    Notice,
    PermissionDecision,
    Plan,
    PlanStep,
    Result,
    RunRequest,
    TextDelta,
    Thought,
    ToolResult,
    ToolUse,
)
from .diff import unified_diff
from .dispatch import OrderedDispatcher
from .session import BackendSession
from .terminal import DEFAULT_OUTPUT_LIMIT, MAX_OUTPUT_LIMIT, Terminal, spawn

logger = logging.getLogger(__name__)

# Tool output is relayed as a short excerpt: it arrives on every status update
# and can be arbitrarily large (a whole test run, a file dump).
_MAX_TOOL_OUTPUT = 2000

# How many agent subprocesses to keep alive across turns. Each is a real
# process holding a real conversation, so this bounds memory and file handles
# the way the executor's own maps bound sessions.
_MAX_AGENTS = 32

# Terminals an agent may hold open at once within a turn.
_MAX_TERMINALS = 16

# Tool calls one turn is remembered across, for folding their later updates in.
_MAX_TOOL_CALLS = 4096

# How long a finished turn waits for its trailing notifications to be handled.
_DRAIN_TIMEOUT = 10.0

# A shell-safe environment variable name, which an agent's need not be.
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Tool-call statuses that say something a caller can act on.
_TERMINAL = ("completed", "failed")

# How to launch each known ACP agent adapter as a subprocess. A preset is just a
# default command; pass an explicit ``command``/``args`` to drive any other ACP
# agent (or a pinned/locally installed adapter). All three go through npx so a
# preset needs nothing installed beyond Node and the agent's own credential.
_AGENTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "claude": ("npx", ("-y", "@zed-industries/claude-agent-acp")),
    "gemini": ("npx", ("-y", "@google/gemini-cli", "--acp")),
    "codex": ("npx", ("-y", "@zed-industries/codex-acp")),
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
    elif isinstance(update, s.AgentThoughtChunk):
        text = getattr(update.content, "text", None)
        if text:
            yield Thought(text=text)
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
        yield Plan(steps=_steps(update.entries))
    elif isinstance(update, s.AgentPlanContentUpdate):
        yield _plan_content(update.plan)
    elif isinstance(update, s.CurrentModeUpdate):
        yield Notice(text=f"the agent switched to {update.current_mode_id} mode")
    elif isinstance(update, s.SessionInfoUpdate):
        if update.title:
            yield Notice(text=f"the agent titled this session {update.title!r}")


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


def _command_line(command: str, argv: Sequence[str], env: Mapping[str, str]) -> str:
    """Render a command the way the caller must judge it: environment included.

    The agent supplies its own variables, and PATH or LD_PRELOAD decide what a
    plausible-looking command actually executes. Showing the words without them
    would put an innocuous line in front of the caller and run something else.

    The name is agent-controlled too, so one carrying spaces or quotes is shown
    as a single quoted token rather than allowed to split into words the caller
    would read as separate arguments.
    """
    assignments = " ".join(
        f"{name}={shlex.quote(value)}"
        if _ENV_NAME.fullmatch(name)
        else shlex.quote(f"{name}={value}")
        for name, value in sorted(env.items())
    )
    line = shlex.join([command, *argv])
    return f"{assignments} {line}" if assignments else line


def prompt_blocks(
    request: RunRequest, capabilities: s.AgentCapabilities | None
) -> list[Any]:
    """Build the content blocks for one ``session/prompt``.

    An image goes as a real image block when the agent advertises it reads one;
    anything else folds into the text, which every ACP agent understands.
    """
    prompt_caps = getattr(capabilities, "prompt_capabilities", None)
    takes_images = bool(getattr(prompt_caps, "image", False))

    images: list[Any] = []
    inlined: list[Attachment] = []
    for attachment in request.attachments:
        if (
            takes_images
            and attachment.data
            and attachment.media_type.startswith("image/")
        ):
            images.append(
                image_block(
                    base64.b64encode(attachment.data).decode("ascii"),
                    attachment.media_type,
                )
            )
        else:
            inlined.append(attachment)
    return [text_block(append_to_prompt(request.prompt, inlined)), *images]


def _plan_content(plan: object) -> Plan:
    """Map the plan-content union onto one Plan.

    Only the ``items`` variant carries per-entry status; the other two are the
    agent's own prose or a file it keeps the plan in, carried as-is rather than
    flattened into steps with invented states.
    """
    if isinstance(plan, s.PlanUpdateItems):
        return Plan(steps=_steps(plan.entries))
    if isinstance(plan, s.PlanUpdateMarkdown):
        return Plan(markdown=plan.content)
    if isinstance(plan, s.PlanUpdateFile):
        return Plan(uri=plan.uri)
    return Plan()


def _steps(entries: Sequence[s.PlanEntry]) -> list[PlanStep]:
    return [
        PlanStep(
            content=entry.content,
            status=entry.status,
            priority=entry.priority or "",
        )
        for entry in entries
    ]


def _tool_results(update: s.ToolCallStart | s.ToolCallProgress) -> Iterator[ToolResult]:
    """Emit a tool's outcome once its status is terminal.

    ACP reports a tool call as a series of updates; only ``completed`` and
    ``failed`` say anything a caller can act on.
    """
    if update.status not in _TERMINAL:
        return
    yield ToolResult(
        tool_use_id=update.tool_call_id,
        name=update.title or "",
        failed=update.status == "failed",
        output=_tool_output(update.content),
    )


def _tool_output(content: Sequence[object] | None) -> str:
    """Collect the text an agent attached to a tool call, capped."""
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


@dataclass
class _ToolCall:
    """What a tool call has said about itself so far."""

    title: str = ""
    kind: Any = None
    raw_input: Any = None
    announced: bool = False


class _ToolCalls:
    """The turn's tool calls, each folded together from its own updates.

    ACP announces a tool call before it has parsed the arguments and fills them
    in on a later update, absent meaning unchanged. So a ToolUse is held back
    until the call can say what it acts on, and every later update is completed
    from what the call already said before the pure mapper sees it.
    """

    def __init__(self) -> None:
        self._calls: dict[str, _ToolCall] = {}

    def feed(self, update: object) -> Iterator[object]:
        """Rewrite one update into the updates worth mapping."""
        if isinstance(update, s.ToolCallStart | s.ToolCallProgress):
            yield from self._fold(update)
        else:
            yield update

    def flush(self) -> Iterator[s.ToolCallStart]:
        """Announce, at end of turn, calls the agent never said more about."""
        for call_id, call in self._calls.items():
            if not call.announced:
                call.announced = True
                yield s.ToolCallStart(
                    session_update="tool_call",
                    tool_call_id=call_id,
                    title=call.title,
                    kind=call.kind,
                )

    def _fold(
        self, update: s.ToolCallStart | s.ToolCallProgress
    ) -> Iterator[s.ToolCallStart | s.ToolCallProgress]:
        call = self._calls.get(update.tool_call_id)
        if call is None:
            if len(self._calls) >= _MAX_TOOL_CALLS:
                # Ids are the agent's to invent, and every other per-turn
                # collection here is bounded. Past the bound, map the update as
                # it came rather than remember it: less detail, nothing dropped.
                logger.warning("tool calls in one turn exceed %d", _MAX_TOOL_CALLS)
                yield update
                return
            call = self._calls.setdefault(update.tool_call_id, _ToolCall())
        if update.title:
            call.title = update.title
        if update.kind:
            call.kind = update.kind
        if call.announced:
            yield update.model_copy(
                update={"title": call.title or None, "kind": call.kind}
            )
            return
        if isinstance(update.raw_input, Mapping) and update.raw_input:
            call.raw_input = update.raw_input
        # Content and a terminal status are worth telling the caller about even
        # with no arguments to name; anything else waits for the next update.
        if not (call.raw_input or update.content or update.status in _TERMINAL):
            return
        call.announced = True
        start = s.ToolCallStart(
            session_update="tool_call",
            tool_call_id=update.tool_call_id,
            title=call.title,
            kind=call.kind,
            status=update.status,
            content=update.content,
            locations=update.locations,
            raw_input=call.raw_input,
            raw_output=update.raw_output,
        )
        # Released once the ToolUse carries it: a Write's arguments are the whole
        # file, and a turn holds every call it made.
        call.raw_input = None
        yield start


class _BridgeClient(Client):
    """ACP client that forwards agent output onto a BackendSession.

    The agent's notifications and permission requests arrive on the ACP
    connection's reader task; this translates each onto the session queue, and
    parks a permission request on ``session.request_permission`` until the A2A
    caller answers.

    A connection outlives any single turn, so the session it forwards to is
    rebound per turn; between turns nothing is bound and stray output is
    dropped, there being no caller to route it to.
    """

    def __init__(self, session: BackendSession | None = None, cwd: str = ".") -> None:
        self._session = session
        # Resolved workspace root: every fs read/write is confined under it so a
        # buggy or hostile agent can't reach arbitrary files via the capability
        # we advertise. ACP paths are absolute, but we still contain them.
        self._cwd = Path(cwd).resolve()
        self.cost_usd: float | None = None
        self._terminals: dict[str, Terminal] = {}
        self._tool_calls = _ToolCalls()

    def bind(self, session: BackendSession) -> None:
        """Route this connection's output to ``session`` for one turn."""
        self._session = session
        self.cost_usd = None
        # Tool call ids belong to the turn that opened them.
        self._tool_calls = _ToolCalls()

    async def unbind(self) -> None:
        self._session = None
        # Terminals belong to the turn that opened them. An agent is meant to
        # release its own, but a crashed or cancelled one would otherwise leave
        # processes running on the server with nobody to reap them.
        terminals, self._terminals = list(self._terminals.values()), {}
        # Concurrently: each close waits on a process that may not die promptly,
        # and this runs inside the turn's lock, holding up the next one.
        await asyncio.gather(*(t.close() for t in terminals), return_exceptions=True)

    async def _approve(
        self, name: str, tool_input: dict[str, Any], description: str
    ) -> PermissionDecision:
        """Put an action to the A2A caller, denying if nobody can answer."""
        if self._session is None:
            # Between turns: approving would act with nobody watching.
            return PermissionDecision(
                request_id="", allow=False, message="no caller attached"
            )
        return await self._session.request_permission(name, tool_input, description)

    def _safe_path(self, path: str) -> Path:
        target = Path(path)
        if not target.is_absolute():
            target = self._cwd / target
        target = target.resolve()
        if not target.is_relative_to(self._cwd):
            raise PermissionError(f"path escapes workspace {self._cwd}: {path!r}")
        return target

    async def session_update(self, session_id: str, update: Any, **_: Any) -> None:
        if self._session is None:
            return
        if isinstance(update, s.UsageUpdate) and update.cost is not None:
            self.cost_usd = update.cost.amount
        for folded in self._tool_calls.feed(update):
            for event in events_from_update(folded):
                await self._session.emit(event)

    async def flush_tool_calls(self) -> None:
        """Emit whatever a tool call did say, for one that never went further."""
        if self._session is None:
            return
        for update in self._tool_calls.flush():
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
        decision = await self._approve(name, _as_dict(tool_call.raw_input), name)
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

    async def create_terminal(
        self,
        session_id: str,
        command: str,
        args: list[str] | None = None,
        env: list[s.EnvVariable] | None = None,
        cwd: str | None = None,
        output_byte_limit: int | None = None,
        **_: Any,
    ) -> s.CreateTerminalResponse:
        argv = list(args or [])
        environment = {var.name: var.value for var in env or []}
        # Running a command is exactly the kind of act the caller holds the
        # decision on, so it takes the same round trip as any tool. Without this
        # gate, advertising the capability would hand the agent a way around the
        # permission model rather than a safer place to execute.
        decision = await self._approve(
            "Terminal",
            {"command": command, "args": argv, "cwd": cwd, "env": environment},
            _command_line(command, argv, environment),
        )
        if not decision.allow:
            # A protocol error, not a crash: a refusal is a normal outcome, and
            # raising anything else logs a traceback for every denied command.
            raise RequestError.auth_required(
                {"reason": decision.message or "terminal denied by the A2A caller"}
            )

        if len(self._terminals) >= _MAX_TERMINALS:
            raise RuntimeError(f"too many open terminals (limit {_MAX_TERMINALS})")

        limit = min(output_byte_limit or DEFAULT_OUTPUT_LIMIT, MAX_OUTPUT_LIMIT)
        terminal = await spawn(
            command,
            argv,
            cwd=self._safe_path(cwd) if cwd else self._cwd,
            env=environment,
            limit=max(limit, 1),
        )
        terminal_id = uuid4().hex
        self._terminals[terminal_id] = terminal
        return s.CreateTerminalResponse(terminal_id=terminal_id)

    async def terminal_output(
        self, session_id: str, terminal_id: str, **_: Any
    ) -> s.TerminalOutputResponse:
        terminal = self._terminal(terminal_id)
        status = terminal.exit_status()
        return s.TerminalOutputResponse(
            output=terminal.output,
            truncated=terminal.truncated,
            exit_status=None
            if status is None
            else s.TerminalExitStatus(exit_code=status[0], signal=status[1]),
        )

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **_: Any
    ) -> s.WaitForTerminalExitResponse:
        exit_code, signal = await self._terminal(terminal_id).wait()
        return s.WaitForTerminalExitResponse(exit_code=exit_code, signal=signal)

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **_: Any
    ) -> s.KillTerminalResponse | None:
        await self._terminal(terminal_id).kill()
        return s.KillTerminalResponse()

    async def release_terminal(
        self, session_id: str, terminal_id: str, **_: Any
    ) -> s.ReleaseTerminalResponse | None:
        terminal = self._terminals.pop(terminal_id, None)
        if terminal is not None:
            await terminal.close()
        return s.ReleaseTerminalResponse()

    def _terminal(self, terminal_id: str) -> Terminal:
        terminal = self._terminals.get(terminal_id)
        if terminal is None:
            raise ValueError(f"unknown terminal {terminal_id!r}")
        return terminal

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


@dataclass
class _Agent:
    """One live agent subprocess and the ACP connection to it."""

    stack: AsyncExitStack
    conn: Any
    process: Any
    client: _BridgeClient
    queue: Any
    capabilities: s.AgentCapabilities | None
    session_id: str | None = None
    pooled: bool = True
    broken: bool = False
    # Turns holding this agent or about to. Taken under the pool lock, so an
    # agent is never idle to eviction in the gap before it takes ``lock``.
    claims: int = 0
    # Held for a whole turn, including while parked on a permission: one ACP
    # connection carries one conversation.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def usable(self) -> bool:
        return not self.broken and self.process.returncode is None


async def _close_agent(agent: _Agent) -> None:
    # Shutdown races a subprocess that may already be gone; the pool must not
    # fail a turn (or a server shutdown) over how an agent chose to exit.
    with suppress(Exception):
        await agent.stack.aclose()


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
        self._agents: dict[str, _Agent] = {}
        # Serializes lookup, eviction, and spawning, so two concurrent first
        # turns in one context cannot each launch their own process.
        self._pool_lock = asyncio.Lock()

    async def drive(self, session: BackendSession, request: RunRequest) -> None:
        agent = await self._acquire(request.context_id)
        try:
            if agent.lock.locked():
                # The turn ahead may be parked on a permission this caller has
                # not answered, so the wait is worth naming rather than stalling.
                await session.emit(
                    Notice(
                        "waiting for the turn already in flight on this "
                        "conversation to finish before starting this one"
                    )
                )
            async with agent.lock:
                agent.client.bind(session)
                try:
                    await self._run_turn(agent, session, request)
                except BaseException:
                    # Cancelled prompt, dead subprocess: the connection's state
                    # is unknown, so retire it rather than reuse it.
                    agent.broken = True
                    raise
                finally:
                    # A turn that ends by cancel or crash never reaches the flush
                    # in _run_turn, and a tool call the agent had started would
                    # go unmentioned. Already-announced calls yield nothing, so
                    # the normal path is unaffected. Nested so a cancellation
                    # landing here still propagates without skipping unbind.
                    try:
                        with suppress(Exception):
                            await agent.client.flush_tool_calls()
                    finally:
                        await agent.client.unbind()
        finally:
            # Outside the lock: a turn cancelled while queued behind another
            # never takes it, and must still give the claim back.
            agent.claims -= 1
            if not agent.pooled or agent.broken:
                await _close_agent(agent)

    async def aclose(self) -> None:
        """Shut down every pooled agent process."""
        async with self._pool_lock:
            agents = list(self._agents.values())
            self._agents.clear()
        for agent in agents:
            await _close_agent(agent)

    async def _run_turn(
        self, agent: _Agent, session: BackendSession, request: RunRequest
    ) -> None:
        session_id = await self._open_session(agent, request, session)
        # So an A2A cancel ends the turn cleanly instead of killing the process
        # from under whatever tool was mid-flight.
        session.set_canceller(lambda: agent.conn.cancel(session_id=session_id))
        response = await agent.conn.prompt(
            prompt=prompt_blocks(request, agent.capabilities),
            session_id=session_id,
        )
        # A prompt's reply is handled inline by the receive loop while the
        # session/update notifications it interleaved go through the dispatch
        # queue. Without waiting, the turn ends while the agent's last words are
        # still in flight and they are lost behind the end-of-stream sentinel.
        with suppress(TimeoutError):
            await asyncio.wait_for(agent.queue.join(), _DRAIN_TIMEOUT)
        await agent.client.flush_tool_calls()
        await session.emit(
            Result(
                session_id=session_id,
                cost_usd=agent.client.cost_usd,
                # ACP reports no turn count; its usage is token-based.
                num_turns=None,
                usage=response.usage.model_dump() if response.usage else None,
                stop_reason=response.stop_reason,
            )
        )

    async def _acquire(self, context_id: str | None) -> _Agent:
        """Return the agent process for a context, launching one if needed."""
        if context_id is None:
            # Nothing to pool under: one process for this turn only.
            agent = await self._spawn()
            agent.pooled = False
            agent.claims += 1
            return agent
        async with self._pool_lock:
            existing = self._agents.pop(context_id, None)
            if existing is not None:
                if existing.usable:
                    # Re-insert so the most recently used lands last, making
                    # eviction least-recently-used.
                    self._agents[context_id] = existing
                    existing.claims += 1
                    return existing
                await _close_agent(existing)
            await self._evict_if_full()
            agent = await self._spawn()
            agent.claims += 1
            self._agents[context_id] = agent
            return agent

    async def _evict_if_full(self) -> None:
        """Close unclaimed agents until there is room for one more.

        Evicting a claimed agent would close the connection under a turn about
        to prompt on it, so a fully claimed pool overshoots instead.
        """
        while len(self._agents) >= _MAX_AGENTS:
            victim = next(
                (key for key, a in self._agents.items() if a.claims == 0), None
            )
            if victim is None:
                logger.warning("ACP agent pool at capacity with every agent claimed")
                return
            logger.info("closing idle ACP agent for context %s", victim)
            await _close_agent(self._agents.pop(victim))

    async def _spawn(self) -> _Agent:
        """Launch an agent subprocess and complete the ACP handshake."""
        # The ACP Client base declares terminal/* and ext_* with empty bodies as
        # optional overrides; we advertise no terminal capability, so the agent
        # never calls them. mypy reads the empty bodies as abstract, hence the
        # scoped ignore.
        client = _BridgeClient(cwd=self.cwd)  # type: ignore[abstract]
        stack = AsyncExitStack()
        # Our own queue and dispatcher, so a turn can wait for the agent's
        # updates to have actually become events. See dispatch.py and _run_turn.
        queue = InMemoryMessageQueue()
        try:
            conn, process = await stack.enter_async_context(
                spawn_agent_process(
                    client,
                    self.command,
                    *self.args,
                    env=self.env,
                    cwd=self.cwd,
                    queue=queue,
                    dispatcher_factory=OrderedDispatcher,
                )
            )
            init = await conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=s.ClientCapabilities(
                    fs=s.FileSystemCapabilities(
                        read_text_file=True, write_text_file=True
                    ),
                    terminal=True,
                ),
            )
        except BaseException:
            await stack.aclose()
            raise
        return _Agent(
            stack=stack,
            conn=conn,
            process=process,
            client=client,
            queue=queue,
            capabilities=init.agent_capabilities,
        )

    async def _open_session(
        self, agent: _Agent, request: RunRequest, session: BackendSession
    ) -> str:
        if agent.session_id is not None and request.resume in (None, agent.session_id):
            # This process already holds the conversation.
            return agent.session_id
        if request.resume:
            if getattr(agent.capabilities, "load_session", False):
                await agent.conn.load_session(
                    cwd=self.cwd, session_id=request.resume, mcp_servers=[]
                )
                agent.session_id = request.resume
                return request.resume
            # Otherwise the caller gets a confident answer from an agent that
            # never saw the conversation the turn refers to.
            await session.emit(
                Notice(
                    f"the {self.agent} agent does not support session/load, so "
                    "this turn starts a fresh session; earlier turns in this "
                    "context are not in its view"
                )
            )
        # No resume, or the agent can't reload one: start fresh. The executor
        # learns the id from the Result and maps the A2A context onto it.
        response = await agent.conn.new_session(cwd=self.cwd, mcp_servers=[])
        agent.session_id = response.session_id
        return response.session_id
