"""Backend abstraction.

A backend drives Claude Code and yields a normalized stream of events. The
A2A layer never imports the Claude Agent SDK directly; it only consumes these
events. That keeps the protocol mapping in one place and lets us swap the
underlying driver (Agent SDK today, raw CLI later) without touching the server.

Backends implement ``drive(session, request)``: they push events onto the
session and, when a tool needs approval, call ``session.request_permission(...)``
which parks until the A2A caller responds. This is what lets a permission prompt
become an A2A ``input-required`` round trip rather than being silently skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .session import BackendSession


@dataclass(slots=True)
class TextDelta:
    """A chunk of assistant-authored text."""

    text: str


@dataclass(slots=True)
class ToolUse:
    """The agent decided to run a tool (Bash, Edit, Read, ...)."""

    name: str
    tool_input: dict[str, Any]
    tool_use_id: str


@dataclass(slots=True)
class ToolResult:
    """A tool call reached a terminal state.

    ``name`` is best-effort: an agent may report the outcome without repeating
    the title, leaving the consumer to fall back on the ``ToolUse``'s name.
    """

    tool_use_id: str
    name: str = ""
    failed: bool = False
    output: str = ""


@dataclass(slots=True)
class FileChange:
    """A file was written or edited during the run."""

    path: str
    diff: str


@dataclass(slots=True)
class PlanStep:
    """One entry in the agent's plan."""

    content: str
    status: str = "pending"
    priority: str = ""


@dataclass(slots=True)
class Plan:
    """The agent's current plan for the turn.

    Reported by replacement rather than by delta, so a consumer renders the
    latest one instead of accumulating them.
    """

    steps: list[PlanStep]


@dataclass(slots=True)
class Notice:
    """A note about the run itself rather than about the agent's work.

    Emitted when the server cannot do exactly what the caller asked and carries
    on differently — a resume the agent cannot honour, say — so the answer does
    not arrive under conditions the caller does not know about.
    """

    text: str


@dataclass(slots=True)
class PermissionRequest:
    """A tool needs the caller's approval before it can run."""

    request_id: str
    tool_name: str
    tool_input: dict[str, Any]
    description: str = ""


@dataclass(slots=True)
class PermissionDecision:
    """The caller's answer to a PermissionRequest."""

    request_id: str
    allow: bool
    message: str = ""


@dataclass(slots=True)
class Result:
    """Terminal event carrying run metadata."""

    session_id: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    usage: dict[str, Any] | None = None


BackendEvent = (
    TextDelta
    | ToolUse
    | ToolResult
    | FileChange
    | Plan
    | Notice
    | PermissionRequest
    | Result
)


@dataclass(slots=True)
class Attachment:
    """A non-text part of the caller's message, handed to the agent.

    At most one of ``text``, ``data``, and ``uri`` is set. ``truncated`` says
    the content was cut to fit, so a backend passes that on rather than
    presenting a fragment as the whole file.
    """

    name: str
    media_type: str = ""
    text: str | None = None
    data: bytes | None = None
    uri: str | None = None
    truncated: bool = False


@dataclass(slots=True)
class RunRequest:
    """One turn of work handed to a backend."""

    prompt: str
    context_id: str | None = None
    resume: str | None = None
    attachments: list[Attachment] = field(default_factory=list)


@runtime_checkable
class Backend(Protocol):
    """Anything that can drive Claude Code and emit normalized events."""

    name: str

    async def drive(self, session: BackendSession, request: RunRequest) -> None:
        """Run one turn, emitting events onto ``session`` until it returns."""
        ...


@runtime_checkable
class ClosableBackend(Protocol):
    """A backend holding resources that outlive a single run.

    Separate from ``Backend`` so the base protocol stays "anything with a name
    and a drive"; only a backend that pools something implements this.
    """

    async def aclose(self) -> None:
        """Release everything held across runs."""
        ...
