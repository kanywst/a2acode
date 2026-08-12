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
class Thought:
    """A chunk of the agent's reasoning, distinct from its answer."""

    text: str


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
    latest one instead of accumulating them. A plan is either a step list, free
    markdown, or a pointer to a file the agent keeps it in — whichever the agent
    chose; the other fields stay empty.
    """

    steps: list[PlanStep] = field(default_factory=list)
    markdown: str = ""
    uri: str = ""


@dataclass(slots=True)
class Notice:
    """A note about the run itself rather than about the agent's work.

    Emitted when the server cannot do exactly what the caller asked and carries
    on differently — a resume the agent cannot honour, say — so the answer does
    not arrive under conditions the caller does not know about.
    """

    text: str


@dataclass(slots=True)
class PermissionOption:
    """One of the choices the agent offered for a permission request.

    ``kind`` is ACP's hint at what the choice means (``allow_once``,
    ``reject_once``, ...); ``option_id`` is what actually binds.
    """

    option_id: str
    name: str = ""
    kind: str = ""


@dataclass(slots=True)
class PermissionRequest:
    """A tool needs the caller's approval before it can run.

    ``options`` is what the agent offered, when it offered anything: a plan-mode
    gate is a three-way choice (accept the edits that follow, keep gating each
    one, keep planning) that allow/deny cannot express.
    """

    request_id: str
    tool_name: str
    tool_input: dict[str, Any]
    description: str = ""
    options: list[PermissionOption] = field(default_factory=list)


@dataclass(slots=True)
class PermissionDecision:
    """The caller's answer to a PermissionRequest.

    ``option_id`` names one of the offered options directly; without it the
    answer resolves to whichever option matches ``allow``. ``answers`` carries
    what the caller replied to questions the tool asked, keyed by question: for
    a gate that only asks, an approval alone says nothing.
    """

    request_id: str
    allow: bool
    message: str = ""
    option_id: str = ""
    answers: dict[str, str | list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class Result:
    """Terminal event carrying run metadata.

    ``stop_reason`` is why the turn ended — completed normally, hit a token or
    request ceiling, was refused, was cancelled — which a caller needs to tell a
    finished answer from a truncated one.
    """

    session_id: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    usage: dict[str, Any] | None = None
    stop_reason: str | None = None


BackendEvent = (
    TextDelta
    | Thought
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
