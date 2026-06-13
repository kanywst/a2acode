"""Backend abstraction.

A backend drives Claude Code and yields a normalized stream of events. The
A2A layer never imports the Claude Agent SDK directly — it only consumes these
events. That keeps the protocol mapping in one place and lets us swap the
underlying driver (Agent SDK today, raw CLI later) without touching the server.

Backends implement ``drive(session, request)``: they push events onto the
session and, when a tool needs approval, call ``session.request_permission(...)``
which parks until the A2A caller responds. This is what lets a permission prompt
become an A2A ``input-required`` round trip rather than being silently skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
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
class FileChange:
    """A file was written or edited during the run."""

    path: str
    diff: str


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


BackendEvent = TextDelta | ToolUse | FileChange | PermissionRequest | Result


@dataclass(slots=True)
class RunRequest:
    """One turn of work handed to a backend."""

    prompt: str
    context_id: str | None = None
    resume: str | None = None


@runtime_checkable
class Backend(Protocol):
    """Anything that can drive Claude Code and emit normalized events."""

    name: str

    async def drive(self, session: BackendSession, request: RunRequest) -> None:
        """Run one turn, emitting events onto ``session`` until it returns."""
        ...
