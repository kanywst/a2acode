"""Backends drive Claude Code and emit normalized events."""

from __future__ import annotations

from .base import (
    Backend,
    BackendEvent,
    FileChange,
    PermissionDecision,
    PermissionRequest,
    Result,
    RunRequest,
    TextDelta,
    ToolUse,
)
from .echo import EchoBackend
from .session import BackendSession

__all__ = [
    "Backend",
    "BackendEvent",
    "BackendSession",
    "FileChange",
    "PermissionDecision",
    "PermissionRequest",
    "Result",
    "RunRequest",
    "TextDelta",
    "ToolUse",
    "EchoBackend",
    "make_backend",
]


def make_backend(name: str, **kwargs) -> Backend:
    """Construct a backend by name.

    ``claude`` is imported lazily so the echo backend works without the Claude
    Agent SDK's runtime dependencies present.
    """
    if name == "echo":
        return EchoBackend()
    if name == "claude":
        from .claude import ClaudeBackend

        return ClaudeBackend(**kwargs)
    raise ValueError(f"unknown backend: {name!r} (expected 'echo' or 'claude')")
