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

    ``acp`` and ``claude`` are imported lazily so the echo backend works without
    their runtime dependencies (the ACP SDK / the Claude Agent SDK) present.
    """
    if name == "echo":
        return EchoBackend()
    if name == "acp":
        from .acp import ACPBackend

        return ACPBackend(**kwargs)
    if name == "claude":
        from .claude import ClaudeBackend

        return ClaudeBackend(**kwargs)
    raise ValueError(f"unknown backend: {name!r} (expected 'acp', 'claude', or 'echo')")
