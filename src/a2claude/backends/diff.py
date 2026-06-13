"""Synthesize unified diffs from Claude Code's file-editing tool calls.

The tool input describes the intended change, so the diff is the proposed edit
rather than a post-hoc comparison of disk state — which is what a caller wants
to see streamed while the work is still in progress.
"""

from __future__ import annotations

import difflib
from typing import Any

from .base import FileChange

_EDIT_TOOLS = {"Write", "Edit", "MultiEdit"}


def _unified(path: str, before: str, after: str) -> str:
    before_lines = before.splitlines(keepends=True) if before else []
    after_lines = after.splitlines(keepends=True) if after else []
    diff = "".join(
        difflib.unified_diff(
            before_lines, after_lines, fromfile=f"a/{path}", tofile=f"b/{path}"
        )
    )
    if diff and not diff.endswith("\n"):
        diff += "\n"
    return diff


def file_changes(tool_name: str, tool_input: dict[str, Any]) -> list[FileChange]:
    """Return the file changes a tool call would make, if any."""
    if tool_name not in _EDIT_TOOLS:
        return []
    path = tool_input.get("file_path") or tool_input.get("path")
    if not path:
        return []

    if tool_name == "Write":
        diff = _unified(path, "", str(tool_input.get("content", "")))
    elif tool_name == "Edit":
        diff = _unified(
            path,
            str(tool_input.get("old_string", "")),
            str(tool_input.get("new_string", "")),
        )
    else:  # MultiEdit — edits may be malformed; tolerate anything non-dict.
        edits = tool_input.get("edits")
        if not isinstance(edits, list):
            return []
        diff = "".join(
            _unified(
                path,
                str(edit.get("old_string", "")),
                str(edit.get("new_string", "")),
            )
            for edit in edits
            if isinstance(edit, dict)
        )

    return [FileChange(path=str(path), diff=diff)] if diff else []
