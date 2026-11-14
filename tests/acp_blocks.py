"""Builders for the ACP schema objects the tests send.

``agent-client-protocol`` 1.0 dropped the ``acp.helpers`` shortcuts
(``text_block``, ``update_plan``, ...). These rebuild the ones the tests used,
with the same signatures, so a test reads as the update it sends.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from acp import schema as s


def text_block(text: str) -> s.TextContentBlock:
    return s.TextContentBlock(type="text", text=text)


def image_block(data: str, mime_type: str) -> s.ImageContentBlock:
    return s.ImageContentBlock(type="image", data=data, mime_type=mime_type)


def tool_content(
    block: s.TextContentBlock | s.ImageContentBlock,
) -> s.ContentToolCallContent:
    return s.ContentToolCallContent(type="content", content=block)


def tool_diff_content(
    path: str, new_text: str, old_text: str | None = None
) -> s.FileEditToolCallContent:
    return s.FileEditToolCallContent(
        type="diff", path=path, new_text=new_text, old_text=old_text
    )


def plan_entry(
    content: str, *, priority: str = "medium", status: str = "pending"
) -> s.PlanEntry:
    return s.PlanEntry(content=content, priority=priority, status=status)  # type: ignore[arg-type]


def update_plan(entries: Iterable[s.PlanEntry]) -> s.AgentPlanUpdate:
    return s.AgentPlanUpdate(session_update="plan", entries=list(entries))


def update_agent_message_text(text: str) -> s.AgentMessageChunk:
    return s.AgentMessageChunk(
        session_update="agent_message_chunk", content=text_block(text)
    )


def update_agent_thought_text(text: str) -> s.AgentThoughtChunk:
    return s.AgentThoughtChunk(
        session_update="agent_thought_chunk", content=text_block(text)
    )


def start_tool_call(
    tool_call_id: str,
    title: str,
    *,
    kind: Any = None,
    status: Any = None,
    content: Sequence[Any] | None = None,
    raw_input: Any = None,
) -> s.ToolCallStart:
    return s.ToolCallStart(
        session_update="tool_call",
        tool_call_id=tool_call_id,
        title=title,
        kind=kind,
        status=status,
        content=list(content) if content is not None else None,
        raw_input=raw_input,
    )


def update_tool_call(
    tool_call_id: str,
    *,
    title: str | None = None,
    kind: Any = None,
    status: Any = None,
    content: Sequence[Any] | None = None,
    raw_input: Any = None,
) -> s.ToolCallProgress:
    return s.ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id=tool_call_id,
        title=title,
        kind=kind,
        status=status,
        content=list(content) if content is not None else None,
        raw_input=raw_input,
    )
