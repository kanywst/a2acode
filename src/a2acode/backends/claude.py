"""Claude backend.

Drives Claude Code through the Claude Agent SDK's bidirectional client and
normalizes its typed message stream into backend events. Tool calls, file edits,
run cost, and the session id: everything the "text in, text out" wrappers
discard is preserved for the A2A layer to map onto the protocol.

Permission prompts are routed through ``can_use_tool`` into the session's
``request_permission``, so the caller approves or denies a tool over A2A instead
of the server skipping it.

Authentication follows whatever the Claude CLI is configured with. For a server
that answers on behalf of other agents that means an Anthropic API key (or
Bedrock/Vertex); subscription credentials are not permitted for third-party
serving.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionMode,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SettingSource,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from .base import (
    BackendEvent,
    Plan,
    PlanStep,
    Result,
    RunRequest,
    TextDelta,
    ToolResult,
    ToolUse,
)
from .diff import file_changes
from .session import BackendSession

# Tool output is relayed as a short excerpt; a single result can be a whole test
# run or file dump.
_MAX_TOOL_OUTPUT = 2000

# The tool whose input carries Claude's plan for the turn.
_PLAN_TOOL = "TodoWrite"


def events_from_message(message: object) -> Iterator[BackendEvent]:
    """Map one Claude Agent SDK message to normalized backend events.

    Pure and side-effect free so the translation can be unit tested without a
    live Claude session.
    """
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                if block.text:
                    yield TextDelta(text=block.text)
            elif isinstance(block, ToolUseBlock):
                tool_input = dict(block.input or {})
                yield ToolUse(block.name, tool_input, block.id)
                yield from file_changes(block.name, tool_input)
                if block.name == _PLAN_TOOL:
                    yield from _plan_from_todos(tool_input)
    elif isinstance(message, UserMessage):
        # Tool results come back as a user message: this is where the run says
        # whether the tool the caller just watched start actually succeeded.
        if isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    yield _tool_result(block)
    elif isinstance(message, ResultMessage):
        yield Result(
            session_id=message.session_id,
            cost_usd=message.total_cost_usd,
            num_turns=message.num_turns,
            usage=message.usage,
        )


def _plan_from_todos(tool_input: dict[str, Any]) -> Iterator[Plan]:
    """Derive the agent's plan from a TodoWrite call.

    The Claude SDK has no plan message of its own; the todo list *is* the plan,
    and it arrives as the input to a tool call. ACP models the same thing as a
    first-class session update, so this is where the two backends converge on
    one event.
    """
    todos = tool_input.get("todos")
    if not isinstance(todos, list):
        return
    steps = [
        PlanStep(
            content=str(todo.get("content")),
            status=str(todo.get("status") or "pending"),
        )
        for todo in todos
        if isinstance(todo, dict) and todo.get("content")
    ]
    if steps:
        yield Plan(steps=steps)


def _tool_result(block: ToolResultBlock) -> ToolResult:
    """Normalize one tool result.

    The SDK's content is either a plain string or the raw block list the tool
    returned; only the text of the latter is meaningful to a remote caller, so
    non-text blocks are skipped rather than stringified.
    """
    content = block.content
    if isinstance(content, list):
        text = "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("text")
        )
    else:
        text = content or ""
    if len(text) > _MAX_TOOL_OUTPUT:
        text = text[:_MAX_TOOL_OUTPUT] + " …"
    return ToolResult(
        tool_use_id=block.tool_use_id, failed=bool(block.is_error), output=text
    )


class ClaudeBackend:
    name = "claude"

    def __init__(
        self,
        *,
        cwd: str | None = None,
        allowed_tools: list[str] | None = None,
        permission_mode: PermissionMode | None = None,
        model: str | None = None,
        max_budget_usd: float | None = None,
        setting_sources: list[SettingSource] | None = None,
    ) -> None:
        self.cwd = os.path.abspath(cwd or os.getcwd())
        self.allowed_tools = allowed_tools
        self.permission_mode = permission_mode
        self.model = model
        self.max_budget_usd = max_budget_usd
        # A server should not inherit a developer's personal tool allowlist:
        # default to loading no settings so every tool routes through the A2A
        # permission round trip. Pass e.g. ["project"] to opt back in.
        self.setting_sources: list[SettingSource] = (
            [] if setting_sources is None else setting_sources
        )

    def _options(self, request: RunRequest, can_use_tool) -> ClaudeAgentOptions:
        options = ClaudeAgentOptions(
            cwd=self.cwd,
            can_use_tool=can_use_tool,
            setting_sources=self.setting_sources,
        )
        if request.resume:
            options.resume = request.resume
        if self.allowed_tools:
            options.allowed_tools = self.allowed_tools
        if self.permission_mode:
            options.permission_mode = self.permission_mode
        if self.model:
            options.model = self.model
        if self.max_budget_usd is not None:
            options.max_budget_usd = self.max_budget_usd
        return options

    async def drive(self, session: BackendSession, request: RunRequest) -> None:
        async def can_use_tool(tool_name, tool_input, context):
            description = getattr(context, "display_name", "") or tool_name
            decision = await session.request_permission(
                tool_name, dict(tool_input or {}), description
            )
            if decision.allow:
                return PermissionResultAllow()
            return PermissionResultDeny(
                message=decision.message or "Denied by A2A caller"
            )

        options = self._options(request, can_use_tool)
        async with ClaudeSDKClient(options=options) as client:
            await client.query(request.prompt)
            async for message in client.receive_response():
                for event in events_from_message(message):
                    await session.emit(event)
