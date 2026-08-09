"""Claude backend.

Drives Claude Code through the Claude Agent SDK's bidirectional client and
normalizes its typed message stream into backend events. Tool calls, file edits,
run cost, and the session id: everything the "text in, text out" wrappers
discard is preserved for the A2A layer to map onto the protocol.

The agent's plan is not a message either: the task list is the plan, and it
arrives as tool calls that change one entry at a time, so ``PlanTracker`` holds
it across the run.

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
import re
from collections.abc import Iterator
from dataclasses import dataclass
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
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from .attach import append_to_prompt
from .base import (
    BackendEvent,
    Plan,
    PlanStep,
    Result,
    RunRequest,
    TextDelta,
    Thought,
    ToolResult,
    ToolUse,
)
from .diff import file_changes
from .session import BackendSession

# Tool output is relayed as a short excerpt; a single result can be a whole test
# run or file dump.
_MAX_TOOL_OUTPUT = 2000

# The tools whose calls carry Claude's plan for the turn. TodoWrite wrote the
# whole list in one call; the task tools that replaced it mutate one entry each,
# so the list has to be kept across calls rather than read off a single input.
_TODO_TOOL = "TodoWrite"
_TASK_CREATE = "TaskCreate"
_TASK_UPDATE = "TaskUpdate"
_PLAN_TOOLS = (_TODO_TOOL, _TASK_CREATE, _TASK_UPDATE)

# Contexts whose task list is kept between turns, and how much one such list
# holds: a context can be resumed for as long as the server runs, and a plan is
# a checklist someone reads, so the oldest entries are the ones to lose.
_MAX_PLANS = 256
_MAX_STEPS = 256
_MAX_PENDING = 64

# "Task #1 created successfully: ..." - a created task's id is reported by its
# result, not by the call, and TaskUpdate addresses it by that id. The creation
# wording is required: a result mentioning another task first ("Blocked by task
# #7. Task #8 created") would otherwise bind this step to that one.
_TASK_ID = re.compile(r"[Tt]ask #(\w+) created")


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
            elif isinstance(block, ThinkingBlock):
                if block.thinking:
                    yield Thought(text=block.thinking)
            elif isinstance(block, ToolUseBlock):
                tool_input = dict(block.input or {})
                yield ToolUse(block.name, tool_input, block.id)
                yield from file_changes(block.name, tool_input)
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


def _bound(store: dict[str, Any], cap: int) -> None:
    while len(store) > cap:
        del store[next(iter(store))]


@dataclass
class _Task:
    content: str
    status: str = "pending"


class PlanTracker:
    """The agent's task list, accumulated across the calls that change it.

    The Claude SDK has no plan message of its own: the task list *is* the plan
    and it arrives as tool calls. One TodoWrite carried the whole list, so a call
    was a whole plan; TaskCreate and TaskUpdate each change one entry, so the
    list lives here and every change replays it. ACP models the same thing as a
    first-class session update, which is where the two backends converge.

    A change lands when its call has succeeded: a call the caller denied would
    otherwise leave the plan showing work the agent never took on.
    """

    def __init__(self) -> None:
        # Keyed by the tool call that created the entry: a handle we have before
        # the task reports an id of its own, and ordered by creation.
        self._tasks: dict[str, _Task] = {}
        # Plan calls still waiting on their result.
        self._calls: dict[str, tuple[str, dict[str, Any]]] = {}
        self._keys: dict[str, str] = {}

    def absorb(self, event: BackendEvent) -> Plan | None:
        """Fold one event in, returning the plan when it changed."""
        if isinstance(event, ToolUse):
            if event.name in _PLAN_TOOLS:
                self._calls[event.tool_use_id] = (event.name, event.tool_input)
                # A call whose result never arrives would otherwise sit here for
                # the life of the conversation.
                _bound(self._calls, _MAX_PENDING)
        elif isinstance(event, ToolResult):
            return self._apply(event)
        return None

    def _apply(self, event: ToolResult) -> Plan | None:
        call = self._calls.pop(event.tool_use_id, None)
        if call is None or event.failed:
            return None
        name, tool_input = call
        if name == _TODO_TOOL:
            return self._todos(tool_input)
        if name == _TASK_CREATE:
            return self._create(tool_input, event.tool_use_id, event.output)
        return self._update(tool_input)

    def _todos(self, tool_input: dict[str, Any]) -> Plan | None:
        todos = tool_input.get("todos")
        if not isinstance(todos, list):
            return None
        tasks = {
            f"todo:{i}": _Task(
                content=str(todo.get("content")),
                status=str(todo.get("status") or "pending"),
            )
            for i, todo in enumerate(todos[:_MAX_STEPS])
            if isinstance(todo, dict) and todo.get("content")
        }
        if not tasks:
            return None
        # Capped at the head: one call carries the whole list, so the entries to
        # lose are the last ones, not the first.
        self._tasks, self._keys = tasks, {}
        return self._plan()

    def _create(
        self, tool_input: dict[str, Any], tool_use_id: str, output: str
    ) -> Plan | None:
        subject = str(tool_input.get("subject") or "").strip()
        if not subject:
            return None
        self._tasks[tool_use_id] = _Task(content=subject)
        _bound(self._tasks, _MAX_STEPS)
        # The id TaskUpdate will address it by is reported here, not by the call.
        found = _TASK_ID.search(output)
        if found:
            self._keys[found.group(1)] = tool_use_id
            # One id per created task, so it is bounded alongside them; an entry
            # left pointing at an evicted task just misses on lookup.
            _bound(self._keys, _MAX_STEPS)
        return self._plan()

    def _update(self, tool_input: dict[str, Any]) -> Plan | None:
        task_id = tool_input.get("taskId")
        key = self._keys.get(str(task_id)) if task_id is not None else None
        task = self._tasks.get(key) if key is not None else None
        if key is None or task is None:
            return None
        status = str(tool_input.get("status") or "")
        if status == "deleted":
            del self._tasks[key]
        else:
            task.status = status or task.status
            task.content = str(tool_input.get("subject") or task.content)
        return self._plan()

    def _plan(self) -> Plan:
        return Plan(
            steps=[
                PlanStep(content=task.content, status=task.status)
                for task in self._tasks.values()
            ]
        )


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
        # One task list per A2A context, bounded oldest-first. A resumed turn
        # keeps working the list an earlier turn built, and TaskUpdate addresses
        # a task by an id only the turn that created it saw.
        self._plans: dict[str, PlanTracker] = {}

    def _plan_for(self, request: RunRequest) -> PlanTracker:
        context_id = request.context_id
        if context_id is None:
            return PlanTracker()
        # Popped either way, so a reused context moves off the eviction front: a
        # plain reassignment would leave the key in its original position.
        previous = self._plans.pop(context_id, None)
        # Only a resumed turn inherits one. Without a resume the agent gets a
        # fresh conversation, so an earlier list is not what it works from.
        tracker = previous if previous is not None and request.resume else PlanTracker()
        self._plans[context_id] = tracker
        while len(self._plans) > _MAX_PLANS:
            del self._plans[next(iter(self._plans))]
        return tracker

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
        plan = self._plan_for(request)
        async with ClaudeSDKClient(options=options) as client:
            await client.query(append_to_prompt(request.prompt, request.attachments))
            async for message in client.receive_response():
                for event in events_from_message(message):
                    await session.emit(event)
                    update = plan.absorb(event)
                    if update is not None:
                        await session.emit(update)
