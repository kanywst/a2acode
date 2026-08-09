"""Unit tests for the claude backend's mapping and plan tracking, no live Claude."""

from __future__ import annotations

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from a2acode.backends.base import (
    FileChange,
    Plan,
    Result,
    RunRequest,
    TextDelta,
    Thought,
    ToolResult,
    ToolUse,
)
from a2acode.backends.claude import (
    ClaudeBackend,
    PlanTracker,
    events_from_message,
)


def test_events_from_assistant_message_with_write():
    message = AssistantMessage(
        content=[
            TextBlock(text="creating the file"),
            ToolUseBlock(
                id="t1",
                name="Write",
                input={"file_path": "a.py", "content": "x = 1\n"},
            ),
        ],
        model="claude-test",
    )
    events = list(events_from_message(message))

    assert len(events) == 3
    assert isinstance(events[0], TextDelta)
    assert events[0].text == "creating the file"
    assert isinstance(events[1], ToolUse)
    assert events[1].name == "Write"
    assert isinstance(events[2], FileChange)
    assert events[2].path == "a.py"
    assert "+x = 1" in events[2].diff


def test_events_from_result_message():
    message = ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=2,
        session_id="s1",
        total_cost_usd=0.0123,
        usage={"input_tokens": 5},
    )
    events = list(events_from_message(message))

    assert len(events) == 1
    result = events[0]
    assert isinstance(result, Result)
    assert result.session_id == "s1"
    assert result.cost_usd == 0.0123
    assert result.num_turns == 2


def test_tool_result_block_maps_to_tool_result():
    message = UserMessage(
        content=[ToolResultBlock(tool_use_id="t1", content="a.py\nb.py\n")]
    )
    events = list(events_from_message(message))

    assert len(events) == 1
    result = events[0]
    assert isinstance(result, ToolResult)
    assert result.tool_use_id == "t1"
    assert not result.failed
    assert result.output == "a.py\nb.py\n"


def test_errored_tool_result_is_flagged_and_block_content_is_joined():
    message = UserMessage(
        content=[
            ToolResultBlock(
                tool_use_id="t1",
                content=[{"type": "text", "text": "boom"}, {"type": "image"}],
                is_error=True,
            )
        ]
    )
    result = next(iter(events_from_message(message)))

    assert result.failed
    # The image block carries no text, so it is skipped rather than stringified.
    assert result.output == "boom"


def _created(tool_use_id: str, task_id: str) -> ToolResult:
    """The result a TaskCreate call gets back, which is where its id is."""
    return ToolResult(
        tool_use_id=tool_use_id,
        output=f"Task #{task_id} created successfully: something",
    )


def _steps(plan: Plan) -> list[tuple[str, str]]:
    return [(step.content, step.status) for step in plan.steps]


def test_the_task_tools_build_a_plan_one_entry_at_a_time():
    tracker = PlanTracker()

    first = tracker.absorb(ToolUse("TaskCreate", {"subject": "read the code"}, "t1"))
    assert _steps(first) == [("read the code", "pending")]
    # The result only teaches the tracker the id; it is not a plan change.
    assert tracker.absorb(_created("t1", "1")) is None

    second = tracker.absorb(ToolUse("TaskCreate", {"subject": "write the fix"}, "t2"))
    assert tracker.absorb(_created("t2", "2")) is None
    assert len(second.steps) == 2

    started = tracker.absorb(
        ToolUse("TaskUpdate", {"taskId": "1", "status": "completed"}, "t3")
    )
    assert _steps(started) == [
        ("read the code", "completed"),
        ("write the fix", "pending"),
    ]


def test_a_deleted_task_leaves_the_plan():
    tracker = PlanTracker()
    tracker.absorb(ToolUse("TaskCreate", {"subject": "drop me"}, "t1"))
    tracker.absorb(_created("t1", "1"))

    plan = tracker.absorb(
        ToolUse("TaskUpdate", {"taskId": "1", "status": "deleted"}, "t2")
    )
    assert plan is not None
    assert plan.steps == []


def test_an_update_to_a_task_the_tracker_never_saw_is_ignored():
    tracker = PlanTracker()
    assert (
        tracker.absorb(
            ToolUse("TaskUpdate", {"taskId": "9", "status": "completed"}, "t1")
        )
        is None
    )


def test_todowrite_still_carries_a_whole_plan():
    # Older CLIs write the list in one call, so one call is a whole plan.
    plan = PlanTracker().absorb(
        ToolUse(
            "TodoWrite",
            {
                "todos": [
                    {"content": "read the code", "status": "completed"},
                    {"content": "write the fix", "status": "in_progress"},
                ]
            },
            "t1",
        )
    )
    assert _steps(plan) == [
        ("read the code", "completed"),
        ("write the fix", "in_progress"),
    ]


def test_a_malformed_task_list_does_not_raise():
    tracker = PlanTracker()
    for todos in ("oops", [1, "x"], []):
        assert tracker.absorb(ToolUse("TodoWrite", {"todos": todos}, "t1")) is None
    assert (
        tracker.absorb(ToolUse("TaskCreate", {"description": "no subject"}, "t2"))
        is None
    )


def test_a_plan_tool_call_is_only_a_tool_use_to_the_mapper():
    # The plan needs state across calls, so it is the tracker's, not the pure
    # mapper's, which is why keying off one tool name went stale unnoticed.
    message = AssistantMessage(
        content=[ToolUseBlock(id="t1", name="TaskCreate", input={"subject": "x"})],
        model="claude-test",
    )
    assert [type(e) for e in events_from_message(message)] == [ToolUse]


def test_thinking_block_maps_to_a_thought():
    message = AssistantMessage(
        content=[
            ThinkingBlock(thinking="weighing two options", signature="sig"),
            TextBlock(text="going with the second"),
        ],
        model="claude-test",
    )
    events = list(events_from_message(message))

    assert [type(e) for e in events] == [Thought, TextDelta]
    assert events[0].text == "weighing two options"


def test_plain_text_user_message_yields_nothing():
    assert list(events_from_message(UserMessage(content="just text"))) == []


def test_tool_result_output_is_capped():
    message = UserMessage(
        content=[ToolResultBlock(tool_use_id="t1", content="x" * 5000)]
    )
    output = next(iter(events_from_message(message))).output
    assert output.endswith(" …")
    assert len(output) == 2002


def test_empty_text_block_is_skipped():
    message = AssistantMessage(content=[TextBlock(text="")], model="claude-test")
    assert list(events_from_message(message)) == []


def test_options_applies_settings():
    backend = ClaudeBackend(
        cwd="/tmp/project",
        permission_mode="acceptEdits",
        max_budget_usd=0.5,
        model="claude-test",
    )
    options = backend._options(
        RunRequest(prompt="hi", resume="sess-1"), can_use_tool=lambda *a: None
    )

    assert options.cwd == "/tmp/project"
    assert options.resume == "sess-1"
    assert options.permission_mode == "acceptEdits"
    assert options.max_budget_usd == 0.5
    # A server must not inherit the developer's personal allowlist.
    assert options.setting_sources == []
