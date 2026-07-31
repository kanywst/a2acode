"""Unit tests for the claude backend's pure mapping, with no live Claude."""

from __future__ import annotations

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from a2acode.backends.base import (
    FileChange,
    Result,
    RunRequest,
    TextDelta,
    ToolResult,
    ToolUse,
)
from a2acode.backends.claude import ClaudeBackend, events_from_message


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
