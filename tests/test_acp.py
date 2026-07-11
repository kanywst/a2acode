"""Unit tests for the ACP backend's pure mapping and permission selection.

These exercise the protocol translation without launching an agent subprocess:
``events_from_update`` and ``select_option`` are pure, and ``_BridgeClient``'s
permission round trip is driven against a fake session.
"""

from __future__ import annotations

import pytest
from acp import schema as s
from acp import text_block, tool_diff_content

from a2acode.backends.acp import (
    _BridgeClient,
    events_from_update,
    select_option,
)
from a2acode.backends.base import (
    FileChange,
    PermissionDecision,
    TextDelta,
    ToolUse,
)


def _opts() -> list[s.PermissionOption]:
    return [
        s.PermissionOption(option_id="a", name="Allow", kind="allow_once"),
        s.PermissionOption(option_id="A", name="Always", kind="allow_always"),
        s.PermissionOption(option_id="r", name="Reject", kind="reject_once"),
    ]


def test_agent_message_chunk_maps_to_text_delta():
    update = s.AgentMessageChunk(
        session_update="agent_message_chunk", content=text_block("hello")
    )
    events = list(events_from_update(update))
    assert events == [TextDelta(text="hello")]


def test_empty_text_chunk_yields_nothing():
    update = s.AgentMessageChunk(
        session_update="agent_message_chunk", content=text_block("")
    )
    assert list(events_from_update(update)) == []


def test_tool_call_start_with_diff_yields_tooluse_and_filechange():
    update = s.ToolCallStart(
        session_update="tool_call",
        tool_call_id="t1",
        title="Write a.py",
        kind="edit",
        raw_input={"file_path": "a.py"},
        content=[tool_diff_content(path="a.py", new_text="x = 1\n", old_text=None)],
    )
    events = list(events_from_update(update))

    assert len(events) == 2
    assert isinstance(events[0], ToolUse)
    assert events[0].name == "Write a.py"
    assert events[0].tool_use_id == "t1"
    assert events[0].tool_input == {"file_path": "a.py"}
    assert isinstance(events[1], FileChange)
    assert events[1].path == "a.py"
    assert "+x = 1" in events[1].diff


def test_tool_call_progress_yields_only_filechange():
    update = s.ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="t1",
        content=[
            tool_diff_content(path="a.py", new_text="y = 2\n", old_text="x = 1\n")
        ],
    )
    events = list(events_from_update(update))
    assert len(events) == 1
    assert isinstance(events[0], FileChange)
    assert "-x = 1" in events[0].diff
    assert "+y = 2" in events[0].diff


def test_usage_update_yields_nothing():
    update = s.UsageUpdate(session_update="usage_update", used=10, size=100)
    assert list(events_from_update(update)) == []


def test_non_mapping_raw_input_becomes_empty_dict():
    update = s.ToolCallStart(
        session_update="tool_call", tool_call_id="t1", title="x", raw_input="not-a-dict"
    )
    events = list(events_from_update(update))
    assert events[0].tool_input == {}


def test_select_option_prefers_one_shot():
    assert select_option(_opts(), allow=True) == "a"
    assert select_option(_opts(), allow=False) == "r"


def test_select_option_falls_back_to_always_when_no_once():
    opts = [s.PermissionOption(option_id="A", name="Always", kind="allow_always")]
    assert select_option(opts, allow=True) == "A"
    assert select_option(opts, allow=False) is None


class _FakeSession:
    def __init__(self, decision: PermissionDecision) -> None:
        self._decision = decision
        self.asked: tuple[str, dict, str] | None = None

    async def request_permission(self, name, tool_input, description):
        self.asked = (name, tool_input, description)
        return self._decision


@pytest.mark.asyncio
async def test_request_permission_allow_selects_allow_option():
    session = _FakeSession(PermissionDecision(request_id="x", allow=True))
    client = _BridgeClient(session)  # type: ignore[arg-type]
    tool_call = s.ToolCallUpdate(tool_call_id="t1", title="Run ls", kind="execute")

    resp = await client.request_permission("sess", tool_call, _opts())

    assert isinstance(resp.outcome, s.AllowedOutcome)
    assert resp.outcome.option_id == "a"
    assert session.asked == ("Run ls", {}, "Run ls")


@pytest.mark.asyncio
async def test_request_permission_deny_selects_reject_option():
    session = _FakeSession(PermissionDecision(request_id="x", allow=False))
    client = _BridgeClient(session)  # type: ignore[arg-type]
    tool_call = s.ToolCallUpdate(tool_call_id="t1", title="rm -rf", kind="execute")

    resp = await client.request_permission("sess", tool_call, _opts())

    assert isinstance(resp.outcome, s.AllowedOutcome)
    assert resp.outcome.option_id == "r"


@pytest.mark.asyncio
async def test_request_permission_cancels_when_no_matching_option():
    session = _FakeSession(PermissionDecision(request_id="x", allow=False))
    client = _BridgeClient(session)  # type: ignore[arg-type]
    allow_only = [s.PermissionOption(option_id="a", name="Allow", kind="allow_once")]
    tool_call = s.ToolCallUpdate(tool_call_id="t1", title="x")

    resp = await client.request_permission("sess", tool_call, allow_only)

    assert isinstance(resp.outcome, s.DeniedOutcome)


@pytest.mark.asyncio
async def test_write_and_read_within_workspace(tmp_path):
    session = _FakeSession(PermissionDecision(request_id="x", allow=True))
    client = _BridgeClient(session, str(tmp_path))  # type: ignore[arg-type]
    target = tmp_path / "sub" / "a.txt"

    await client.write_text_file("sess", str(target), "hello\n")
    assert target.read_text() == "hello\n"
    resp = await client.read_text_file("sess", str(target))
    assert resp.content == "hello\n"


@pytest.mark.asyncio
async def test_read_outside_workspace_is_rejected(tmp_path):
    session = _FakeSession(PermissionDecision(request_id="x", allow=True))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret\n")
    client = _BridgeClient(session, str(workspace))  # type: ignore[arg-type]

    with pytest.raises(PermissionError):
        await client.read_text_file("sess", str(secret))
    with pytest.raises(PermissionError):
        await client.write_text_file("sess", str(tmp_path / "escape.txt"), "x")


@pytest.mark.asyncio
async def test_read_with_line_and_limit(tmp_path):
    session = _FakeSession(PermissionDecision(request_id="x", allow=True))
    client = _BridgeClient(session, str(tmp_path))  # type: ignore[arg-type]
    target = tmp_path / "a.txt"
    target.write_text("l1\nl2\nl3\nl4\n")

    resp = await client.read_text_file("sess", str(target), line=2, limit=2)
    assert resp.content == "l2\nl3\n"
    # A non-positive line number must not slice from the end.
    resp = await client.read_text_file("sess", str(target), line=0, limit=1)
    assert resp.content == "l1\n"
    # A negative limit must not slice from the end; it yields nothing.
    resp = await client.read_text_file("sess", str(target), line=1, limit=-1)
    assert resp.content == ""


@pytest.mark.asyncio
async def test_session_update_captures_cost():
    session = _FakeSession(PermissionDecision(request_id="x", allow=True))
    client = _BridgeClient(session)  # type: ignore[arg-type]
    await client.session_update(
        "sess",
        s.UsageUpdate(
            session_update="usage_update",
            used=10,
            size=100,
            cost=s.Cost(amount=0.42, currency="USD"),
        ),
    )
    assert client.cost_usd == 0.42
