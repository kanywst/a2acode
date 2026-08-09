"""End-to-end against a real ACP agent subprocess.

Everything else mocks one side of the pipe. These drive the whole backend
against ``fake_agent.py`` over actual stdio: the handshake, session lifecycle,
the permission call arriving from the agent, terminals, file reads, and process
reuse across turns. No vendor, credential, or network involved, so it runs in
CI like any other test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from a2acode.backends.acp import ACPBackend
from a2acode.backends.base import (
    FileChange,
    Notice,
    PermissionDecision,
    PermissionRequest,
    Plan,
    Result,
    RunRequest,
    TextDelta,
    Thought,
    ToolResult,
    ToolUse,
)
from a2acode.backends.session import BackendSession

AGENT = str(Path(__file__).parent / "fake_agent.py")

pytestmark = pytest.mark.asyncio


def _backend(tmp_path, **env) -> ACPBackend:
    return ACPBackend(
        command=sys.executable,
        args=[AGENT],
        cwd=str(tmp_path),
        env={"PYTHONUNBUFFERED": "1", **env},
    )


async def _turn(backend, prompt, *, context_id="ctx-1", resume=None, allow=True):
    """Run one turn to completion, answering any permission request."""
    session = BackendSession()
    session.start(
        lambda s: backend.drive(
            s, RunRequest(prompt=prompt, context_id=context_id, resume=resume)
        )
    )
    events: list = []
    try:
        while not session.done:
            async for event in session.drain():
                events.append(event)
                if isinstance(event, PermissionRequest):
                    session.resolve(
                        PermissionDecision(request_id=event.request_id, allow=allow)
                    )
    finally:
        await session.close()
    return events


async def test_a_whole_turn_round_trips_through_a_real_subprocess(tmp_path):
    backend = _backend(tmp_path)
    try:
        events = await _turn(backend, "just talk")
    finally:
        await backend.aclose()

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "turn 1: done"
    assert any(isinstance(e, Thought) and e.text == "considering" for e in events)

    result = next(e for e in events if isinstance(e, Result))
    assert result.session_id == "sess-1"
    assert result.stop_reason == "end_turn"
    assert result.usage["total_tokens"] == 42


async def test_a_follow_up_turn_reuses_the_same_process_and_session(tmp_path):
    backend = _backend(tmp_path)
    try:
        first = await _turn(backend, "hello")
        session_id = next(e for e in first if isinstance(e, Result)).session_id

        second = await _turn(backend, "again", resume=session_id)
    finally:
        await backend.aclose()

    text = "".join(e.text for e in second if isinstance(e, TextDelta))
    # The agent counts its own turns, so "turn 2" proves the same process - and
    # the same session on it - served both, with no respawn and no session/load.
    assert text == "turn 2: done"
    assert next(e for e in second if isinstance(e, Result)).session_id == session_id


async def test_a_different_context_gets_its_own_process(tmp_path):
    backend = _backend(tmp_path)
    try:
        await _turn(backend, "hello", context_id="ctx-a")
        other = await _turn(backend, "hello", context_id="ctx-b")
    finally:
        await backend.aclose()

    text = "".join(e.text for e in other if isinstance(e, TextDelta))
    assert text == "turn 1: done"


async def test_the_agents_permission_request_reaches_the_caller_and_back(tmp_path):
    backend = _backend(tmp_path)
    try:
        allowed = await _turn(backend, "ask", allow=True)
        denied = await _turn(backend, "ask", context_id="ctx-2", allow=False)
    finally:
        await backend.aclose()

    request = next(e for e in allowed if isinstance(e, PermissionRequest))
    assert request.tool_name == "rm -rf /"
    assert "permission=ok" in "".join(
        e.text for e in allowed if isinstance(e, TextDelta)
    )
    assert "permission=no" in "".join(
        e.text for e in denied if isinstance(e, TextDelta)
    )


async def test_a_tool_call_carries_its_diff_and_its_outcome(tmp_path):
    backend = _backend(tmp_path)
    try:
        events = await _turn(backend, "edit the file")
    finally:
        await backend.aclose()

    assert any(isinstance(e, ToolUse) and e.name == "Write calc.py" for e in events)
    change = next(e for e in events if isinstance(e, FileChange))
    assert change.path == "calc.py"
    assert "-a" in change.diff and "+b" in change.diff
    result = next(e for e in events if isinstance(e, ToolResult))
    assert not result.failed
    assert result.output == "written"


async def test_a_tool_call_keeps_the_arguments_it_reported_late(tmp_path):
    backend = _backend(tmp_path)
    try:
        events = await _turn(backend, "peek at the file")
    finally:
        await backend.aclose()

    uses = [e for e in events if isinstance(e, ToolUse)]
    assert len(uses) == 1
    assert uses[0].tool_input == {"file_path": "app.py"}
    assert uses[0].name == "Read app.py"
    assert next(e for e in events if isinstance(e, ToolResult)).name == "Read app.py"


async def test_a_failing_tool_is_reported_as_failed(tmp_path):
    backend = _backend(tmp_path)
    try:
        events = await _turn(backend, "boom")
    finally:
        await backend.aclose()

    result = next(e for e in events if isinstance(e, ToolResult) and e.failed)
    assert "2 tests failed" in result.output


async def test_the_agents_plan_arrives_as_steps(tmp_path):
    backend = _backend(tmp_path)
    try:
        events = await _turn(backend, "make a plan")
    finally:
        await backend.aclose()

    plan = next(e for e in events if isinstance(e, Plan))
    assert [(s.content, s.status) for s in plan.steps] == [
        ("look", "completed"),
        ("fix", "in_progress"),
    ]


async def test_the_agent_can_run_a_terminal_through_us(tmp_path):
    backend = _backend(tmp_path)
    try:
        events = await _turn(backend, "shell", allow=True)
    finally:
        await backend.aclose()

    # The terminal went through the caller for approval like any tool.
    assert any(
        isinstance(e, PermissionRequest) and e.tool_name == "Terminal" for e in events
    )
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "shell said from the terminal" in text


async def test_a_denied_terminal_is_refused_without_killing_the_turn(tmp_path):
    backend = _backend(tmp_path)
    try:
        events = await _turn(backend, "shell", allow=False)
    finally:
        await backend.aclose()

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    # The refusal reaches the agent as a protocol error it can handle, and the
    # command never ran.
    assert "shell refused" in text
    assert "from the terminal" not in text
    assert text.endswith("done")


async def test_the_agent_reads_files_through_us_inside_the_workspace(tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\n")
    backend = _backend(tmp_path)
    try:
        events = await _turn(backend, "readfile")
    finally:
        await backend.aclose()

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "file has 6 chars" in text


async def test_an_agent_without_load_session_says_continuity_was_lost(tmp_path):
    backend = _backend(tmp_path)
    try:
        # A resume the pooled process does not already hold, on an agent that
        # cannot reload one: the caller must be told, not quietly given a fresh
        # conversation.
        events = await _turn(backend, "hello", resume="sess-from-a-dead-process")
    finally:
        await backend.aclose()

    notice = next(e for e in events if isinstance(e, Notice))
    assert "session/load" in notice.text


async def test_an_agent_with_load_session_reloads_it(tmp_path):
    backend = _backend(tmp_path, FAKE_AGENT_LOAD_SESSION="1")
    try:
        events = await _turn(backend, "hello", resume="sess-from-a-dead-process")
    finally:
        await backend.aclose()

    assert not any(isinstance(e, Notice) for e in events)
    result = next(e for e in events if isinstance(e, Result))
    assert result.session_id == "sess-from-a-dead-process"
