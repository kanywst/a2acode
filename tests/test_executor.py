"""Executor lifecycle: eviction and session-continuity bookkeeping."""

from __future__ import annotations

from a2acode import executor as executor_mod
from a2acode.backends import (
    BackendSession,
    Notice,
    Plan,
    PlanStep,
    RunRequest,
    TextDelta,
    Thought,
    ToolResult,
    ToolUse,
    make_backend,
)
from a2acode.executor import ClaudeCodeExecutor


def test_remember_session_moves_reused_context_to_most_recent():
    executor = ClaudeCodeExecutor(make_backend("echo"))
    executor._remember_session("a", "sess-a")
    executor._remember_session("b", "sess-b")
    executor._remember_session("c", "sess-c")

    # Reusing "a" must move it off the eviction front; "b" becomes oldest.
    executor._remember_session("a", "sess-a2")
    assert next(iter(executor._session_ids)) == "b"
    assert executor._session_ids["a"] == "sess-a2"


def test_remember_session_evicts_least_recently_used(monkeypatch):
    monkeypatch.setattr(executor_mod, "_MAX_CONTEXTS", 2)
    executor = ClaudeCodeExecutor(make_backend("echo"))

    executor._remember_session("a", "sess-a")
    executor._remember_session("b", "sess-b")
    # Touch "a" so "b" is now the least recently used, then overflow.
    executor._remember_session("a", "sess-a2")
    executor._remember_session("c", "sess-c")

    assert "b" not in executor._session_ids
    assert set(executor._session_ids) == {"a", "c"}


async def _parked_session() -> BackendSession:
    """An echo session driven up to its permission request and left parked."""
    session = BackendSession()
    session.start(
        lambda s: make_backend("echo").drive(s, RunRequest(prompt="sudo reboot"))
    )
    async for _ in session.drain():
        pass
    assert session.is_parked
    return session


def _running_session() -> BackendSession:
    """A session that is neither parked nor drained (not awaiting input)."""
    session = BackendSession()
    session.start(lambda s: make_backend("echo").drive(s, RunRequest(prompt="hello")))
    assert not session.is_parked
    return session


async def test_eviction_prefers_parked_over_running(monkeypatch):
    monkeypatch.setattr(executor_mod, "_MAX_LIVE", 2)
    executor = ClaudeCodeExecutor(make_backend("echo"))

    running = _running_session()
    parked = await _parked_session()
    # Insert running first so the oldest-by-insertion entry is the running one;
    # a naive "evict oldest" would drop it. The parked one must go instead.
    executor._live["running"] = running
    executor._live["parked"] = parked

    try:
        await executor._evict_if_full()
        assert "parked" not in executor._live
        assert "running" in executor._live
    finally:
        # Close both: the evicted one is already closed (close is idempotent),
        # but on an unexpected eviction outcome the survivor would leak its
        # background runner without this.
        await running.close()
        await parked.close()


async def test_eviction_falls_back_to_oldest_when_none_parked(monkeypatch):
    monkeypatch.setattr(executor_mod, "_MAX_LIVE", 2)
    executor = ClaudeCodeExecutor(make_backend("echo"))

    first = _running_session()
    second = _running_session()
    executor._live["first"] = first
    executor._live["second"] = second

    try:
        await executor._evict_if_full()
        assert "first" not in executor._live
        assert "second" in executor._live
    finally:
        await first.close()
        await second.close()


class _RecordingUpdater:
    """Captures the calls _pump makes, without a real event queue."""

    def __init__(self) -> None:
        self.did_fail = False
        self.did_complete = False
        self.status_lines: list[str] = []
        self.artifacts: list[tuple[str | None, str]] = []
        self.artifact_ids: list[str | None] = []

    def new_agent_message(self, parts, metadata=None):
        return parts

    async def add_artifact(self, parts, *, name=None, artifact_id=None, **_kwargs):
        self.artifacts.append((name, "".join(p.text for p in parts)))
        self.artifact_ids.append(artifact_id)

    async def update_status(self, _state, message=None):
        self.status_lines.append("".join(p.text for p in message or []))

    async def failed(self, message=None):
        self.did_fail = True

    async def complete(self, message=None):
        self.did_complete = True


async def _pump_events(*events) -> _RecordingUpdater:
    """Run _pump over a fixed event list and return what the updater saw."""

    async def _emit(session):
        for event in events:
            await session.emit(event)

    session = BackendSession()
    session.start(_emit)
    updater = _RecordingUpdater()
    try:
        await ClaudeCodeExecutor(make_backend("echo"))._pump(
            updater, "task-x", "ctx-x", session
        )
    finally:
        await session.close()
    return updater


async def test_pump_reports_a_tool_outcome_against_the_tool_name():
    updater = await _pump_events(
        ToolUse(name="Bash", tool_input={"command": "ls"}, tool_use_id="t1"),
        # No name on the result: it must resolve through the ToolUse above.
        ToolResult(tool_use_id="t1"),
    )
    assert updater.status_lines == ["$ ls", "✓ Bash"]


async def test_pump_reports_a_failure_with_its_first_line():
    updater = await _pump_events(
        ToolUse(name="Bash", tool_input={"command": "nope"}, tool_use_id="t1"),
        ToolResult(tool_use_id="t1", failed=True, output="command not found\ntrace\n"),
    )
    assert updater.status_lines[-1] == "✗ Bash: command not found"


async def test_pump_does_not_repeat_a_path_the_tool_name_already_carries():
    updater = await _pump_events(
        # ACP titles a tool call for a human, so the path is often in the name.
        ToolUse(
            name="Write calc.py", tool_input={"file_path": "calc.py"}, tool_use_id="t1"
        ),
        ToolUse(name="Write", tool_input={"file_path": "other.py"}, tool_use_id="t2"),
    )
    assert updater.status_lines == ["Write calc.py", "Write other.py"]


async def test_pump_falls_back_when_a_result_has_no_matching_tool_use():
    updater = await _pump_events(ToolResult(tool_use_id="unknown"))
    assert updater.status_lines == ["✓ tool"]


async def test_pump_replaces_the_plan_artifact_on_every_update():
    updater = await _pump_events(
        Plan(steps=[PlanStep(content="step one", status="in_progress")]),
        Plan(
            steps=[
                PlanStep(content="step one", status="completed"),
                PlanStep(content="step two", priority="high"),
            ]
        ),
    )
    names = [name for name, _ in updater.artifacts]
    assert names == ["plan", "plan"]
    assert updater.artifacts[0][1] == "- [>] step one\n"
    assert updater.artifacts[1][1] == "- [x] step one\n- [ ] (high) step two\n"
    # One artifact id across updates, so the caller replaces rather than stacks.
    assert len({a for a in updater.artifact_ids if a}) == 1


async def test_pump_streams_thinking_into_its_own_artifact():
    updater = await _pump_events(
        Thought(text="first "),
        Thought(text="second"),
        TextDelta(text="the answer"),
    )
    thinking = [(name, text) for name, text in updater.artifacts if name == "thinking"]

    assert thinking == [("thinking", "first "), ("thinking", "second")]
    # The answer must not carry the reasoning.
    answers = [text for name, text in updater.artifacts if name == "response"]
    assert "first" not in "".join(answers)


async def test_pump_renders_a_markdown_plan_verbatim():
    updater = await _pump_events(Plan(markdown="# rewrite the parser\n"))
    assert updater.artifacts == [("plan", "# rewrite the parser\n")]


async def test_pump_points_at_a_plan_the_agent_keeps_in_a_file():
    updater = await _pump_events(Plan(uri="file:///tmp/plan.md"))
    assert "file:///tmp/plan.md" in updater.artifacts[0][1]


async def test_pump_relays_a_notice_as_a_status_update():
    updater = await _pump_events(Notice(text="starting a fresh session"))
    assert updater.status_lines == ["starting a fresh session"]


async def test_pump_skips_an_empty_plan_when_none_was_ever_shown():
    updater = await _pump_events(Plan(steps=[]))
    assert updater.artifacts == []


async def test_pump_clears_a_plan_the_agent_abandoned():
    updater = await _pump_events(
        Plan(steps=[PlanStep(content="step one")]),
        Plan(steps=[]),
    )
    # The second update must replace the checklist, not leave the stale one up.
    assert [text for _, text in updater.artifacts] == ["- [ ] step one\n", ""]
    assert len({a for a in updater.artifact_ids if a}) == 1


async def test_pump_fails_an_evicted_session():
    # A session that finished (its runner returned, queuing the done sentinel)
    # but was flagged evicted: _pump must fail the task, not complete it with the
    # partial buffer.
    async def _noop(_session):
        return

    session = BackendSession()
    session.start(_noop)
    session.evicted = True

    executor = ClaudeCodeExecutor(make_backend("echo"))
    updater = _RecordingUpdater()
    try:
        await executor._pump(updater, "task-x", "ctx-x", session)
        assert updater.did_fail
        assert not updater.did_complete
    finally:
        await session.close()
