"""Unit tests for the ACP backend's pure mapping and permission selection.

These exercise the protocol translation without launching an agent subprocess:
``events_from_update`` and ``select_option`` are pure, and ``_BridgeClient``'s
permission round trip is driven against a fake session.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, suppress

import pytest
from acp import (
    image_block,
    plan_entry,
    text_block,
    tool_content,
    tool_diff_content,
    update_plan,
)
from acp import schema as s
from acp.task import InMemoryMessageQueue

from a2acode.backends import acp as acp_mod
from a2acode.backends.acp import (
    ACPBackend,
    _Agent,
    _BridgeClient,
    events_from_update,
    select_option,
)
from a2acode.backends.base import (
    FileChange,
    Notice,
    PermissionDecision,
    Plan,
    RunRequest,
    TextDelta,
    Thought,
    ToolResult,
    ToolUse,
)
from a2acode.backends.session import BackendSession


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


def test_completed_tool_call_yields_tool_result_with_output():
    update = s.ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="t1",
        title="Run ls",
        status="completed",
        content=[tool_content(text_block("a.py\nb.py\n"))],
    )
    events = list(events_from_update(update))

    assert len(events) == 1
    result = events[0]
    assert isinstance(result, ToolResult)
    assert result.tool_use_id == "t1"
    assert result.name == "Run ls"
    assert not result.failed
    assert result.output == "a.py\nb.py\n"


def test_failed_tool_call_is_flagged():
    update = s.ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="t1",
        status="failed",
        content=[tool_content(text_block("command not found"))],
    )
    events = list(events_from_update(update))

    assert len(events) == 1
    assert isinstance(events[0], ToolResult)
    assert events[0].failed
    # No title on the update: the consumer falls back to the ToolUse's name.
    assert events[0].name == ""


def test_non_terminal_tool_status_yields_no_result():
    for status in ("pending", "in_progress"):
        update = s.ToolCallProgress(
            session_update="tool_call_update", tool_call_id="t1", status=status
        )
        assert list(events_from_update(update)) == []


def test_tool_call_start_with_terminal_status_yields_result():
    update = s.ToolCallStart(
        session_update="tool_call",
        tool_call_id="t1",
        title="Read a.py",
        status="completed",
    )
    events = list(events_from_update(update))

    assert [type(e) for e in events] == [ToolUse, ToolResult]


def test_tool_output_is_capped():
    update = s.ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="t1",
        status="completed",
        content=[tool_content(text_block("x" * 5000))],
    )
    output = next(iter(events_from_update(update))).output
    assert output.endswith(" …")
    assert len(output) == 2002


def test_non_text_tool_content_is_skipped():
    update = s.ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="t1",
        status="completed",
        content=[tool_content(image_block("ZGF0YQ==", "image/png"))],
    )
    assert next(iter(events_from_update(update))).output == ""


def test_plan_update_maps_to_plan_steps():
    update = update_plan(
        [
            plan_entry("read the code", status="completed", priority="high"),
            plan_entry("write the fix", status="in_progress"),
        ]
    )
    events = list(events_from_update(update))

    assert len(events) == 1
    plan = events[0]
    assert isinstance(plan, Plan)
    assert [(step.content, step.status, step.priority) for step in plan.steps] == [
        ("read the code", "completed", "high"),
        ("write the fix", "in_progress", "medium"),
    ]


def test_plan_content_update_with_items_maps_to_plan_steps():
    update = s.AgentPlanContentUpdate(
        session_update="plan_update",
        plan=s.PlanUpdateItems(
            type="items", plan_id="p1", entries=[plan_entry("do the thing")]
        ),
    )
    events = list(events_from_update(update))

    assert len(events) == 1
    assert isinstance(events[0], Plan)
    assert events[0].steps[0].content == "do the thing"


def test_markdown_plan_is_carried_as_prose_not_flattened_into_steps():
    update = s.AgentPlanContentUpdate(
        session_update="plan_update",
        plan=s.PlanUpdateMarkdown(type="markdown", plan_id="p1", content="# do it"),
    )
    events = list(events_from_update(update))

    assert len(events) == 1
    assert events[0] == Plan(markdown="# do it")
    # No per-entry status exists to invent.
    assert events[0].steps == []


def test_file_plan_is_carried_as_a_pointer():
    update = s.AgentPlanContentUpdate(
        session_update="plan_update",
        plan=s.PlanUpdateFile(type="file", plan_id="p1", uri="file:///tmp/plan.md"),
    )
    assert list(events_from_update(update)) == [Plan(uri="file:///tmp/plan.md")]


def test_thought_chunk_maps_to_a_thought():
    update = s.AgentThoughtChunk(
        session_update="agent_thought_chunk", content=text_block("hmm")
    )
    assert list(events_from_update(update)) == [Thought(text="hmm")]


def test_empty_thought_chunk_yields_nothing():
    update = s.AgentThoughtChunk(
        session_update="agent_thought_chunk", content=text_block("")
    )
    assert list(events_from_update(update)) == []


def test_mode_switch_becomes_a_notice():
    update = s.CurrentModeUpdate(
        session_update="current_mode_update", current_mode_id="plan"
    )
    events = list(events_from_update(update))

    assert len(events) == 1
    assert isinstance(events[0], Notice)
    assert "plan" in events[0].text


def test_session_title_becomes_a_notice_and_an_untitled_one_does_not():
    titled = s.SessionInfoUpdate(session_update="session_info_update", title="Fix auth")
    assert "Fix auth" in next(iter(events_from_update(titled))).text

    untitled = s.SessionInfoUpdate(session_update="session_info_update")
    assert list(events_from_update(untitled)) == []


def test_usage_update_yields_nothing():
    update = s.UsageUpdate(session_update="usage_update", used=10, size=100)
    assert list(events_from_update(update)) == []


def test_non_mapping_raw_input_becomes_empty_dict():
    update = s.ToolCallStart(
        session_update="tool_call", tool_call_id="t1", title="x", raw_input="not-a-dict"
    )
    events = list(events_from_update(update))
    assert events[0].tool_input == {}


class _Emitting:
    """A session that only collects what the client emits."""

    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, event) -> None:
        self.events.append(event)


async def _feed(*updates, flush: bool = True) -> list:
    """Push updates through a bound client, as the connection's reader would."""
    session = _Emitting()
    client = _BridgeClient(session)  # type: ignore[arg-type]
    for update in updates:
        await client.session_update("sess", update)
    if flush:
        await client.flush_tool_calls()
    return session.events


@pytest.mark.asyncio
async def test_arguments_that_arrive_after_a_call_opens_reach_its_tool_use():
    # The sequence a real agent sends: the call is announced before its
    # arguments are parsed, and they land on a later update.
    events = await _feed(
        s.ToolCallStart(
            session_update="tool_call",
            tool_call_id="t1",
            title="Read File",
            kind="read",
            status="pending",
        ),
        s.ToolCallProgress(
            session_update="tool_call_update",
            tool_call_id="t1",
            title="Read app.py",
            raw_input={"file_path": "/w/app.py"},
        ),
        s.ToolCallProgress(session_update="tool_call_update", tool_call_id="t1"),
        s.ToolCallProgress(
            session_update="tool_call_update", tool_call_id="t1", status="completed"
        ),
    )

    uses = [e for e in events if isinstance(e, ToolUse)]
    assert len(uses) == 1
    assert uses[0].tool_input == {"file_path": "/w/app.py"}
    assert uses[0].name == "Read app.py"
    # The terminal update omits the title, meaning unchanged, so the outcome
    # resolves to the name the call ended up with.
    result = next(e for e in events if isinstance(e, ToolResult))
    assert result.name == "Read app.py"


@pytest.mark.asyncio
async def test_a_call_that_opens_with_its_arguments_is_announced_at_once():
    events = await _feed(
        s.ToolCallStart(
            session_update="tool_call",
            tool_call_id="t1",
            title="Write calc.py",
            kind="edit",
            raw_input={"file_path": "calc.py"},
        ),
        flush=False,
    )

    assert [type(e) for e in events] == [ToolUse]
    assert events[0].tool_input == {"file_path": "calc.py"}


@pytest.mark.asyncio
async def test_a_terminal_status_announces_a_call_that_never_had_arguments():
    events = await _feed(
        s.ToolCallStart(
            session_update="tool_call",
            tool_call_id="t2",
            title="Run tests",
            kind="execute",
        ),
        s.ToolCallProgress(
            session_update="tool_call_update",
            tool_call_id="t2",
            status="failed",
            content=[tool_content(text_block("2 tests failed"))],
        ),
    )

    assert [type(e) for e in events] == [ToolUse, ToolResult]
    assert events[0].name == "Run tests"


@pytest.mark.asyncio
async def test_a_call_the_agent_never_returns_to_is_announced_at_end_of_turn():
    session = _Emitting()
    client = _BridgeClient(session)  # type: ignore[arg-type]
    await client.session_update(
        "sess",
        s.ToolCallStart(
            session_update="tool_call", tool_call_id="t3", title="Fetch", kind="fetch"
        ),
    )
    assert session.events == []

    await client.flush_tool_calls()
    assert [type(e) for e in session.events] == [ToolUse]
    assert session.events[0].name == "Fetch"

    # Idempotent, so the flush a cancelled or crashed turn runs cannot repeat a
    # call the normal path already announced.
    await client.flush_tool_calls()
    assert len(session.events) == 1


@pytest.mark.asyncio
async def test_a_diff_is_not_replayed_by_the_updates_that_follow_it():
    events = await _feed(
        s.ToolCallStart(
            session_update="tool_call",
            tool_call_id="t1",
            title="Write a.py",
            raw_input={"file_path": "a.py"},
            content=[tool_diff_content(path="a.py", new_text="y\n", old_text="x\n")],
        ),
        s.ToolCallProgress(
            session_update="tool_call_update",
            tool_call_id="t1",
            status="completed",
            content=[tool_content(text_block("written"))],
        ),
    )

    assert len([e for e in events if isinstance(e, FileChange)]) == 1


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


class _FakeConn:
    """Records the session-lifecycle calls the backend makes."""

    def __init__(self) -> None:
        self.loaded: str | None = None
        self.opened = False

    async def load_session(self, *, cwd, session_id, mcp_servers):
        self.loaded = session_id

    async def new_session(self, *, cwd, mcp_servers):
        self.opened = True
        return s.NewSessionResponse(session_id="fresh")


class _FakeProcess:
    returncode: int | None = None


def _agent(conn=None, *, load_session=False, session_id=None) -> _Agent:
    return _Agent(
        stack=AsyncExitStack(),
        conn=conn or _FakeConn(),
        process=_FakeProcess(),
        client=_BridgeClient(),
        queue=InMemoryMessageQueue(),
        capabilities=s.AgentCapabilities(load_session=load_session),
        session_id=session_id,
    )


async def _open(agent, resume) -> tuple[str, list]:
    """Open a session on a fake agent, collecting the events it emits."""
    events: list = []

    class _Collector:
        async def emit(self, event):
            events.append(event)

    session_id = await ACPBackend(agent="gemini")._open_session(
        agent, RunRequest(prompt="hi", resume=resume), _Collector()
    )
    return session_id, events


@pytest.mark.asyncio
async def test_resume_loads_the_session_when_the_agent_supports_it():
    agent = _agent(load_session=True)
    session_id, events = await _open(agent, "sess-1")

    assert session_id == "sess-1"
    assert agent.conn.loaded == "sess-1"
    assert not agent.conn.opened
    assert events == []


@pytest.mark.asyncio
async def test_resume_without_load_support_notices_the_lost_continuity():
    agent = _agent(load_session=False)
    session_id, events = await _open(agent, "sess-1")

    assert session_id == "fresh"
    assert agent.conn.opened
    assert len(events) == 1
    assert isinstance(events[0], Notice)
    assert "gemini" in events[0].text
    assert "session/load" in events[0].text


@pytest.mark.asyncio
async def test_a_first_turn_opens_a_session_without_a_notice():
    agent = _agent(load_session=False)
    session_id, events = await _open(agent, None)

    assert session_id == "fresh"
    assert events == []


@pytest.mark.asyncio
async def test_a_process_that_already_holds_the_session_skips_the_handshake():
    # The point of pooling: a follow-up turn costs neither a session/load nor a
    # new session, because this process is already in that conversation.
    agent = _agent(load_session=True, session_id="sess-1")
    session_id, events = await _open(agent, "sess-1")

    assert session_id == "sess-1"
    assert agent.conn.loaded is None
    assert not agent.conn.opened
    assert events == []


def _pooling_backend(monkeypatch) -> tuple[ACPBackend, list[_Agent]]:
    """A backend whose _spawn hands out fake agents instead of subprocesses."""
    backend = ACPBackend(agent="gemini")
    spawned: list[_Agent] = []

    async def _spawn(self):
        agent = _agent()
        spawned.append(agent)
        return agent

    monkeypatch.setattr(ACPBackend, "_spawn", _spawn)
    return backend, spawned


@pytest.mark.asyncio
async def test_a_context_reuses_its_agent_and_a_new_context_gets_its_own(monkeypatch):
    backend, spawned = _pooling_backend(monkeypatch)

    first = await backend._acquire("ctx-a")
    again = await backend._acquire("ctx-a")
    other = await backend._acquire("ctx-b")

    assert first is again
    assert other is not first
    assert len(spawned) == 2
    await backend.aclose()


@pytest.mark.asyncio
async def test_a_dead_agent_is_replaced_rather_than_handed_out(monkeypatch):
    backend, _ = _pooling_backend(monkeypatch)

    first = await backend._acquire("ctx-a")
    first.process.returncode = 1
    replacement = await backend._acquire("ctx-a")

    assert replacement is not first
    await backend.aclose()


@pytest.mark.asyncio
async def test_a_broken_agent_is_replaced_rather_than_handed_out(monkeypatch):
    backend, _ = _pooling_backend(monkeypatch)

    first = await backend._acquire("ctx-a")
    first.broken = True

    assert await backend._acquire("ctx-a") is not first
    await backend.aclose()


@pytest.mark.asyncio
async def test_a_turn_without_a_context_is_not_pooled(monkeypatch):
    backend, _ = _pooling_backend(monkeypatch)

    agent = await backend._acquire(None)

    assert not agent.pooled
    assert backend._agents == {}


async def _turn(backend, context_id) -> _Agent:
    """Acquire an agent and release it, as a completed turn would."""
    agent = await backend._acquire(context_id)
    agent.claims -= 1
    return agent


@pytest.mark.asyncio
async def test_the_pool_evicts_the_least_recently_used_idle_agent(monkeypatch):
    monkeypatch.setattr(acp_mod, "_MAX_AGENTS", 2)
    backend, _ = _pooling_backend(monkeypatch)

    await _turn(backend, "ctx-a")
    await _turn(backend, "ctx-b")
    # Touch "a" so "b" becomes least recently used, then overflow.
    await _turn(backend, "ctx-a")
    await _turn(backend, "ctx-c")

    assert set(backend._agents) == {"ctx-a", "ctx-c"}
    await backend.aclose()


@pytest.mark.asyncio
async def test_a_busy_agent_is_never_evicted(monkeypatch):
    monkeypatch.setattr(acp_mod, "_MAX_AGENTS", 1)
    backend, _ = _pooling_backend(monkeypatch)

    busy = await backend._acquire("ctx-a")
    async with busy.lock:
        # Rather than kill a running turn's agent, the pool overshoots.
        await _turn(backend, "ctx-b")
        assert set(backend._agents) == {"ctx-a", "ctx-b"}
    await backend.aclose()


@pytest.mark.asyncio
async def test_an_agent_handed_out_is_not_evicted_before_its_turn_starts(monkeypatch):
    # _acquire drops the pool lock before drive takes the agent's own lock. In
    # that gap the agent is idle but spoken for, and evicting it would close the
    # connection the turn is about to prompt on.
    monkeypatch.setattr(acp_mod, "_MAX_AGENTS", 1)
    backend, _ = _pooling_backend(monkeypatch)

    claimed = await backend._acquire("ctx-a")
    assert not claimed.lock.locked()

    await backend._acquire("ctx-b")

    assert set(backend._agents) == {"ctx-a", "ctx-b"}
    await backend.aclose()


@pytest.mark.asyncio
async def test_an_agent_becomes_evictable_once_its_turn_releases_it(monkeypatch):
    monkeypatch.setattr(acp_mod, "_MAX_AGENTS", 1)
    backend, _ = _pooling_backend(monkeypatch)

    async def _noop(self, agent, session, request):
        return

    monkeypatch.setattr(ACPBackend, "_run_turn", _noop)
    session = BackendSession()
    await backend.drive(session, RunRequest(prompt="a", context_id="ctx-a"))
    await session.close()

    await _turn(backend, "ctx-b")

    assert set(backend._agents) == {"ctx-b"}
    await backend.aclose()


@pytest.mark.asyncio
async def test_aclose_empties_the_pool(monkeypatch):
    backend, _ = _pooling_backend(monkeypatch)
    await backend._acquire("ctx-a")
    await backend._acquire("ctx-b")

    await backend.aclose()

    assert backend._agents == {}


@pytest.mark.asyncio
async def test_a_second_turn_in_one_conversation_says_it_is_waiting(monkeypatch):
    backend, _ = _pooling_backend(monkeypatch)
    agent = await backend._acquire("ctx-a")
    started = asyncio.Event()

    async def _park(self, agent, session, request):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(ACPBackend, "_run_turn", _park)

    first = BackendSession()
    first.start(lambda s: backend.drive(s, RunRequest(prompt="a", context_id="ctx-a")))
    await started.wait()

    second = BackendSession()
    second.start(lambda s: backend.drive(s, RunRequest(prompt="b", context_id="ctx-a")))
    # Only the notice: the turn itself is still blocked behind the first one, so
    # draining to completion would wait forever.
    first_event = await anext(aiter(second.drain()))

    assert isinstance(first_event, Notice)
    assert "waiting" in first_event.text
    assert agent.lock.locked()
    await first.close()
    await second.close()


@pytest.mark.asyncio
async def test_a_turn_cancelled_while_queued_releases_its_claim(monkeypatch):
    backend, _ = _pooling_backend(monkeypatch)
    agent = await backend._acquire("ctx-a")
    agent.claims -= 1
    started = asyncio.Event()

    async def _park(self, agent, session, request):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(ACPBackend, "_run_turn", _park)

    first = BackendSession()
    first.start(lambda s: backend.drive(s, RunRequest(prompt="a", context_id="ctx-a")))
    await started.wait()

    queued = asyncio.ensure_future(
        backend.drive(BackendSession(), RunRequest(prompt="b", context_id="ctx-a"))
    )
    await asyncio.sleep(0)
    assert agent.claims == 2
    queued.cancel()
    with suppress(asyncio.CancelledError):
        await queued

    # Only the running turn still holds it; a claim stuck at 2 would pin the
    # agent in the pool forever.
    assert agent.claims == 1
    await first.close()


@pytest.mark.asyncio
async def test_a_failed_turn_retires_its_agent(monkeypatch):
    backend, _ = _pooling_backend(monkeypatch)

    async def _boom(self, agent, session, request):
        raise RuntimeError("the agent fell over")

    monkeypatch.setattr(ACPBackend, "_run_turn", _boom)

    with pytest.raises(RuntimeError):
        await backend.drive(BackendSession(), RunRequest(prompt="hi", context_id="c"))

    # It stays in the map, but as unusable, so the next turn replaces it rather
    # than resuming a conversation on a connection of unknown state.
    assert not backend._agents["c"].usable


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
