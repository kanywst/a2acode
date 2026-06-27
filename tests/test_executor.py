"""Executor lifecycle: eviction and session-continuity bookkeeping."""

from __future__ import annotations

from a2acode import executor as executor_mod
from a2acode.backends import BackendSession, RunRequest, make_backend
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
    """Captures the terminal call _pump makes, without a real event queue."""

    def __init__(self) -> None:
        self.did_fail = False
        self.did_complete = False

    def new_agent_message(self, parts, metadata=None):
        return parts

    async def add_artifact(self, *_args, **_kwargs):  # pragma: no cover - unused here
        pass

    async def update_status(self, *_args, **_kwargs):  # pragma: no cover - unused
        pass

    async def failed(self, message=None):
        self.did_fail = True

    async def complete(self, message=None):
        self.did_complete = True


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
