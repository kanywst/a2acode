"""Executor lifecycle: session-continuity bookkeeping."""

from __future__ import annotations

from a2claude import executor as executor_mod
from a2claude.backends import make_backend
from a2claude.executor import ClaudeCodeExecutor


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
