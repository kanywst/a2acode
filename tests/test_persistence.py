"""The optional persistent task store.

In-memory is the default; --task-db swaps in the A2A SDK's database stores so a
restart does not lose task history or push-notification registrations.
"""

from __future__ import annotations

import pytest
from a2a.server.tasks import (
    DatabasePushNotificationConfigStore,
    DatabaseTaskStore,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)

from a2acode.backends import make_backend
from a2acode.server import _stores, build_app


def test_no_dsn_keeps_everything_in_memory():
    engine, task_store, push_store = _stores(None)

    assert engine is None
    assert isinstance(task_store, InMemoryTaskStore)
    assert isinstance(push_store, InMemoryPushNotificationConfigStore)


def test_a_dsn_selects_the_database_stores(tmp_path):
    engine, task_store, push_store = _stores(
        f"sqlite+aiosqlite:///{tmp_path / 'tasks.db'}"
    )

    assert engine is not None
    assert isinstance(task_store, DatabaseTaskStore)
    assert isinstance(push_store, DatabasePushNotificationConfigStore)


def test_an_unusable_dsn_fails_at_build_time_not_on_the_first_request():
    from sqlalchemy.exc import ArgumentError, NoSuchModuleError

    with pytest.raises((ArgumentError, NoSuchModuleError)):
        _stores("nonsense://not-a-driver/x")


async def test_the_lifespan_creates_the_schema_and_disposes_the_engine(tmp_path):
    db = tmp_path / "tasks.db"
    app = build_app(
        make_backend("echo"),
        url="http://localhost:9100/",
        task_db=f"sqlite+aiosqlite:///{db}",
    )

    async with app.router.lifespan_context(app):
        pass

    # initialize() ran, so the file exists with the SDK's tables in it.
    assert db.exists()


async def test_a_persisted_task_survives_a_new_store_over_the_same_database(tmp_path):
    from a2a.server.context import ServerCallContext
    from a2a.types import Task, TaskState, TaskStatus

    dsn = f"sqlite+aiosqlite:///{tmp_path / 'tasks.db'}"
    context = ServerCallContext()
    task = Task(
        id="task-1",
        context_id="ctx-1",
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
    )

    engine, store, _ = _stores(dsn)
    await store.initialize()
    await store.save(task, context)
    await engine.dispose()

    # A fresh process would build its stores from scratch; the task is still on
    # disk, which is the whole point of the flag.
    engine, reopened, _ = _stores(dsn)
    await reopened.initialize()
    try:
        assert (await reopened.get("task-1", context)) is not None
    finally:
        await engine.dispose()
