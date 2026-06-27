"""Fast offline checks. The full HTTP round trip is verified via the CLI."""

from __future__ import annotations

import pytest

from a2acode.backends import (
    BackendSession,
    PermissionDecision,
    PermissionRequest,
    Result,
    RunRequest,
    TextDelta,
    ToolUse,
    make_backend,
)
from a2acode.backends.diff import file_changes
from a2acode.server import build_app


async def _drive(backend, request):
    """Run a backend to completion, auto-allowing any permission request."""
    session = BackendSession()
    session.start(lambda s: backend.drive(s, request))
    events = []
    try:
        while not session.done:
            async for event in session.drain():
                events.append(event)
                if isinstance(event, PermissionRequest):
                    session.resolve(
                        PermissionDecision(request_id=event.request_id, allow=True)
                    )
    finally:
        await session.close()
    return events


async def test_echo_emits_tool_text_and_result():
    events = await _drive(make_backend("echo"), RunRequest(prompt="hello world"))

    assert any(isinstance(e, ToolUse) for e in events)
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text.strip() == "hello world"
    assert any(isinstance(e, Result) for e in events)


async def test_echo_permission_round_trip():
    session = BackendSession()
    session.start(
        lambda s: make_backend("echo").drive(s, RunRequest(prompt="sudo reboot"))
    )

    first = [e async for e in session.drain()]
    request = next(e for e in first if isinstance(e, PermissionRequest))
    assert request.tool_name == "Bash"
    assert not session.done
    assert session.is_parked

    session.resolve(PermissionDecision(request_id=request.request_id, allow=True))
    rest = [e async for e in session.drain()]
    text = "".join(e.text for e in rest if isinstance(e, TextDelta))

    assert session.done
    assert not session.is_parked
    assert text.strip() == "sudo reboot"
    await session.close()


async def test_echo_permission_denied():
    session = BackendSession()
    session.start(
        lambda s: make_backend("echo").drive(s, RunRequest(prompt="sudo rm -rf /"))
    )

    first = [e async for e in session.drain()]
    request = next(e for e in first if isinstance(e, PermissionRequest))

    session.resolve(PermissionDecision(request_id=request.request_id, allow=False))
    rest = [e async for e in session.drain()]
    text = "".join(e.text for e in rest if isinstance(e, TextDelta))

    assert "denied" in text
    await session.close()


def test_file_changes_from_edit():
    changes = file_changes(
        "Edit",
        {"file_path": "app.py", "old_string": "x = 1", "new_string": "x = 2"},
    )
    assert len(changes) == 1
    assert changes[0].path == "app.py"
    assert "-x = 1" in changes[0].diff
    assert "+x = 2" in changes[0].diff


def test_file_changes_multiedit():
    changes = file_changes(
        "MultiEdit",
        {
            "file_path": "a.py",
            "edits": [
                {"old_string": "a", "new_string": "b"},
                {"old_string": "c", "new_string": "d"},
            ],
        },
    )
    assert len(changes) == 1
    assert "-a" in changes[0].diff
    assert "-c" in changes[0].diff


def test_file_changes_multiedit_malformed_does_not_raise():
    assert file_changes("MultiEdit", {"file_path": "a.py", "edits": "oops"}) == []
    assert file_changes("MultiEdit", {"file_path": "a.py", "edits": ["x", 1]}) == []


def test_file_changes_handles_null_fields():
    changes = file_changes(
        "Edit", {"file_path": "a.py", "old_string": None, "new_string": "x = 1"}
    )
    assert changes
    assert "None" not in changes[0].diff
    assert "+x = 1" in changes[0].diff


def test_file_changes_ignores_non_edit_tools():
    assert file_changes("Bash", {"command": "ls"}) == []


async def test_resolve_unknown_request_raises():
    session = BackendSession()
    with pytest.raises(ValueError):
        session.resolve(PermissionDecision(request_id="missing", allow=True))


def test_build_app_returns_asgi_app():
    app = build_app(make_backend("echo"), url="http://localhost:9100/")
    assert app.routes


def test_card_advertises_jsonrpc_and_rest():
    from a2a.utils.constants import TransportProtocol

    from a2acode.card import build_card

    card = build_card("http://localhost:9100/")
    bindings = {iface.protocol_binding for iface in card.supported_interfaces}
    assert TransportProtocol.JSONRPC in bindings
    assert TransportProtocol.HTTP_JSON in bindings


def test_card_version_tracks_package_version():
    from importlib.metadata import PackageNotFoundError, version

    from a2acode.card import build_card

    try:
        expected = version("a2acode")
    except PackageNotFoundError:  # source tree without an install: same fallback
        expected = "0.0.0"
    assert build_card("http://localhost:9100/").version == expected
