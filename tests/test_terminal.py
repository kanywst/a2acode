"""Terminals the ACP agent runs through us.

The capability is only safe to advertise because every terminal goes through the
same caller approval as any tool, starts inside the workspace, and is reaped
with the turn. These tests hold that line.
"""

from __future__ import annotations

import asyncio
import sys

import pytest
from acp import schema as s
from acp.exceptions import RequestError

from a2acode.backends import terminal as terminal_mod
from a2acode.backends.acp import _BridgeClient
from a2acode.backends.base import PermissionDecision
from a2acode.backends.terminal import spawn


class _FakeSession:
    def __init__(self, *, allow: bool = True) -> None:
        self._allow = allow
        self.asked: list[tuple[str, dict, str]] = []

    async def request_permission(self, name, tool_input, description, options=()):
        self.asked.append((name, tool_input, description))
        return PermissionDecision(
            request_id="x", allow=self._allow, message="" if self._allow else "nope"
        )


def _client(session, tmp_path) -> _BridgeClient:
    return _BridgeClient(session, str(tmp_path))  # type: ignore[arg-type]


async def _run(client, command, *args, **kwargs):
    resp = await client.create_terminal("sess", command, list(args), **kwargs)
    await client.wait_for_terminal_exit("sess", resp.terminal_id)
    return resp.terminal_id


@pytest.mark.asyncio
async def test_a_terminal_needs_the_callers_approval(tmp_path):
    session = _FakeSession()
    client = _client(session, tmp_path)

    terminal_id = await _run(client, sys.executable, "-c", "print('hi')")
    out = await client.terminal_output("sess", terminal_id)

    assert "hi" in out.output
    # It went through the same round trip as any tool, with the command shown.
    assert len(session.asked) == 1
    name, tool_input, description = session.asked[0]
    assert name == "Terminal"
    assert tool_input["command"] == sys.executable
    # The caller is shown the actual command line it is approving.
    assert description.startswith(sys.executable)
    assert "hi" in description


@pytest.mark.asyncio
async def test_a_denied_terminal_never_runs(tmp_path):
    session = _FakeSession(allow=False)
    client = _client(session, tmp_path)
    marker = tmp_path / "ran"

    with pytest.raises(RequestError):
        await client.create_terminal(
            "sess",
            sys.executable,
            ["-c", f"open({str(marker)!r}, 'w').write('x')"],
        )

    assert not marker.exists()
    assert client._terminals == {}


@pytest.mark.asyncio
async def test_an_unbound_client_refuses_to_run_anything(tmp_path):
    # Between turns there is no caller to answer, so nothing may execute.
    client = _BridgeClient(None, str(tmp_path))  # type: ignore[abstract]

    with pytest.raises(RequestError):
        await client.create_terminal("sess", sys.executable, ["-c", "pass"])


@pytest.mark.asyncio
async def test_exit_code_and_running_status_are_reported(tmp_path):
    client = _client(_FakeSession(), tmp_path)

    terminal_id = await _run(client, sys.executable, "-c", "raise SystemExit(3)")
    out = await client.terminal_output("sess", terminal_id)

    assert out.exit_status is not None
    assert out.exit_status.exit_code == 3


@pytest.mark.asyncio
async def test_output_is_still_running_before_exit(tmp_path):
    client = _client(_FakeSession(), tmp_path)
    resp = await client.create_terminal(
        "sess", sys.executable, ["-c", "import time; time.sleep(30)"]
    )

    out = await client.terminal_output("sess", resp.terminal_id)
    assert out.exit_status is None

    await client.kill_terminal("sess", resp.terminal_id)
    await client.release_terminal("sess", resp.terminal_id)


@pytest.mark.asyncio
async def test_a_killed_terminal_reports_its_signal(tmp_path):
    client = _client(_FakeSession(), tmp_path)
    resp = await client.create_terminal(
        "sess", sys.executable, ["-c", "import time; time.sleep(30)"]
    )

    await client.kill_terminal("sess", resp.terminal_id)
    out = await client.terminal_output("sess", resp.terminal_id)

    assert out.exit_status is not None
    assert out.exit_status.signal == "SIGKILL"


@pytest.mark.asyncio
async def test_a_terminal_cannot_start_outside_the_workspace(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    client = _client(_FakeSession(), workspace)

    with pytest.raises(PermissionError):
        await client.create_terminal(
            "sess", sys.executable, ["-c", "pass"], cwd=str(tmp_path)
        )


@pytest.mark.asyncio
async def test_releasing_a_terminal_drops_it(tmp_path):
    client = _client(_FakeSession(), tmp_path)
    terminal_id = await _run(client, sys.executable, "-c", "pass")

    await client.release_terminal("sess", terminal_id)

    assert client._terminals == {}
    with pytest.raises(ValueError):
        await client.terminal_output("sess", terminal_id)


@pytest.mark.asyncio
async def test_unbinding_reaps_terminals_the_agent_left_running(tmp_path):
    client = _client(_FakeSession(), tmp_path)
    resp = await client.create_terminal(
        "sess", sys.executable, ["-c", "import time; time.sleep(30)"]
    )
    process = client._terminals[resp.terminal_id].process

    await client.unbind()

    assert client._terminals == {}
    assert process.returncode is not None


@pytest.mark.asyncio
async def test_too_many_terminals_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("a2acode.backends.acp._MAX_TERMINALS", 1)
    client = _client(_FakeSession(), tmp_path)
    await client.create_terminal("sess", sys.executable, ["-c", "import time;"])

    with pytest.raises(RuntimeError):
        await client.create_terminal("sess", sys.executable, ["-c", "pass"])

    await client.unbind()


@pytest.mark.asyncio
async def test_output_keeps_the_tail_when_it_overflows(tmp_path):
    # The end of a failing build says more than its opening banner.
    term = await spawn(
        sys.executable,
        ["-c", "print('a' * 100); print('TAIL')"],
        cwd=tmp_path,
        env={},
        limit=16,
    )
    await term.wait()

    assert term.truncated
    # Trimming is by byte, so one long line cannot carry the tail away with it.
    assert "TAIL" in term.output
    assert len(term.output) == 16


@pytest.mark.asyncio
async def test_the_agents_output_limit_cannot_exceed_the_hard_ceiling(tmp_path):
    monkey = terminal_mod.MAX_OUTPUT_LIMIT
    client = _client(_FakeSession(), tmp_path)

    resp = await client.create_terminal(
        "sess", sys.executable, ["-c", "pass"], output_byte_limit=monkey * 10
    )

    assert client._terminals[resp.terminal_id].limit == monkey
    await client.unbind()


@pytest.mark.asyncio
async def test_the_caller_sees_the_environment_it_is_approving(tmp_path):
    # An agent supplies its own variables, and PATH or LD_PRELOAD decide what a
    # plausible-looking command actually runs. Approving "make test" without
    # them would be approving the wrong thing.
    session = _FakeSession()
    client = _client(session, tmp_path)
    empty_bin = tmp_path / "bin"
    empty_bin.mkdir()
    preload = tmp_path / "evil.so"

    # Supplied PATH-first so the assertion below fails if the sort is dropped.
    # The spawn then cannot find "make", which is the point: PATH really is
    # replaced by what the agent sent, so the caller had to be shown it.
    with pytest.raises(FileNotFoundError):
        await client.create_terminal(
            "sess",
            "make",
            ["test"],
            env=[
                s.EnvVariable(name="PATH", value=str(empty_bin)),
                s.EnvVariable(name="LD_PRELOAD", value=str(preload)),
            ],
        )

    _, tool_input, description = session.asked[0]
    assert tool_input["env"] == {"PATH": str(empty_bin), "LD_PRELOAD": str(preload)}
    assert description == f"LD_PRELOAD={preload} PATH={empty_bin} make test"
    await client.unbind()


@pytest.mark.asyncio
async def test_an_environment_value_needing_quoting_is_shown_unambiguously(tmp_path):
    session = _FakeSession()
    client = _client(session, tmp_path)

    await client.create_terminal(
        "sess",
        "sh",
        ["-c", "true"],
        env=[s.EnvVariable(name="X", value="a b; rm -rf /")],
    )

    assert session.asked[0][2] == "X='a b; rm -rf /' sh -c true"
    await client.unbind()


@pytest.mark.asyncio
async def test_an_environment_name_that_is_not_an_identifier_cannot_split(tmp_path):
    # The name is agent-controlled too. One carrying a space must not render as
    # two words the caller reads as separate arguments.
    session = _FakeSession()
    client = _client(session, tmp_path)

    await client.create_terminal(
        "sess",
        "sh",
        ["-c", "true"],
        env=[s.EnvVariable(name="A B", value="c")],
    )

    assert session.asked[0][2] == "'A B=c' sh -c true"
    await client.unbind()


@pytest.mark.asyncio
async def test_server_credentials_do_not_reach_an_agent_chosen_command(
    tmp_path, monkeypatch
):
    # Approving that a command runs is not approving that it can read every
    # secret the server was launched with.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    client = _client(_FakeSession(), tmp_path)

    resp = await client.create_terminal(
        "sess",
        sys.executable,
        ["-c", "import os; print(sorted(os.environ))"],
    )
    await client.wait_for_terminal_exit("sess", resp.terminal_id)
    out = await client.terminal_output("sess", resp.terminal_id)

    assert "ANTHROPIC_API_KEY" not in out.output
    assert "PATH" in out.output


@pytest.mark.asyncio
async def test_env_variables_reach_the_process(tmp_path):
    client = _client(_FakeSession(), tmp_path)

    resp = await client.create_terminal(
        "sess",
        sys.executable,
        ["-c", "import os; print(os.environ['A2ACODE_TEST'])"],
        env=[s.EnvVariable(name="A2ACODE_TEST", value="present")],
    )
    await client.wait_for_terminal_exit("sess", resp.terminal_id)
    out = await client.terminal_output("sess", resp.terminal_id)

    assert "present" in out.output


@pytest.mark.asyncio
async def test_wait_returns_immediately_for_a_finished_process(tmp_path):
    client = _client(_FakeSession(), tmp_path)
    resp = await client.create_terminal("sess", sys.executable, ["-c", "pass"])

    first = await client.wait_for_terminal_exit("sess", resp.terminal_id)
    second = await asyncio.wait_for(
        client.wait_for_terminal_exit("sess", resp.terminal_id), 5
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
