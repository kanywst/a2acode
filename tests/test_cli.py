"""CLI argument validation that must fail before the server starts."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from a2claude.cli import app

runner = CliRunner()


def _plain(output: str) -> str:
    """Reduce Typer's Rich-rendered error to bare tokens.

    The BadParameter message is drawn in a bordered panel whose width follows the
    terminal, so it wraps differently across environments (CI at 80 columns
    splits ``--permission-mode`` across the box border). Strip ANSI codes and
    keep only word characters and hyphens so the message text can be matched
    regardless of where the panel wrapped it.
    """
    no_ansi = re.sub(r"\x1b\[[0-9;]*m", "", output)
    return re.sub(r"[^a-zA-Z0-9-]", "", no_ansi)


def test_serve_rejects_invalid_permission_mode():
    # Fails during validation, before uvicorn.run, so the runner does not block.
    result = runner.invoke(
        app, ["serve", "--backend", "echo", "--permission-mode", "bogus"]
    )
    assert result.exit_code != 0
    plain = _plain(result.output)
    assert "permission-mode" in plain
    assert "bogus" in plain


def test_serve_accepts_a_valid_permission_mode():
    # 'plan' is valid, so validation passes and execution proceeds to bind a
    # socket; abort there to avoid actually serving. A BadParameter would instead
    # have exited before any networking happened.
    import a2claude.cli as cli_mod

    def _boom(*_args, **_kwargs):
        raise RuntimeError("reached uvicorn.run")

    original = cli_mod.uvicorn.run
    cli_mod.uvicorn.run = _boom
    try:
        result = runner.invoke(
            app, ["serve", "--backend", "echo", "--permission-mode", "plan"]
        )
    finally:
        cli_mod.uvicorn.run = original

    assert isinstance(result.exception, RuntimeError)
    assert "reached uvicorn.run" in str(result.exception)
