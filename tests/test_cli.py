"""CLI argument validation that must fail before the server starts."""

from __future__ import annotations

from typer.testing import CliRunner

from a2claude.cli import app

runner = CliRunner()


def test_serve_rejects_invalid_permission_mode():
    # Fails during validation, before uvicorn.run, so the runner does not block.
    result = runner.invoke(
        app, ["serve", "--backend", "echo", "--permission-mode", "bogus"]
    )
    assert result.exit_code != 0
    assert "permission-mode" in result.output
    assert "bogus" in result.output


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
