"""CLI argument validation that must fail before the server starts."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from a2acode.cli import app

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


@patch("a2acode.cli.uvicorn.run")
def test_serve_rejects_invalid_permission_mode(mock_run: MagicMock) -> None:
    # --permission-mode is a Claude-only flag, so it is validated on the claude
    # path. A bad value fails before uvicorn.run, so the server never starts.
    result = runner.invoke(
        app, ["serve", "--backend", "claude", "--permission-mode", "bogus"]
    )
    assert result.exit_code != 0
    mock_run.assert_not_called()
    plain = _plain(result.output)
    assert "permission-mode" in plain
    assert "bogus" in plain


@patch("a2acode.cli.uvicorn.run")
def test_serve_accepts_a_valid_permission_mode(mock_run: MagicMock) -> None:
    # 'plan' is valid, so validation passes and execution proceeds to uvicorn.run
    # (mocked here so no socket is bound).
    result = runner.invoke(
        app, ["serve", "--backend", "claude", "--permission-mode", "plan"]
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()


@patch("a2acode.cli.uvicorn.run")
def test_serve_ignores_permission_mode_off_the_claude_path(
    mock_run: MagicMock,
) -> None:
    # A non-Claude backend never uses --permission-mode, so an otherwise invalid
    # value must not block startup: validation belongs to the claude path only.
    result = runner.invoke(
        app, ["serve", "--backend", "echo", "--permission-mode", "bogus"]
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()


@patch("a2acode.cli.uvicorn.run")
def test_serve_rejects_malformed_agent_command(mock_run: MagicMock) -> None:
    # An unmatched quote makes shlex.split raise; it must surface as a clean
    # BadParameter, not a traceback, and the server must not start.
    result = runner.invoke(
        app, ["serve", "--backend", "acp", "--agent-command", "npx 'unterminated"]
    )
    assert result.exit_code != 0
    mock_run.assert_not_called()
    assert "agent-command" in _plain(result.output)


def test_answer_pairs_fold_into_one_object() -> None:
    from a2acode.cli import _answers

    answers = _answers(["Runner=pytest", "Files=a.py", "Files=b.py"])

    # Repeating one question is how a multi-select answer names both choices.
    assert answers == {"Runner": "pytest", "Files": ["a.py", "b.py"]}


def test_answer_without_a_choice_is_rejected() -> None:
    result = runner.invoke(app, ["call", "allow", "--answer", "Runner"])

    assert result.exit_code != 0
    assert "questionchoice" in _plain(result.output)


def test_input_required_lists_the_questions_and_how_to_answer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from a2acode.cli import _render_input_required

    _render_input_required(
        "Permission requested for AskUserQuestion: asking",
        {"task": "t1", "context": "c1", "request": "r1"},
        "http://localhost:9100/",
        {
            "request_id": "r1",
            "tool": "AskUserQuestion",
            "input": {
                "questions": [
                    {
                        "question": "Which runner?\x1b[2K",
                        "header": "Runner",
                        "options": [{"label": "pytest", "description": "fast"}],
                    }
                ]
            },
        },
    )

    out = capsys.readouterr().out
    # A caller answers by question text, so it has to be shown one; the escape
    # is rendered, not acted on, the same as anywhere else in the pause.
    assert "\x1b" not in out
    assert "pytest: fast" in out
    assert "--answer 'Which runner?\\x1b[2K=<choice>'" in out
