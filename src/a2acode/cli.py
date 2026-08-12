"""Command line interface.

Three commands, enough to run the server and exercise it by hand:

    a2acode serve        start the A2A server
    a2acode call TEXT    send a message and print the streamed events
    a2acode card         fetch and print the agent card
"""

from __future__ import annotations

import asyncio
import json
import shlex
from typing import Annotated
from uuid import uuid4

import httpx
import typer
import uvicorn

app = typer.Typer(
    add_completion=False,
    help="Serve a coding agent (Claude Code and other ACP agents) over A2A.",
)

# Human-facing card names for the ACP agents shipped with a launch preset.
_AGENT_CARD_NAMES = {
    "claude": "Claude Code",
    "gemini": "Gemini CLI",
    "codex": "Codex",
}


def _validate_permission_mode(value: str | None) -> None:
    """Reject an invalid --permission-mode at startup.

    Without this the bad value flows into ClaudeAgentOptions and only surfaces as
    a generic "Claude Code run failed" on the first request, after the server is
    already up. The valid set is read from the SDK's own literal so it cannot
    drift; the import stays lazy so the echo backend needs no SDK at hand.

    Skip validation rather than block startup when the accepted set cannot be
    determined: the SDK absent (echo without it installed), or PermissionMode no
    longer a Literal so get_args returns an empty tuple. In both cases the SDK
    itself still rejects a genuinely bad value at run time.
    """
    if value is None:
        return
    from typing import get_args

    try:
        from claude_agent_sdk import PermissionMode
    except ImportError:
        return

    # Keep only string members, so a non-Literal form (e.g. a Union with
    # non-string args) neither breaks the join below nor is matched against.
    valid = [v for v in get_args(PermissionMode) if isinstance(v, str)]
    if valid and value not in valid:
        raise typer.BadParameter(
            f"invalid --permission-mode {value!r}; expected one of {', '.join(valid)}"
        )


def _check_task_db(dsn: str) -> None:
    """Open the database once, so a bad DSN fails here rather than at startup.

    ``build_app`` only builds the engine; the schema is created during the ASGI
    lifespan, which would surface a refused connection as a traceback out of
    uvicorn long after the flag was accepted.
    """
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError as e:
        raise typer.BadParameter(
            f"--task-db needs the persistence extra (uv sync --extra persistence): {e}"
        ) from e

    async def _connect() -> None:
        engine = create_async_engine(dsn)
        try:
            async with engine.connect():
                pass
        finally:
            await engine.dispose()

    try:
        asyncio.run(_connect())
    except Exception as e:
        raise typer.BadParameter(f"invalid --task-db: {e}") from e


def _local_url(host: str, port: int) -> str:
    shown = "localhost" if host in ("0.0.0.0", "::") else host
    return f"http://{shown}:{port}/"


@app.command()
def serve(
    backend: str = typer.Option(
        "acp", help="Backend: 'acp' (any ACP agent), 'claude' (Claude SDK), 'echo'."
    ),
    agent: str = typer.Option(
        "claude",
        help="ACP agent the 'acp' backend fronts: 'claude', 'gemini', 'codex', "
        "or any agent reachable via --agent-command.",
    ),
    agent_command: str | None = typer.Option(
        None,
        help="Explicit launch command for an ACP agent, overriding --agent's "
        "preset (e.g. 'npx -y @zed-industries/claude-agent-acp').",
    ),
    cwd: str = typer.Option(".", help="Project directory the coding agent works in."),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(9100),
    permission_mode: str | None = typer.Option(
        None,
        help="Claude permission mode (e.g. acceptEdits). Omit to use defaults.",
    ),
    max_budget_usd: float | None = typer.Option(
        None, help="Hard cost ceiling per run, in USD."
    ),
    sign_key: str | None = typer.Option(
        None,
        help="Path to a file holding the signing key (a PEM private key, or a "
        "shared secret for HS256) used to sign the agent card so callers can "
        "verify who issued it.",
    ),
    sign_kid: str | None = typer.Option(
        None, help="Key id (kid) recorded in the card signature."
    ),
    sign_alg: str = typer.Option(
        "ES256", help="JWS algorithm for the card signature (e.g. ES256, RS256)."
    ),
    auth_token_file: str | None = typer.Option(
        None,
        help="Path to a file holding a bearer token. When set, callers must "
        "send 'Authorization: Bearer <token>' and the card advertises it.",
    ),
    task_db: str | None = typer.Option(
        None,
        help="SQLAlchemy async DSN for persisting tasks and push-notification "
        "registrations across restarts (e.g. sqlite+aiosqlite:///a2acode.db). "
        "Needs the 'persistence' extra. Omit to keep them in memory.",
    ),
) -> None:
    """Start the A2A server."""
    from pathlib import Path

    from .backends import make_backend
    from .server import build_app

    kwargs: dict[str, object] = {}
    card_name = None
    if backend == "acp":
        kwargs = {"agent": agent, "cwd": cwd}
        if agent_command:
            try:
                parts = shlex.split(agent_command)
            except ValueError as e:
                raise typer.BadParameter(f"invalid --agent-command: {e}") from e
            if not parts:
                raise typer.BadParameter("--agent-command is empty or whitespace-only")
            kwargs["command"], kwargs["args"] = parts[0], parts[1:]
            # The identity of an arbitrary command is unknown, so don't advertise
            # it under a preset name (which would default to "Claude Code").
            card_name = "Coding Agent"
        else:
            card_name = _AGENT_CARD_NAMES.get(agent, agent.capitalize())
    elif backend == "claude":
        # Claude-only flag: validate it only on the Claude path so an ACP run
        # isn't rejected for a flag it never uses.
        _validate_permission_mode(permission_mode)
        kwargs = {
            "cwd": cwd,
            "permission_mode": permission_mode,
            "max_budget_usd": max_budget_usd,
        }
    drv = make_backend(backend, **kwargs)

    card_signer = None
    if sign_kid and not sign_key:
        raise typer.BadParameter("--sign-key is required when --sign-kid is set")
    if sign_key:
        if not sign_kid:
            raise typer.BadParameter("--sign-kid is required when --sign-key is set")
        from .card import signer_from_key_file

        try:
            card_signer = signer_from_key_file(sign_key, kid=sign_kid, alg=sign_alg)
        except (OSError, ValueError) as e:
            raise typer.BadParameter(f"invalid --sign-key: {e}") from e

    auth_token = None
    if auth_token_file:
        try:
            auth_token = Path(auth_token_file).read_text(encoding="utf-8").strip()
        except (OSError, ValueError) as e:
            raise typer.BadParameter(f"invalid --auth-token-file: {e}") from e
        if not auth_token:
            raise typer.BadParameter("--auth-token-file is empty")

    if task_db:
        _check_task_db(task_db)

    asgi_app = build_app(
        drv,
        url=_local_url(host, port),
        card_name=card_name,
        card_signer=card_signer,
        auth_token=auth_token,
        task_db=task_db,
    )
    label = f"{backend}:{agent}" if backend == "acp" else backend
    typer.echo(f"a2acode: backend={label} card={_local_url(host, port)}")
    uvicorn.run(asgi_app, host=host, port=port, log_level="info")


@app.command()
def call(
    text: str = typer.Argument(..., help="Message to send to the agent."),
    url: str = typer.Option("http://localhost:9100/", help="Server URL."),
    context: str | None = typer.Option(
        None, help="contextId to continue a conversation."
    ),
    task: str | None = typer.Option(
        None, help="taskId to answer an input-required prompt (e.g. 'allow')."
    ),
    request: str | None = typer.Option(
        None,
        help="requestId from the prompt being answered, so a resend cannot "
        "settle a later one.",
    ),
    answer: Annotated[
        list[str] | None,
        typer.Option(
            "--answer",
            help="Answer to a question the agent asked, as 'question=choice'. "
            "Repeat for each question, and again on one question to pick several.",
        ),
    ] = None,
) -> None:
    """Send a message and print the streamed task events."""
    asyncio.run(_call(text, url, context, task, request, _answers(answer or [])))


@app.command()
def card(url: str = typer.Option("http://localhost:9100/")) -> None:
    """Fetch and print the agent card."""
    base = url.rstrip("/")
    resp = httpx.get(f"{base}/.well-known/agent-card.json", timeout=10)
    resp.raise_for_status()
    typer.echo(json.dumps(resp.json(), indent=2))


def _parts_text(parts) -> str:
    return "".join(p.text for p in parts if p.text)


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


def _state_name(state) -> str:
    from a2a.types import TaskState

    return TaskState.Name(state).removeprefix("TASK_STATE_").lower()


def _answers(pairs: list[str]) -> dict[str, str | list[str]]:
    """Fold 'question=choice' options into the object the server reads."""
    answers: dict[str, str | list[str]] = {}
    for pair in pairs:
        question, sep, choice = pair.partition("=")
        question, choice = question.strip(), choice.strip()
        if not sep or not question or not choice:
            raise typer.BadParameter(f"--answer must be 'question=choice': {pair!r}")
        previous = answers.get(question)
        if previous is None:
            answers[question] = choice
        elif isinstance(previous, list):
            previous.append(choice)
        else:
            answers[question] = [previous, choice]
    return answers


async def _call(
    text: str,
    url: str,
    context: str | None,
    task: str | None,
    request_id: str | None = None,
    answers: dict[str, str | list[str]] | None = None,
) -> None:
    from a2a.client import create_client
    from a2a.client.client import ClientConfig
    from a2a.types import Message, Part, Role, SendMessageRequest

    message = Message(
        message_id=uuid4().hex,
        role=Role.ROLE_USER,
        parts=[Part(text=text)],
    )
    if context:
        message.context_id = context
    if task:
        message.task_id = task
    block: dict[str, object] = {}
    if request_id:
        # Names the prompt this answers, so the server can turn it down rather
        # than apply it to whatever it is waiting on now.
        block["request_id"] = request_id
    if answers:
        block["answers"] = answers
    if block:
        message.metadata.update({"a2acode_permission": block})

    ids = {"task": task or "", "context": context or "", "request": ""}
    streaming = False
    timeout = httpx.Timeout(600.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as http:
        client = await create_client(
            url, ClientConfig(streaming=True, httpx_client=http)
        )
        try:
            request = SendMessageRequest(message=message)
            async for event in client.send_message(request):
                which = event.WhichOneof("payload")
                if which == "task":
                    t = event.task
                    ids["task"], ids["context"] = t.id, t.context_id
                    typer.echo(f"task {t.id}")
                    typer.echo(f"context {t.context_id}\n")
                elif which == "status_update":
                    s = event.status_update.status
                    line = _parts_text(s.message.parts) if s.message else ""
                    state = _state_name(s.state)
                    if streaming and state != "working":
                        typer.echo("")
                        streaming = False
                    if state == "working" and line:
                        typer.echo(f"  · {line}")
                    elif state == "input_required":
                        asked = _permission_block(s.message)
                        ids["request"] = str(asked.get("request_id", ""))
                        _render_input_required(line, ids, url, asked)
                    elif state != "working":
                        meta = _format_meta(s.message) if s.message else ""
                        typer.echo(f"[{state}] {meta}".rstrip())
                elif which == "artifact_update":
                    artifact = event.artifact_update.artifact
                    text = _parts_text(artifact.parts)
                    if artifact.name == "response":
                        streaming = True
                        typer.echo(text, nl=False)
                    elif text:
                        # Plans, reasoning, and diffs are their own artifacts;
                        # labelling them keeps them out of the answer's stream.
                        if streaming:
                            typer.echo("")
                            streaming = False
                        typer.echo(f"  [{artifact.name}]")
                        typer.echo(_indent(text.rstrip()))
                elif which == "message":
                    typer.echo(_parts_text(event.message.parts))
        finally:
            closer = client.close()
            if closer is not None:
                await closer


def _permission_block(msg) -> dict:
    """What the pause published about the request, for the reply to answer."""
    from google.protobuf.json_format import MessageToDict

    block = (MessageToDict(msg).get("metadata") or {}).get("a2acode_permission")
    return block if isinstance(block, dict) else {}


def _questions(block: dict) -> list[dict]:
    """The questions a gate asked, when asking is all it does."""
    asked = block.get("input")
    questions = asked.get("questions") if isinstance(asked, dict) else None
    if not isinstance(questions, list):
        return []
    return [q for q in questions if isinstance(q, dict)]


def _render_questions(questions: list[dict]) -> str:
    """List each question and its choices, and return the --answer flags for them.

    Folded through the same sanitizer as the rest of the pause, so a question
    carrying control characters is shown rather than obeyed — and the template it
    prints answers that folded text, which such a question will not match.
    """
    from .executor import one_line

    flags = ""
    for question in questions:
        text = one_line(str(question.get("question", "")))
        header = one_line(str(question.get("header", "")))
        typer.echo(f"  {text}" + (f"  [{header}]" if header else ""))
        options = question.get("options")
        for option in options if isinstance(options, list) else []:
            if isinstance(option, dict):
                label = one_line(str(option.get("label", "")))
                about = one_line(str(option.get("description", "")))
                typer.echo(f"      {label} — {about}" if about else f"      {label}")
        if question.get("multiSelect"):
            typer.echo("      (repeat --answer to pick several)")
        flags += " --answer " + shlex.quote(f"{text}=<choice>")
    return flags


def _render_input_required(
    line: str, ids: dict[str, str], url: str, block: dict
) -> None:
    typer.echo(f"[input-required] {line}")
    questions = _questions(block)
    flags = _render_questions(questions) if questions else ""
    named = f" --request {ids['request']}" if ids.get("request") else ""
    follow = (
        f'a2acode call "allow" --task {ids["task"]} '
        f"--context {ids['context']}{named} --url {url}{flags}"
    )
    typer.echo(f"  reply: {follow}")
    typer.echo('  (or "deny" to refuse)')


def _format_meta(msg) -> str:
    from google.protobuf.json_format import MessageToDict

    meta = MessageToDict(msg).get("metadata") if msg else None
    if not meta:
        return ""
    bits = []
    if "cost_usd" in meta:
        bits.append(f"${meta['cost_usd']}")
    if "num_turns" in meta:
        bits.append(f"{meta['num_turns']} turns")
    return " · ".join(bits)


if __name__ == "__main__":
    app()
