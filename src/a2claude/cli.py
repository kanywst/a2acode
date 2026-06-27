"""Command line interface.

Three commands, enough to run the server and exercise it by hand:

    a2claude serve        start the A2A server
    a2claude call TEXT    send a message and print the streamed events
    a2claude card         fetch and print the agent card
"""

from __future__ import annotations

import asyncio
import json
import shlex
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
    agent_command: str = typer.Option(
        None,
        help="Explicit launch command for an ACP agent, overriding --agent's "
        "preset (e.g. 'npx -y @zed-industries/claude-agent-acp').",
    ),
    cwd: str = typer.Option(".", help="Project directory the coding agent works in."),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(9100),
    permission_mode: str = typer.Option(
        None,
        help="Claude permission mode (e.g. acceptEdits). Omit to use defaults.",
    ),
    max_budget_usd: float = typer.Option(
        None, help="Hard cost ceiling per run, in USD."
    ),
    sign_key: str = typer.Option(
        None,
        help="Path to a file holding the signing key (a PEM private key, or a "
        "shared secret for HS256) used to sign the agent card so callers can "
        "verify who issued it.",
    ),
    sign_kid: str = typer.Option(
        None, help="Key id (kid) recorded in the card signature."
    ),
    sign_alg: str = typer.Option(
        "ES256", help="JWS algorithm for the card signature (e.g. ES256, RS256)."
    ),
    auth_token_file: str = typer.Option(
        None,
        help="Path to a file holding a bearer token. When set, callers must "
        "send 'Authorization: Bearer <token>' and the card advertises it.",
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

    asgi_app = build_app(
        drv,
        url=_local_url(host, port),
        card_name=card_name,
        card_signer=card_signer,
        auth_token=auth_token,
    )
    label = f"{backend}:{agent}" if backend == "acp" else backend
    typer.echo(f"a2claude: backend={label} card={_local_url(host, port)}")
    uvicorn.run(asgi_app, host=host, port=port, log_level="info")


@app.command()
def call(
    text: str = typer.Argument(..., help="Message to send to the agent."),
    url: str = typer.Option("http://localhost:9100/", help="Server URL."),
    context: str = typer.Option(None, help="contextId to continue a conversation."),
    task: str = typer.Option(
        None, help="taskId to answer an input-required prompt (e.g. 'allow')."
    ),
) -> None:
    """Send a message and print the streamed task events."""
    asyncio.run(_call(text, url, context, task))


@app.command()
def card(url: str = typer.Option("http://localhost:9100/")) -> None:
    """Fetch and print the agent card."""
    base = url.rstrip("/")
    resp = httpx.get(f"{base}/.well-known/agent-card.json", timeout=10)
    resp.raise_for_status()
    typer.echo(json.dumps(resp.json(), indent=2))


def _parts_text(parts) -> str:
    return "".join(p.text for p in parts if p.text)


def _state_name(state) -> str:
    from a2a.types import TaskState

    return TaskState.Name(state).removeprefix("TASK_STATE_").lower()


async def _call(text: str, url: str, context: str | None, task: str | None) -> None:
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

    ids = {"task": task or "", "context": context or ""}
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
                        _render_input_required(line, ids, url)
                    elif state != "working":
                        meta = _format_meta(s.message) if s.message else ""
                        typer.echo(f"[{state}] {meta}".rstrip())
                elif which == "artifact_update":
                    streaming = True
                    parts = event.artifact_update.artifact.parts
                    typer.echo(_parts_text(parts), nl=False)
                elif which == "message":
                    typer.echo(_parts_text(event.message.parts))
        finally:
            closer = client.close()
            if closer is not None:
                await closer


def _render_input_required(line: str, ids: dict[str, str], url: str) -> None:
    typer.echo(f"[input-required] {line}")
    follow = (
        f'a2claude call "allow" --task {ids["task"]} '
        f"--context {ids['context']} --url {url}"
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
