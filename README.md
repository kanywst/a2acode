<img src="assets/mascot.png" alt="a2claude" width="150" align="right">

# a2claude

Run Claude Code as an [A2A](https://a2aprotocol.ai/) agent server. Other agents call it over the protocol; it drives a real Claude Code session in your project and streams the work back as it happens.

[![CI](https://github.com/kanywst/a2claude/actions/workflows/ci.yml/badge.svg)](https://github.com/kanywst/a2claude/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![Protocol: A2A 1.0](https://img.shields.io/badge/protocol-A2A%201.0-D97757.svg)](https://a2aprotocol.ai/)

![a2claude streaming a task, then pausing on a permission prompt](assets/demo.gif)

Most adapters that put a coding agent behind A2A flatten everything to text in, text out. a2claude keeps the structure Claude Code produces: the tools it runs, the files it changes, what it costs, and how to continue on the next turn.

## How it maps to A2A

| Claude Code produces     | A2A surface it lands on                            |
| ------------------------ | -------------------------------------------------- |
| Assistant text           | A streamed artifact (`append` / `last_chunk`)      |
| A tool call (Bash, Edit) | A `working` status update for the action           |
| A file edit              | A named artifact carrying the diff                 |
| Run result               | Cost, turns, and usage on the completion message   |
| Session id               | Mapped to the A2A `contextId` so follow-ups resume |

The mapping is all in `executor.py`. Backends only emit normalized events; they never touch the protocol.

## Where this fits

Anthropic now ships its own ways to run Claude Code beyond the terminal: Claude Code on the web, background agents, cloud-hosted Routines, and the Managed Agents API. These are the right choices when you want Anthropic to host the run and you live in their ecosystem, and they are typically tied to Anthropic infrastructure and a GitHub-centric flow.

a2claude solves a different problem: making Claude Code a first-class peer on a vendor-neutral [A2A](https://a2aprotocol.ai/) mesh. An orchestrator built on any framework discovers it through its agent card and delegates coding work to it the same way it would to any other A2A agent. The run happens on infrastructure you control, in a workspace you point it at. Reach for a2claude when:

- another agent (not a human at a prompt) is the caller, and it speaks A2A;
- you want the run on your own infrastructure and data boundary, not a vendor VM;
- you are wiring Claude Code into a multi-vendor agent system rather than standardizing on one vendor's hosted stack.

The practical user is the platform team building that mesh, not the individual developer; the developer reaches it through whatever orchestrator the team puts in front of them.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Claude Code CLI on `PATH` (only for the `claude` backend)

## Quick start

Install:

```bash
uv sync
```

The `echo` backend needs no API key and no Claude install, so you can exercise the whole path offline first:

```bash
uv run a2claude serve --backend echo &
# once the "Uvicorn running" line appears:
uv run a2claude call "fix the failing test"
```

```text
task 189b1c63-1a7b-4908-87c4-c8f3bba8f6b5
context 0b2a901e-2b6f-4c56-bba2-d0da546936e9

  · Echo
fix the failing test
[completed] $0.0 · 1 turns
```

Then point it at a real project:

```bash
uv run a2claude serve --backend claude --cwd /path/to/project
uv run a2claude call "add a /health endpoint" --url http://localhost:9100/
```

Continue the same conversation by passing the `context` from a previous turn:

```bash
uv run a2claude call "now add a test for it" --context <context-id>
```

## Commands

| Command              | Description                                  |
| -------------------- | -------------------------------------------- |
| `a2claude serve`     | Start the A2A server                         |
| `a2claude call TEXT` | Send a message and print the streamed events |
| `a2claude card`      | Fetch and print the agent card               |

The agent card is served at `/.well-known/agent-card.json` and advertises Claude Code's abilities as discrete skills (generation, refactor, debug, review, test, explain).

## Backends

A backend turns a prompt into a stream of normalized events. Two ship today:

- `echo`: no dependencies, mirrors the input. For wiring checks and tests.
- `claude`: drives Claude Code through the Claude Agent SDK.

The split keeps the A2A layer independent of how Claude Code is invoked, so a raw-CLI backend can be added later without touching the server or the protocol mapping.

## Authentication

The `claude` backend uses whatever the Claude CLI is configured with. When the server answers on behalf of other agents, that has to be an Anthropic API key (or Bedrock / Vertex). Anthropic does not permit subscription credentials for third-party serving. Set a per-run cost ceiling with `--max-budget-usd`.

## Signed agent cards

A caller that discovers this server only has the agent card to go on. Sign it so the caller can confirm the card came from you and was not swapped in transit:

```bash
uv run a2claude serve --sign-key card-signing.pem --sign-kid my-key-1 --sign-alg ES256
```

The card is then served with a JWS signature over its canonical form. `--sign-key` is a path to a file holding the key: a PEM private key for asymmetric algorithms (`ES256`, `RS256`), or a shared secret for `HS256`. `--sign-kid` is the key id a verifier uses to look up the matching public key. Unsigned is still the default.

## Caller authentication

A signed card proves who the server is; this proves the caller is allowed in. Require a bearer token and the server rejects any task request that does not carry it:

```bash
uv run a2claude serve --auth-token-file caller-token.txt
```

When `--auth-token-file` is set, callers must send `Authorization: Bearer <token>`; a request without a valid token gets `401 Unauthorized`. The agent card stays public so a caller can still fetch it to discover the requirement, and the card advertises the bearer scheme in `securitySchemes`. Without the flag the server stays open, as before.

A2A keeps the credential at the HTTP layer, so this composes with whatever your gateway already does: terminate TLS, validate OAuth, or rate-limit in front, and let the server enforce the token behind it.

## Permissions

A tool that needs approval pauses the task in the A2A `input-required` state instead of being skipped. The caller answers with a follow-up message on the same task:

```bash
uv run a2claude call "sudo reboot"
# ... [input-required] Permission requested for Bash: $ sudo reboot
#       reply: a2claude call "allow" --task <id> --context <id>
uv run a2claude call "allow" --task <id> --context <id>
```

`allow` (or `yes`, `approve`, `ok`) approves; anything else denies. The Claude session stays alive across the pause, so it resumes exactly where it stopped.

The server does not inherit your personal Claude settings, so it has no pre-approved tool allowlist; every tool that needs approval routes through the caller. Read-only actions Claude already treats as safe still run without a prompt.

## Long-running tasks

The agent card advertises push notifications. A caller can register a webhook for a task and receive status and artifact updates by HTTP POST instead of holding a stream open, which helps when a run takes minutes. Streaming and polling (`tasks/get`) both work too.

## Observability

Debugging one agent is hard; debugging a chain of them without traces is worse. Because A2A runs over HTTP, it drops straight into OpenTelemetry: install the extra and the A2A SDK's instrumentation plus a per-task `a2claude.execute` span light up, with W3C trace context propagating across the call so client and server spans share one trace.

```bash
uv sync --extra telemetry
```

Tracing is off unless OpenTelemetry is installed, and you configure the exporter the standard way (e.g. `OTEL_EXPORTER_OTLP_ENDPOINT`, or run under `opentelemetry-instrument`). It works against an on-prem or air-gapped collector, so traces never have to leave your network.

## Development

```bash
uv sync --dev
uv run ruff check src tests
uv run ruff format src tests
uv run mypy
uv run pytest
```

CI runs these on Python 3.13 and 3.14, plus a Markdown lint, on every push and pull request.

## Releasing

Pushing a `v*` tag builds the package, creates a GitHub release with the artifacts, and publishes to PyPI via trusted publishing:

```bash
git tag v0.1.0
git push origin v0.1.0
```

## Status

The mapping is complete end to end and verified against real Claude: text round trip, tool-progress updates, streaming artifacts, file diffs as artifacts, run metadata, session continuity, the permission-to-`input-required` round trip, and push notifications. The offline `echo` backend covers every path including permissions, so it can all be exercised without an API key.

## License

Apache 2.0. See [LICENSE](LICENSE).
