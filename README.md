<img src="assets/mascot.png" alt="a2acode" width="150" align="right">

# a2acode

Serve a coding agent over the [A2A](https://a2aprotocol.ai/) protocol. Other agents call it over A2A; it drives a real coding-agent session in your project — Claude Code, or any agent that speaks Zed's [Agent Client Protocol](https://agentclientprotocol.com) (ACP): Gemini CLI, Codex, OpenHands, and more — and streams the work back as it happens.

[![CI](https://github.com/kanywst/a2acode/actions/workflows/ci.yml/badge.svg)](https://github.com/kanywst/a2acode/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![Protocol: A2A 1.0](https://img.shields.io/badge/protocol-A2A%201.0-D97757.svg)](https://a2aprotocol.ai/)

![a2acode streaming a task, then pausing on a permission prompt](assets/demo.gif)

Most adapters that put a coding agent behind A2A flatten everything to text in, text out. a2acode keeps the structure the agent produces: the tools it runs, the files it changes, what it costs, the approvals it needs, and how to continue on the next turn. It bridges two Linux Foundation interop standards — **ACP** (how editors and clients talk to coding agents) on the agent side, **A2A** (how agents delegate to each other) on the caller side — so any ACP agent becomes a peer any A2A orchestrator can call.

## How it maps to A2A

| The coding agent produces | A2A surface it lands on                            |
| ------------------------- | -------------------------------------------------- |
| Assistant text            | A streamed artifact (`append` / `last_chunk`)      |
| A tool call (Bash, Edit)  | A `working` status update for the action           |
| A file edit (diff)        | A named artifact carrying the diff                 |
| A permission request      | An `input-required` pause the caller answers       |
| Run result               | Cost, turns, and usage on the completion message   |
| Session id                | Mapped to the A2A `contextId` so follow-ups resume |

The mapping is all in `executor.py`. Backends only emit normalized events; they never touch the protocol.

## Where this fits

Anthropic now ships its own ways to run Claude Code beyond the terminal: Claude Code on the web, background agents, cloud-hosted Routines, and the Managed Agents API. These are the right choices when you want Anthropic to host the run and you live in their ecosystem, and they are typically tied to Anthropic infrastructure and a GitHub-centric flow.

a2acode solves a different problem: making any coding agent a first-class peer on a vendor-neutral [A2A](https://a2aprotocol.ai/) mesh. An orchestrator built on any framework discovers it through its agent card and delegates coding work the same way it would to any other A2A agent. The run happens on infrastructure you control, in a workspace you point it at. Reach for a2acode when:

- another agent (not a human at a prompt) is the caller, and it speaks A2A;
- you want the run on your own infrastructure and data boundary, not a vendor VM;
- you do not want to bet on one vendor's coding agent: ACP makes the backend a launch-command choice, so swapping Claude Code for Codex, Gemini CLI, or OpenHands does not touch the protocol surface your callers depend on.

ACP already standardizes the editor↔agent side and a dozen agents speak it; a2acode is the piece that exposes an ACP agent to *remote autonomous callers* over A2A, with permission round-trips and cost preserved as first-class protocol citizens — the part ACP leaves out because it assumes a human in an editor. The practical user is the platform team building that mesh, not the individual developer.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- An ACP agent adapter for the `acp` backend, launched as a subprocess. The `claude` preset uses `npx @zed-industries/claude-agent-acp` (needs Node and a Claude credential); `gemini` uses the Gemini CLI; or point `--agent-command` at any ACP agent.

## Quick start

Install:

```bash
uv sync
```

The `echo` backend needs no API key and no Claude install, so you can exercise the whole path offline first:

```bash
uv run a2acode serve --backend echo &
# once the "Uvicorn running" line appears:
uv run a2acode call "fix the failing test"
```

```text
task 189b1c63-1a7b-4908-87c4-c8f3bba8f6b5
context 0b2a901e-2b6f-4c56-bba2-d0da546936e9

  · Echo
fix the failing test
[completed] $0.0 · 1 turns
```

Then point it at a real project. The default backend is `acp`, fronting Claude Code through its ACP adapter:

```bash
uv run a2acode serve --cwd /path/to/project          # acp + claude by default
uv run a2acode call "add a /health endpoint" --url http://localhost:9100/
```

Swap the agent without touching anything else:

```bash
uv run a2acode serve --agent gemini --cwd /path/to/project
uv run a2acode serve --agent-command "npx -y some-other-acp-agent"
```

Continue the same conversation by passing the `context` from a previous turn:

```bash
uv run a2acode call "now add a test for it" --context <context-id>
```

## Commands

| Command              | Description                                  |
| -------------------- | -------------------------------------------- |
| `a2acode serve`     | Start the A2A server                         |
| `a2acode call TEXT` | Send a message and print the streamed events |
| `a2acode card`      | Fetch and print the agent card               |

The agent card is served at `/.well-known/agent-card.json` and advertises Claude Code's abilities as discrete skills (generation, refactor, debug, review, test, explain).

## Backends

A backend turns a prompt into a stream of normalized events. Three ship today:

- `acp` (default): drives any agent that speaks Zed's Agent Client Protocol as a subprocess. `--agent claude|gemini|codex` selects a launch preset; `--agent-command` drives any other ACP agent. This is the vendor-neutral path.
- `claude`: drives Claude Code directly through the Claude Agent SDK, no subprocess. Install with `uv sync --extra claude`. Use it when you want the SDK-native path (e.g. `--max-budget-usd`) rather than ACP.
- `echo`: no dependencies, mirrors the input. For wiring checks and tests.

The split keeps the A2A layer independent of how the agent is invoked: backends emit normalized events and never import `a2a.*`; the executor maps those events onto the protocol and never imports an agent SDK. Adding a backend never touches the server or the protocol mapping.

## Authentication

Each agent authenticates the way its own tooling does, inherited from the server's environment: the `acp` backend passes the environment through to the adapter subprocess (e.g. `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`), and the `claude` backend uses whatever the Claude CLI is configured with. When the server answers on behalf of other agents, a Claude credential has to be an Anthropic API key (or Bedrock / Vertex); Anthropic does not permit subscription credentials for third-party serving. The `claude` backend can cap per-run cost with `--max-budget-usd`.

## Signed agent cards

A caller that discovers this server only has the agent card to go on. Sign it so the caller can confirm the card came from you and was not swapped in transit:

```bash
uv run a2acode serve --sign-key card-signing.pem --sign-kid my-key-1 --sign-alg ES256
```

The card is then served with a JWS signature over its canonical form. `--sign-key` is a path to a file holding the key: a PEM private key for asymmetric algorithms (`ES256`, `RS256`), or a shared secret for `HS256`. `--sign-kid` is the key id a verifier uses to look up the matching public key. Unsigned is still the default.

## Caller authentication

A signed card proves who the server is; this proves the caller is allowed in. Require a bearer token and the server rejects any task request that does not carry it:

```bash
uv run a2acode serve --auth-token-file caller-token.txt
```

When `--auth-token-file` is set, callers must send `Authorization: Bearer <token>`; a request without a valid token gets `401 Unauthorized`. The agent card stays public so a caller can still fetch it to discover the requirement, and the card advertises the bearer scheme in `securitySchemes`. Without the flag the server stays open, as before.

A2A keeps the credential at the HTTP layer, so this composes with whatever your gateway already does: terminate TLS, validate OAuth, or rate-limit in front, and let the server enforce the token behind it.

## Permissions

A tool that needs approval pauses the task in the A2A `input-required` state instead of being skipped. The caller answers with a follow-up message on the same task:

```bash
uv run a2acode call "sudo reboot"
# ... [input-required] Permission requested for Bash: $ sudo reboot
#       reply: a2acode call "allow" --task <id> --context <id>
uv run a2acode call "allow" --task <id> --context <id>
```

`allow` (or `yes`, `approve`, `ok`) approves; anything else denies. The agent session stays alive across the pause, so it resumes exactly where it stopped. Over ACP this is the agent's `session/request_permission` call answered from the A2A caller's reply; with the `claude` backend it routes through the Claude SDK's `can_use_tool`.

Whatever the agent decides needs approval becomes an `input-required` pause rather than being silently skipped or auto-approved; the caller, not the server, holds the decision. Read-only actions the agent already treats as safe still run without a prompt.

## Long-running tasks

The agent card advertises push notifications. A caller can register a webhook for a task and receive status and artifact updates by HTTP POST instead of holding a stream open, which helps when a run takes minutes. Streaming and polling (`tasks/get`) both work too.

## Observability

Debugging one agent is hard; debugging a chain of them without traces is worse. Because A2A runs over HTTP, it drops straight into OpenTelemetry: install the extra and the A2A SDK's instrumentation plus a per-task `a2acode.execute` span light up, with W3C trace context propagating across the call so client and server spans share one trace.

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
