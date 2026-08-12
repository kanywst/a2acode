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

| The coding agent produces   | A2A surface it lands on                            |
| --------------------------- | -------------------------------------------------- |
| Assistant text              | A streamed artifact (`append` / `last_chunk`)      |
| Its reasoning               | A separate `thinking` artifact, never the answer   |
| A tool call (Bash, Edit)    | A `working` status update for the action           |
| That tool's outcome         | A `working` status update: `✓ Bash` / `✗ Bash: …`  |
| Its plan for the turn       | A `plan` artifact, replaced as steps progress      |
| A file edit (diff)          | A named artifact carrying the diff                 |
| A permission request        | An `input-required` pause the caller answers       |
| Run result                  | Cost, turns, usage, stop reason on the completion  |
| Session id                  | Mapped to the A2A `contextId` so follow-ups resume |

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
- An ACP agent adapter for the `acp` backend, launched as a subprocess. Every preset launches one with `npx`, so Node is the only prerequisite besides that agent's own credential:

| `--agent` | Launches                            | Credential           |
| --------- | ----------------------------------- | -------------------- |
| `claude`  | `@zed-industries/claude-agent-acp`  | Anthropic API key    |
| `codex`   | `@zed-industries/codex-acp`         | OpenAI credential    |
| `gemini`  | `@google/gemini-cli --acp`          | Code Assist Standard or Enterprise |

Or point `--agent-command` at any other ACP agent.

**On the `gemini` preset:** Google stopped serving the Gemini CLI to consumer accounts on 18 June 2026 — the individual Code Assist tier and AI Pro/Ultra access — and points them at Antigravity instead. A session on one of those fails with *"This client is no longer supported for Gemini Code Assist for individuals"*. This is not specific to ACP or to a2acode: it is the whole CLI, on those tiers. Organization Code Assist Standard and Enterprise subscriptions are unaffected, which is why the preset stays.

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

Continuity needs the agent to support ACP's `session/load`. When it does not, the turn still runs, but on a fresh session — and the task says so in a status update rather than answering as if it remembered.

## Commands

| Command              | Description                                  |
| -------------------- | -------------------------------------------- |
| `a2acode serve`     | Start the A2A server                         |
| `a2acode call TEXT` | Send a message and print the streamed events |
| `a2acode card`      | Fetch and print the agent card               |

The agent card is served at `/.well-known/agent-card.json` and advertises Claude Code's abilities as discrete skills (generation, refactor, debug, review, test, explain).

## Attachments

An A2A message is not only text. A caller can attach the failing log, the patch to review, or a screenshot, and the parts reach the agent as content rather than as a note that something was attached: text files are inlined into the prompt, images go to ACP agents that advertise image support as real image blocks, and URL parts arrive as links the agent can fetch. Inlined content is capped per part and per message, and anything trimmed is marked as truncated so the agent knows it is reading a fragment.

## Backends

A backend turns a prompt into a stream of normalized events. Three ship today:

- `acp` (default): drives any agent that speaks Zed's Agent Client Protocol as a subprocess. `--agent claude|gemini|codex` selects a launch preset; `--agent-command` drives any other ACP agent. This is the vendor-neutral path. The subprocess is kept alive per conversation, so a follow-up turn skips the process launch, the ACP handshake, and the session reload and talks straight to the agent that already holds the conversation.
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
#       reply: a2acode call "allow" --task <id> --context <id> --request <id>
uv run a2acode call "allow" --task <id> --context <id> --request <id>
```

An answer of `allow`, `yes`, `y`, `ok`, `approve`, `accept`, or `grant` — the whole answer, nothing around it — approves. Anything else denies, which is what makes it safe to answer in prose. The agent session stays alive across the pause, so it resumes exactly where it stopped. Over ACP this is the agent's `session/request_permission` call answered from the A2A caller's reply; with the `claude` backend it routes through the Claude SDK's `can_use_tool`.

A denial carries the words it was written with, so `no, run pytest -x instead` can redirect the turn rather than merely refuse it. How far the reason travels is the agent protocol's call: the `claude` backend hands it to the agent, and so does a denied ACP terminal, but ACP's answer to an ordinary tool permission carries only the option chosen and has nowhere to put text.

Some gates are not yes/no. Claude Code's end-of-plan gate offers three choices — accept the edits that follow, allow this once and keep gating, or keep planning — so the pause lists whatever the agent offered, each with the `kind` that says what picking it would mean, and the caller answers `option:<id>` with the one it wants. The id is what binds; the kind is the agent's label for it. Anything else still resolves to allow or deny as above; naming an option takes the prefix because the agent chooses both an option's id and its polarity, and a bare answer would let it label a permissive choice with the word a caller reaches for to refuse.

Not every gate is a request to act. When Claude has a clarifying question it calls `AskUserQuestion`, which asks and does nothing else, so approving it says nothing on its own — the answer is the outcome. The pause carries the questions and their choices, and the reply's metadata carries what was picked, keyed by question:

```bash
uv run a2acode call "port the tests" --url http://localhost:9100/
# ... [input-required] Permission requested for AskUserQuestion: ...
#       Which test runner?  [Runner]
#           pytest — what the repo uses
#           unittest — stdlib only
uv run a2acode call "allow" --task <id> --context <id> --request <id> \
  --answer "Which test runner?=pytest"
```

`--answer` repeats, once per question, and again on one question to name several choices for a multi-select. An approval with no answers reaches the agent as nobody having replied, and a denial still travels as the words it was written with, which for a question is a fine way to answer it.

An answer settles one prompt, not whichever is waiting. Each pause carries a `requestId`, and an answer naming it is only applied to that prompt; a run often stops again immediately, so a client retry or a double submit of the previous answer would otherwise decide a tool the caller was never shown. An answer that names nothing is still taken, except when it is the same message arriving twice. Either way the server restates what is actually pending instead of applying the answer.

Whatever the agent decides needs approval becomes an `input-required` pause rather than being silently skipped or auto-approved; the caller, not the server, holds the decision. Read-only actions the agent already treats as safe still run without a prompt.

The same gate covers commands. a2acode serves ACP's terminal capability, so an agent can run a build or a test suite through the server instead of shelling out on its own, and every one of those goes through the caller for approval first — the caller sees the exact command line, environment assignments included, before anything spawns. Without that gate, advertising the capability would have handed the agent a way *around* the permission model, since `terminal/create` is a direct client call and nothing in the protocol obliges an agent to ask permission first.

Be clear about what the gate is and is not. **The approval is the security boundary; the process is not sandboxed.** An approved command runs as the user the server runs as and can read anything that user can, wherever it lives — the workspace only sets its working directory. What a2acode does around that: the command inherits a named set of environment variables (`PATH`, `HOME`, `LANG`, `TMPDIR`, `TERM`, `SHELL`, `USER`, `TZ`, `LC_*`) rather than the server's whole environment, so the provider credentials the server holds are not handed to it; output is capped; and anything still running is killed with the turn. Run the server as a user that owns nothing you would not approve a command to read.

## Long-running tasks

The agent card advertises push notifications. A caller can register a webhook for a task and receive status and artifact updates by HTTP POST instead of holding a stream open, which helps when a run takes minutes. Streaming and polling (`tasks/get`) both work too.

Tasks live in memory by default, so a restart loses their history and any webhook registrations. Point `--task-db` at a database to keep them:

```bash
uv sync --extra persistence
uv run a2acode serve --task-db "sqlite+aiosqlite:///a2acode.db"
```

Any SQLAlchemy async DSN works; the extra ships the SQLite driver because it needs no server. What survives is task history, artifacts, and push registrations — **not** a live agent session, which is a process and dies with the server either way, so a task paused on a permission cannot be answered after a restart.

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

The mapping is complete end to end and verified against real Claude: text round trip, tool calls and their outcomes, the agent's plan, streaming artifacts, file diffs as artifacts, caller attachments, run metadata, session continuity, the permission-to-`input-required` round trip, and push notifications. The offline `echo` backend covers every path including permissions and attachments, so it can all be exercised without an API key.

## License

Apache 2.0. See [LICENSE](LICENSE).
