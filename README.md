<img src="assets/mascot.png" alt="a2claude" width="150" align="right">

# a2claude

Run Claude Code as an [A2A](https://a2aprotocol.ai/) agent server. Other agents
call it over the protocol; it drives a real Claude Code session in your project
and streams back the actual work — the tools it runs, the diffs it writes, what
it costs, and the permissions it needs.

[![CI](https://github.com/kanywst/a2claude/actions/workflows/ci.yml/badge.svg)](https://github.com/kanywst/a2claude/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![Protocol: A2A 1.0](https://img.shields.io/badge/protocol-A2A%201.0-D97757.svg)](https://a2aprotocol.ai/)

![a2claude streaming a task, then pausing on a permission prompt](assets/demo.gif)

Most "wrap a coding agent in A2A" adapters flatten everything to text in, text
out. a2claude keeps the parts that matter for the agent on the other end: which
tools ran, what files changed, what it cost, and how to continue the same
session on the next turn.

## How it maps to A2A

| Claude Code produces | A2A surface it lands on |
| --- | --- |
| Assistant text | A streamed artifact (`append` / `last_chunk`) |
| A tool call (Bash, Edit) | A `working` status update for the action |
| A file edit | A named artifact carrying the diff |
| Run result | Cost, turns, and usage on the completion message |
| Session id | Mapped to the A2A `contextId` so follow-ups resume |

The mapping lives in one place (`executor.py`). Backends only emit
normalized events; they never touch the protocol.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Claude Code CLI on `PATH` (only for the `claude` backend)

## Quick start

Install:

```bash
uv sync
```

The `echo` backend needs no API key and no Claude install, so you can
exercise the whole path offline first:

```bash
uv run a2claude serve --backend echo &
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

| Command | Description |
| --- | --- |
| `a2claude serve` | Start the A2A server |
| `a2claude call TEXT` | Send a message and print the streamed events |
| `a2claude card` | Fetch and print the agent card |

The agent card is served at `/.well-known/agent-card.json` and advertises
Claude Code's abilities as discrete skills (generation, refactor, debug,
review, test, explain).

## Backends

A backend turns a prompt into a stream of normalized events. Two ship today:

- `echo` — no dependencies; mirrors the input. For wiring checks and tests.
- `claude` — drives Claude Code through the Claude Agent SDK.

The split keeps the A2A layer independent of how Claude Code is invoked, so
a raw-CLI backend can be added later without touching the server or the
protocol mapping.

## Authentication

The `claude` backend uses whatever the Claude CLI is configured with. When
the server answers on behalf of other agents, that has to be an Anthropic API
key (or Bedrock / Vertex) — Anthropic does not permit subscription
credentials for third-party serving. Set a per-run cost ceiling with
`--max-budget-usd`.

## Permissions

A tool that needs approval pauses the task in the A2A `input-required` state
instead of being skipped. The caller answers with a follow-up message on the
same task:

```bash
uv run a2claude call "sudo reboot"
# ... [input-required] Permission requested for Bash: $ sudo reboot
#       reply: a2claude call "allow" --task <id> --context <id>
uv run a2claude call "allow" --task <id> --context <id>
```

`allow` (or `yes`, `approve`, `ok`) approves; anything else denies. The Claude
session stays alive across the pause, so it resumes exactly where it stopped.

The server does not inherit your personal Claude settings, so it has no
pre-approved tool allowlist — every tool that needs approval routes through the
caller. Read-only actions Claude already treats as safe still run without a
prompt.

## Long-running tasks

The agent card advertises push notifications. A caller can register a webhook
for a task and receive status and artifact updates by HTTP POST instead of
holding a stream open — useful when a run takes minutes. Streaming and polling
(`tasks/get`) both work too.

## Development

```bash
uv sync --dev
uv run ruff check src tests
uv run ruff format src tests
uv run mypy
uv run pytest
```

CI runs these on Python 3.13 and 3.14, plus a Markdown lint, on every push and
pull request.

## Releasing

Pushing a `v*` tag builds the package, creates a GitHub release with the
artifacts, and publishes to PyPI via trusted publishing:

```bash
git tag v0.1.0
git push origin v0.1.0
```

## Status

The mapping is complete end to end and verified against real Claude: text round
trip, tool-progress updates, streaming artifacts, file diffs as artifacts, run
metadata, session continuity, the permission → `input-required` round trip, and
push notifications. The offline `echo` backend covers every path including
permissions, so it can all be exercised without an API key.

## License

Apache 2.0 — see [LICENSE](LICENSE).
