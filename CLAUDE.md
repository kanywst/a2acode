# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

a2acode serves a coding agent over the [A2A](https://a2aprotocol.ai/) protocol. Other agents discover it through its agent card and delegate coding work; it drives a real coding-agent session and streams the structured work (tool calls, file diffs, permission requests, cost, session continuity) back — not just flattened text in / text out.

It is a **bridge between two interop standards**: it speaks Zed's [Agent Client Protocol](https://agentclientprotocol.com) (ACP) to the coding agent (Claude Code, Gemini CLI, Codex, OpenHands, ... — a launch-command choice) and A2A to the caller. The default `acp` backend makes the agent vendor-neutral; a `claude` backend (Claude Agent SDK, no subprocess) and an `echo` backend also ship.

## Commands

```bash
uv sync --dev                      # install with dev deps
uv run ruff check src tests        # lint
uv run ruff format src tests       # format (CI runs --check)
uv run mypy                        # type check (src only)
uv run pytest                      # all tests
uv run pytest tests/test_auth.py   # one file
uv run pytest -k permission        # tests matching a name
uv build                           # build the package
```

CI (`.github/workflows/ci.yml`) runs lint, format-check, mypy, pytest, and build on Python 3.13 and 3.14, plus `markdownlint-cli2`. `uv sync --locked` must succeed, so keep `uv.lock` current when changing dependencies.

Run the server end to end without an API key using the `echo` backend:

```bash
uv run a2acode serve --backend echo &
uv run a2acode call "fix the failing test"
```

## Architecture

The core idea: a **backend** drives a coding agent and yields a normalized event stream; the **executor** is the only place that knows A2A. Backends never import the A2A SDK, and the executor never imports an agent SDK (neither the ACP nor the Claude SDK). This split lets a new driver be added without touching the protocol mapping.

Data flows in one direction:

```text
CLI / A2A caller
    -> server.py        Starlette app: JSON-RPC + REST routes, agent card, push, auth
    -> executor.py      ClaudeCodeExecutor: maps backend events <-> A2A task lifecycle
    -> backends/        a backend emits normalized BackendEvents via a BackendSession
        -> acp.py       drives any ACP agent as a subprocess (default; agent-neutral)
        -> claude.py    drives the Claude Agent SDK directly (ClaudeSDKClient)
        -> echo.py      dependency-free mirror, for tests and offline wiring checks
```

The `acp` backend is the headline: ACP's `session/update` stream, diff content, and `session/request_permission` map almost one-to-one onto the event vocabulary below, so vendor-neutrality is a launch-command choice rather than a backend per agent. ACP itself targets human-driven editors; a2acode's value is exposing an ACP agent to *remote A2A callers* with the permission round-trip and cost preserved.

### The event vocabulary (`backends/base.py`)

Every backend speaks the same five events, and the executor maps each onto an A2A surface:

| Backend event       | A2A surface                                             |
| ------------------- | ------------------------------------------------------- |
| `TextDelta`         | a streamed `response` artifact (`append` / `last_chunk`) |
| `ToolUse`           | a `working` status update describing the action          |
| `FileChange`        | a named artifact carrying a unified diff                 |
| `PermissionRequest` | an `input-required` pause the caller answers             |
| `Result`            | cost / turns / usage metadata on the completion message  |

A `Backend` is a `Protocol` (`name` + `async drive(session, request)`), so any object with that shape qualifies.

### Session decoupling (`backends/session.py`)

`BackendSession` is the seam that makes the permission round trip work. The backend's `drive` coroutine runs as a background task pushing events onto a queue; the executor consumes them with `drain()`, which **stops** when it hits a `PermissionRequest`, leaving the background task parked inside `request_permission` awaiting a decision. A later `resolve()` un-parks it. This is what lets one A2A `input-required` round trip span two separate `execute()` calls while the Claude session stays alive in between.

### Continuity and lifecycle (`executor.py`)

- `context_id` -> Claude `session_id`: a new task in the same A2A context resumes the same Claude conversation (`resume`).
- `task_id` -> live `BackendSession`: a follow-up message to a paused task carries the permission decision.
- `task_id` -> `_Stream`: response-stream state kept across pauses so the response stays one artifact and the completion text is whole.
- Both in-memory maps are bounded (`_MAX_CONTEXTS`, `_MAX_LIVE`) with oldest-first eviction so a long-running server can't grow without limit.

### Permissions are the headline behavior

The server does **not** load the developer's personal Claude settings (`setting_sources=[]` by default in `claude.py`), so it has no pre-approved tool allowlist — every tool needing approval routes through the caller as an `input-required` pause instead of being silently skipped. In `executor.py`, an answer in `_ALLOW_WORDS` (or starting with `allow`) approves; anything else denies.

## Conventions

- **Keep the layering intact.** If you reach for `acp`/`claude_agent_sdk` outside their own backend module, or for `a2a.*` inside a backend, that's the wrong layer.
- **The translation functions are pure and side-effect free** — `events_from_message` (claude), `events_from_update` + `select_option` (acp), and `diff.py` — so the protocol mapping is unit-testable without a live agent. Keep them that way; stateful concerns (cost capture, permission parking) live in the backend's `Client`/`drive`, not the translator.
- Python 3.13+, full type hints, `from __future__ import annotations`. Ruff enforces `E, F, I, UP, B, SIM` at line length 88.
- The `acp` and `claude` backends are imported lazily (`make_backend`) so `echo` works without their runtime deps. The Claude SDK is an optional extra (`a2acode[claude]`). New optional backends should follow the same lazy pattern.

## Reference material

`_source/` holds upstream A2A spec and sample adapters for reference. It is **gitignored and not part of this project** — read it to understand the protocol, but never edit it or treat it as code to maintain.
