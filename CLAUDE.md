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
        -> attach.py    renders caller attachments into a prompt (shared, pure)
        -> terminal.py  processes run on the server for an ACP agent
```

The `acp` backend is the headline: ACP's `session/update` stream, diff content, and `session/request_permission` map almost one-to-one onto the event vocabulary below, so vendor-neutrality is a launch-command choice rather than a backend per agent. ACP itself targets human-driven editors; a2acode's value is exposing an ACP agent to *remote A2A callers* with the permission round-trip and cost preserved.

### The event vocabulary (`backends/base.py`)

Every backend speaks the same event vocabulary, and the executor maps each onto an A2A surface:

| Backend event       | A2A surface                                              |
| ------------------- | -------------------------------------------------------- |
| `TextDelta`         | a streamed `response` artifact (`append` / `last_chunk`) |
| `Thought`           | a separate `thinking` artifact, never the answer         |
| `ToolUse`           | a `working` status update describing the action          |
| `ToolResult`        | a `working` status update carrying the outcome           |
| `FileChange`        | a named artifact carrying a unified diff                 |
| `Plan`              | a `plan` artifact, replaced on each update               |
| `Notice`            | a `working` status update about the run itself           |
| `PermissionRequest` | an `input-required` pause the caller answers             |
| `Result`            | cost / turns / usage / stop reason on the completion     |

`ToolResult` is optional in both protocols — an agent may never report a terminal status — so the executor treats a missing one as "no outcome reported" rather than assuming success, and resolves the tool's name against the earlier `ToolUse`. `Notice` is the one event the agent does not produce: it is the server telling the caller it had to run the turn differently than asked (a resume the agent cannot honour).

`Plan` is the one event the `claude` backend has to build rather than translate: Claude's task list *is* the plan and it arrives as tool calls that change one entry at a time, so `PlanTracker` accumulates it across the run, and across turns in a context when the turn resumes (bounded by `_MAX_PLANS`), since a `TaskUpdate` addresses a task by an id only the turn that created it saw. A change lands when its call has succeeded — a denied one would otherwise leave the caller reading a plan the agent never took on — and a created task's id comes back in that same result rather than in the call. `TodoWrite`, which wrote the whole list in one call, still works for older CLIs.

`RunRequest` carries the caller's `Attachment`s alongside the prompt; `attach.py` renders them into prompt text, and a backend that can pass a part natively (an ACP agent advertising image support) handles that one itself first.

A `Backend` is a `Protocol` (`name` + `async drive(session, request)`), so any object with that shape qualifies. A backend that holds resources across runs also implements `ClosableBackend` (`aclose`), which `build_app` calls on shutdown.

### Session decoupling (`backends/session.py`)

`BackendSession` is the seam that makes the permission round trip work. The backend's `drive` coroutine runs as a background task pushing events onto a queue; the executor consumes them with `drain()`, which **stops** when it hits a `PermissionRequest`, leaving the background task parked inside `request_permission` awaiting a decision. A later `resolve()` un-parks it. This is what lets one A2A `input-required` round trip span two separate `execute()` calls while the Claude session stays alive in between.

Shutdown runs the seam the other way. `close()` first calls whatever the backend registered with `set_canceller()` — for ACP, `session/cancel` — and gives the run a bounded chance to wind down before cancelling the runner task. A backend that registers nothing falls through to the same hard cancel as before.

### The ACP agent pool (`backends/acp.py`)

An agent subprocess is kept alive per A2A context, not per turn: spawning one costs a process launch, an ACP handshake, and a `session/load`, all of which a follow-up turn in the same conversation skips by talking to the process that already holds it. Consequences worth knowing before touching it:

- One `_BridgeClient` serves a connection for its whole life, so it is **rebound per turn** (`bind`/`unbind`). Between turns nothing is bound and stray agent output is dropped — there is no caller to route it to, and a permission request in that window is refused rather than auto-approved.
- Each `_Agent` holds a lock for the whole turn, *including while parked on a permission*, because one ACP connection carries one conversation.
- The pool is bounded (`_MAX_AGENTS`) and evicts the least recently used **idle** agent. A busy one is never evicted, so the pool overshoots rather than killing a running task's agent.
- Any exception during a turn marks the agent `broken`; the next `_acquire` replaces it instead of resuming on a connection of unknown state.
- A tool call is folded across its own updates (`_ToolCalls`). ACP announces one before it has parsed the arguments and fills them in later, absent meaning unchanged, so the `ToolUse` waits until the call can say what it acts on and later updates are completed from what the call already said. One never announced by end of turn is flushed rather than dropped. A consequence worth knowing: `session/request_permission` carries the arguments itself, so for an agent that announces a call before parsing them the `input-required` pause now precedes the call's `ToolUse`.
- The library hands each notification to its own task, so **`_BridgeClient.session_update` holds a lock** for the whole translation: text chunks are only an answer in the order they were sent, and nothing else orders two updates against each other. Completion is the library's job since 0.12.1 — `prompt` waits for the session's in-flight updates before returning, so the turn no longer ends with the agent's last words still in flight.
- Terminals (`terminal.py`) are gated on the **same caller approval as any tool**. That gate is why advertising the capability is safe: `terminal/create` is a direct client call, and nothing in the protocol obliges an agent to ask permission first. The approval is the boundary — the process is *not* sandboxed, it runs as the server's user and the workspace only sets its cwd. It inherits a named allowlist of environment variables rather than the server's whole environment, so the provider credentials the adapter is launched with are not handed to an agent-chosen command. Do not weaken either the gate or that allowlist without saying so in the README's Permissions section, which states both as guarantees.

### Testing the ACP path

`tests/fake_agent.py` is a real ACP agent (~150 lines) that a2acode launches as a subprocess in `tests/test_live_acp.py`. It covers what only exists once a pipe is involved — handshake, session lifecycle, permission and terminal calls arriving *from* the agent, process reuse across turns — with no vendor, credential, or network. Reach for it when changing anything in `acp.py`; the unit tests mock one side and will not catch an ordering or lifecycle bug.

### Continuity and lifecycle (`executor.py`)

- `context_id` -> Claude `session_id`: a new task in the same A2A context resumes the same Claude conversation (`resume`).
- `task_id` -> live `BackendSession`: a follow-up message to a paused task carries the permission decision.
- `task_id` -> `_Stream`: response-stream state kept across pauses so the response stays one artifact and the completion text is whole.
- Both in-memory maps are bounded (`_MAX_CONTEXTS`, `_MAX_LIVE`) with oldest-first eviction so a long-running server can't grow without limit.

### Permissions are the headline behavior

The server does **not** load the developer's personal Claude settings (`setting_sources=[]` by default in `claude.py`), so it has no pre-approved tool allowlist — every tool needing approval routes through the caller as an `input-required` pause instead of being silently skipped. In `executor.py`, an answer that is entirely one of `_ALLOW_WORDS` approves; anything else denies, and carries its own words to the agent as the reason. The match is on the whole answer on purpose: a prefix match read "allowing that would drop the database, so no" as consent, and prose denials are the documented way to answer. An answer of `option:<id>` naming one the agent offered selects that option instead, so a three-way gate (plan mode's accept-edits / keep-gating / keep-planning) is not flattened to a bool. That prefix is required for the same reason the allow match is strict: the agent picks both an option's id and its polarity, so matching bare text would let it label an `allow_always` choice "Deny" and turn a refusal into consent. An answer is bound to the prompt it names (`requestId`, echoed from the pause's metadata) so a resend cannot settle the request a run stopped on next; a misdirected one restates what is pending rather than deciding it.

Not every gate asks to *act*: `AskUserQuestion` only asks, so an approval alone tells the agent nobody replied. The reply's `a2acode_permission.answers` metadata (question -> choice, or a list of them) rides on the approval as `PermissionDecision.answers`, and `allowed_input` in `claude.py` folds it into the `updated_input` the SDK expects. The verdict still comes from the text alone — the metadata is a payload, never consent — because the answer to a question is prose, and prose is how a caller refuses. Every allow now echoes the tool's input back, which an older CLI requires and which the bare `PermissionResultAllow()` did not.

## Conventions

- **Keep the layering intact.** If you reach for `acp`/`claude_agent_sdk` outside their own backend module, or for `a2a.*` inside a backend, that's the wrong layer.
- **The translation functions are pure and side-effect free** — `events_from_message` (claude), `events_from_update` + `select_option` + `prompt_blocks` (acp), `diff.py`, and `attach.py` — so the protocol mapping is unit-testable without a live agent. Keep them that way; stateful concerns (cost capture, permission parking, the agent pool) live in the backend's `Client`/`drive`, not the translator.
- Python 3.13+, full type hints, `from __future__ import annotations`. Ruff enforces `E, F, I, UP, B, SIM` at line length 88.
- The `acp` and `claude` backends are imported lazily (`make_backend`) so `echo` works without their runtime deps. The Claude SDK is an optional extra (`a2acode[claude]`). New optional backends should follow the same lazy pattern.

## Reference material

`_source/` holds upstream A2A spec and sample adapters for reference. It is **gitignored and not part of this project** — read it to understand the protocol, but never edit it or treat it as code to maintain.
