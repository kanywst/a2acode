"""A real ACP agent, small enough to run in CI.

The backend's mapping is unit-tested against fabricated updates, but nothing was
exercising the parts that only exist once a subprocess is on the other end of a
pipe: the handshake, session lifecycle, the permission call coming back the
other way, terminals, and process reuse across turns. This agent speaks actual
ACP over stdio so those can be tested without a vendor, a credential, or a
network.

It is driven by the prompt: each keyword makes it exercise one surface.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import acp
from acp import schema as s

# Sessions this process has opened, so a test can prove the same process served
# two turns rather than a fresh one being spawned per turn.
SESSIONS: list[str] = []


class FakeAgent:
    def __init__(self) -> None:
        self._conn: Any = None
        self._turns = 0

    async def initialize(
        self, protocol_version: int, client_capabilities=None, **_: Any
    ) -> s.InitializeResponse:
        self._client_capabilities = client_capabilities
        return s.InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=s.AgentCapabilities(
                load_session=os.environ.get("FAKE_AGENT_LOAD_SESSION") == "1",
                prompt_capabilities=s.PromptCapabilities(image=True),
            ),
        )

    async def new_session(self, cwd: str, **_: Any) -> s.NewSessionResponse:
        session_id = f"sess-{len(SESSIONS) + 1}"
        SESSIONS.append(session_id)
        return s.NewSessionResponse(session_id=session_id)

    async def load_session(self, cwd: str, session_id: str, **_: Any) -> None:
        SESSIONS.append(f"load:{session_id}")
        return None

    async def prompt(self, session_id: str, prompt: list[Any], **_: Any) -> Any:
        self._turns += 1
        text = " ".join(getattr(b, "text", "") for b in prompt)
        created = None

        await self._update(session_id, acp.update_agent_thought_text("considering"))
        await self._update(
            session_id, acp.update_agent_message_text(f"turn {self._turns}: ")
        )

        if "plan" in text:
            await self._update(
                session_id,
                acp.update_plan(
                    [
                        acp.plan_entry("look", status="completed"),
                        acp.plan_entry("fix", status="in_progress"),
                    ]
                ),
            )

        if "edit" in text:
            await self._update(
                session_id,
                acp.start_tool_call(
                    "t1",
                    "Write calc.py",
                    kind="edit",
                    raw_input={"file_path": "calc.py"},
                    content=[acp.tool_diff_content("calc.py", "b\n", "a\n")],
                ),
            )
            await self._update(
                session_id,
                acp.update_tool_call(
                    "t1",
                    status="completed",
                    content=[acp.tool_content(acp.text_block("written"))],
                ),
            )

        if "peek" in text:
            # A tool call the way a real agent reports one: announced before its
            # arguments are parsed, refined twice, then completed.
            await self._update(
                session_id, acp.start_tool_call("t4", "Read File", kind="read")
            )
            await self._update(
                session_id,
                acp.update_tool_call(
                    "t4", title="Read app.py", raw_input={"file_path": "app.py"}
                ),
            )
            await self._update(session_id, acp.update_tool_call("t4"))
            await self._update(
                session_id, acp.update_tool_call("t4", status="completed")
            )

        if "boom" in text:
            await self._update(
                session_id,
                acp.start_tool_call("t2", "Run tests", kind="execute"),
            )
            await self._update(
                session_id,
                acp.update_tool_call(
                    "t2",
                    status="failed",
                    content=[
                        acp.tool_content(acp.text_block("2 tests failed\ndetail"))
                    ],
                ),
            )

        if "ask" in text:
            outcome = await self._conn.request_permission(
                session_id=session_id,
                tool_call=s.ToolCallUpdate(
                    tool_call_id="t3", title="rm -rf /", kind="execute"
                ),
                options=[
                    s.PermissionOption(option_id="ok", name="Allow", kind="allow_once"),
                    s.PermissionOption(option_id="no", name="Deny", kind="reject_once"),
                ],
            )
            chosen = getattr(outcome.outcome, "option_id", "cancelled")
            await self._update(
                session_id, acp.update_agent_message_text(f"permission={chosen} ")
            )

        if "shell" in text:
            try:
                created = await self._conn.create_terminal(
                    session_id=session_id,
                    command=sys.executable,
                    args=["-c", "print('from the terminal')"],
                )
            except Exception:
                # A real agent carries on after a refusal rather than dying.
                await self._update(
                    session_id, acp.update_agent_message_text("shell refused ")
                )
                created = None
        if "shell" in text and created is not None:
            await self._conn.wait_for_terminal_exit(
                session_id=session_id, terminal_id=created.terminal_id
            )
            out = await self._conn.terminal_output(
                session_id=session_id, terminal_id=created.terminal_id
            )
            await self._conn.release_terminal(
                session_id=session_id, terminal_id=created.terminal_id
            )
            await self._update(
                session_id,
                acp.update_agent_message_text(f"shell said {out.output.strip()} "),
            )

        if "readfile" in text:
            content = await self._conn.read_text_file(
                session_id=session_id, path="calc.py"
            )
            await self._update(
                session_id,
                acp.update_agent_message_text(
                    f"file has {len(content.content)} chars "
                ),
            )

        await self._update(session_id, acp.update_agent_message_text("done"))
        return s.PromptResponse(
            stop_reason="end_turn",
            usage=s.Usage(total_tokens=42, input_tokens=20, output_tokens=22),
        )

    async def cancel(self, session_id: str, **_: Any) -> None:
        return None

    async def _update(self, session_id: str, update: Any) -> None:
        await self._conn.session_update(session_id=session_id, update=update)


async def main() -> None:
    agent = FakeAgent()

    def build(conn: Any) -> FakeAgent:
        # run_agent hands the agent its client connection, which is how the
        # permission request and the terminal calls get back to a2acode.
        agent._conn = conn
        return agent

    await acp.run_agent(build)


if __name__ == "__main__":
    asyncio.run(main())
