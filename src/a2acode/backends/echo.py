"""Echo backend.

Needs no API key and no Claude install. It exists so the server, the protocol
mapping, and the CLI can be exercised end to end offline. It emits the same
event shapes a real run produces, and when the prompt contains ``sudo`` it asks
for permission first, which lets the full ``input-required`` round trip be
verified without a live Claude session.
"""

from __future__ import annotations

from .attach import append_to_prompt
from .base import Result, RunRequest, TextDelta, ToolResult, ToolUse
from .session import BackendSession


class EchoBackend:
    name = "echo"

    async def drive(self, session: BackendSession, request: RunRequest) -> None:
        # Attachments are folded in exactly as a real backend does, so the
        # offline path exercises that rendering too.
        prompt = append_to_prompt(request.prompt, request.attachments)
        await session.emit(
            ToolUse(
                name="Echo",
                tool_input={"prompt": prompt},
                tool_use_id="echo-1",
            )
        )

        if "sudo" in prompt.lower():
            decision = await session.request_permission(
                "Bash", {"command": prompt}, f"$ {prompt}"
            )
            if not decision.allow:
                # Relayed the way a real agent uses it, so the offline path shows
                # whether a denial's guidance actually reached the backend.
                reason = decision.message or "no reason given"
                await session.emit(
                    ToolResult(
                        tool_use_id="echo-1",
                        name="Echo",
                        failed=True,
                        output=f"permission denied: {reason}",
                    )
                )
                await session.emit(
                    TextDelta(f"permission denied; nothing run ({reason})")
                )
                await session.emit(self._result(request))
                return

        await session.emit(ToolResult(tool_use_id="echo-1", name="Echo"))
        for word in prompt.split():
            await session.emit(TextDelta(word + " "))
        await session.emit(self._result(request))

    @staticmethod
    def _result(request: RunRequest) -> Result:
        return Result(
            session_id=request.context_id or "echo-session",
            cost_usd=0.0,
            num_turns=1,
        )
