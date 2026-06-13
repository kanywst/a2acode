"""Echo backend.

Needs no API key and no Claude install. It exists so the server, the protocol
mapping, and the CLI can be exercised end to end offline. It emits the same
event shapes a real run produces, and when the prompt contains ``sudo`` it asks
for permission first — which lets the full ``input-required`` round trip be
verified without a live Claude session.
"""

from __future__ import annotations

from .base import Result, RunRequest, TextDelta, ToolUse
from .session import BackendSession


class EchoBackend:
    name = "echo"

    async def drive(self, session: BackendSession, request: RunRequest) -> None:
        await session.emit(
            ToolUse(
                name="Echo",
                tool_input={"prompt": request.prompt},
                tool_use_id="echo-1",
            )
        )

        if "sudo" in request.prompt.lower():
            decision = await session.request_permission(
                "Bash", {"command": request.prompt}, f"$ {request.prompt}"
            )
            if not decision.allow:
                await session.emit(TextDelta("permission denied; nothing run"))
                await session.emit(self._result(request))
                return

        for word in request.prompt.split():
            await session.emit(TextDelta(word + " "))
        await session.emit(self._result(request))

    @staticmethod
    def _result(request: RunRequest) -> Result:
        return Result(
            session_id=request.context_id or "echo-session",
            cost_usd=0.0,
            num_turns=1,
        )
