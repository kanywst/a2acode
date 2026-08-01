"""Processes run on the server for an ACP agent.

ACP lets an agent delegate command execution to its client rather than shelling
out itself. Serving it is what makes the delegation safe to accept: the process
starts in the workspace, its output is bounded, and it is reaped with the turn.
The permission gate lives in the backend's client, not here — every terminal is
approved by the A2A caller before this module ever spawns anything.
"""

from __future__ import annotations

import asyncio
import os
import signal as signal_module
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

# Default ceiling on captured output when the agent names none, and the hard
# ceiling it cannot raise past: this buffer lives in the server's memory.
DEFAULT_OUTPUT_LIMIT = 1024 * 1024
MAX_OUTPUT_LIMIT = 8 * 1024 * 1024

# How long a killed process gets to die before we stop waiting on it.
_REAP_TIMEOUT = 5.0


@dataclass
class Terminal:
    """One running command and the tail of its output."""

    process: asyncio.subprocess.Process
    limit: int
    buffer: bytearray = field(default_factory=bytearray)
    truncated: bool = False
    _pump: asyncio.Task[None] | None = None

    @property
    def output(self) -> str:
        # A trim can land mid-codepoint, so decode leniently rather than raise.
        return self.buffer.decode("utf-8", errors="replace")

    async def _read(self) -> None:
        assert self.process.stdout is not None
        while chunk := await self.process.stdout.read(8192):
            self.buffer += chunk
            if len(self.buffer) > self.limit:
                # Keep the tail: the end of a failing build says more than its
                # opening banner. Trimming is by byte, not by read, so one long
                # line cannot take the whole window with it.
                del self.buffer[: len(self.buffer) - self.limit]
                self.truncated = True

    async def wait(self) -> tuple[int | None, str | None]:
        """Wait for exit, returning ``(exit_code, signal_name)``."""
        if self._pump is not None:
            await self._pump
        code = await self.process.wait()
        if code is not None and code < 0:
            return None, _signal_name(-code)
        return code, None

    async def kill(self) -> None:
        if self.process.returncode is None:
            self.process.kill()
        await self.close()

    async def close(self) -> None:
        if self.process.returncode is None:
            self.process.kill()
        if self._pump is not None:
            self._pump.cancel()
            with suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(self._pump, _REAP_TIMEOUT)
        with suppress(TimeoutError):
            await asyncio.wait_for(self.process.wait(), _REAP_TIMEOUT)

    def exit_status(self) -> tuple[int | None, str | None] | None:
        """The exit code/signal, or ``None`` while the process is running."""
        code = self.process.returncode
        if code is None:
            return None
        if code < 0:
            return None, _signal_name(-code)
        return code, None


def _signal_name(number: int) -> str:
    try:
        return signal_module.Signals(number).name
    except ValueError:
        return str(number)


async def spawn(
    command: str,
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    limit: int,
) -> Terminal:
    """Start a command with its output captured up to ``limit`` bytes."""
    process = await asyncio.create_subprocess_exec(
        command,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
        cwd=str(cwd),
        env={**os.environ, **env},
    )
    terminal = Terminal(process=process, limit=limit)
    terminal._pump = asyncio.create_task(terminal._read())
    return terminal
