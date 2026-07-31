"""Folding caller attachments into an agent prompt.

An A2A message can carry files, structured data, and links alongside its text,
and no coding agent takes those as protocol input, so they are rendered into the
prompt: text inline in a fenced block, everything else as a labelled reference
the agent can act on. A backend that can pass a part natively — an ACP agent
advertising image support — handles that one itself and leaves the rest here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .base import Attachment

_BACKTICKS = re.compile(r"`+")


def _fence(body: str) -> str:
    """A code fence longer than any run of backticks inside ``body``.

    An attached file containing a fence would otherwise close the block early
    and spill its tail into the prompt as instructions.
    """
    longest = max((len(run) for run in _BACKTICKS.findall(body)), default=0)
    return "`" * max(3, longest + 1)


def render(attachment: Attachment) -> str:
    """Render one attachment as prompt text."""
    label = f"{attachment.name} ({attachment.media_type or 'file'})"
    if attachment.uri is not None:
        return f"[attached {label}: {attachment.uri}]"
    if attachment.text is not None:
        note = ", truncated" if attachment.truncated else ""
        fence = _fence(attachment.text)
        return f"[attached {label}{note}]\n{fence}\n{attachment.text}\n{fence}"
    if attachment.data is None:
        return f"[attached {label}: too large to pass on, content dropped]"
    return f"[attached {label}, {len(attachment.data)} bytes: binary, not inlined]"


def append_to_prompt(prompt: str, attachments: Sequence[Attachment]) -> str:
    """Return the prompt with every attachment rendered after it."""
    if not attachments:
        return prompt
    return "\n\n".join([prompt, *(render(a) for a in attachments)]).strip()
