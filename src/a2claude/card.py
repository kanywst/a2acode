"""Agent card construction.

The card advertises Claude Code's coding abilities as discrete A2A skills so
that calling agents can route to it deliberately rather than treating it as an
opaque chat box.
"""

from __future__ import annotations

from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol

VERSION = "0.1.0"

SKILLS = [
    AgentSkill(
        id="code-generation",
        name="Code generation",
        description="Implement features, scaffold modules, and write new code "
        "from a natural-language description.",
        tags=["code", "generation"],
        examples=["Add a /health endpoint that returns build info"],
    ),
    AgentSkill(
        id="refactor",
        name="Refactoring",
        description="Restructure existing code without changing behavior: "
        "extract functions, rename, split modules.",
        tags=["code", "refactor"],
        examples=["Split this 400-line file into cohesive modules"],
    ),
    AgentSkill(
        id="debug",
        name="Debugging",
        description="Reproduce, locate, and fix defects, then verify the fix.",
        tags=["code", "debug"],
        examples=["The auth test fails intermittently; find and fix it"],
    ),
    AgentSkill(
        id="review",
        name="Code review",
        description="Review a diff or file for correctness, edge cases, and "
        "consistency with the surrounding code.",
        tags=["code", "review"],
        examples=["Review the changes on this branch"],
    ),
    AgentSkill(
        id="test",
        name="Testing",
        description="Write or extend tests and run them to confirm they pass.",
        tags=["code", "test"],
        examples=["Add unit tests for the payment module"],
    ),
    AgentSkill(
        id="explain",
        name="Code explanation",
        description="Explain how a codebase, file, or function works.",
        tags=["code", "explain"],
        examples=["Walk me through how request routing works here"],
    ),
]


def build_card(
    url: str,
    *,
    name: str = "Claude Code",
    description: str | None = None,
    streaming: bool = True,
    push_notifications: bool = True,
) -> AgentCard:
    return AgentCard(
        name=name,
        description=description
        or "Claude Code as an A2A agent: generation, refactoring, "
        "debugging, review, testing, and explanation over a real project "
        "workspace.",
        version=VERSION,
        capabilities=AgentCapabilities(
            streaming=streaming,
            push_notifications=push_notifications,
        ),
        # The server mounts both JSON-RPC and REST (HTTP+JSON) routes, so the
        # card advertises both bindings; a caller picks whichever its client
        # speaks instead of assuming JSON-RPC.
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            ),
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.HTTP_JSON,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            ),
        ],
        skills=SKILLS,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )
