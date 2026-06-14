"""Agent card construction.

The card advertises Claude Code's coding abilities as discrete A2A skills so
that calling agents can route to it deliberately rather than treating it as an
opaque chat box.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        ],
        skills=SKILLS,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )


def sign_card(
    card: AgentCard,
    *,
    key: str | bytes,
    kid: str,
    alg: str = "ES256",
    jku: str | None = None,
) -> AgentCard:
    """Attach a JWS signature to the agent card.

    A2A signed Agent Cards let a receiving agent verify the card was issued by
    the domain that owns the key (``kid``), rather than trusting whatever a
    discovery endpoint happened to return. The signature covers the canonical
    card minus the ``signatures`` field, so it can be appended in place.

    ``key`` is the private signing material: a PEM string for asymmetric
    algorithms (ES256, RS256) or a shared secret for HMAC (HS256). ``jku``, if
    given, is the JWK Set URL a verifier can fetch the public key from.
    """
    from a2a.utils.signing import ProtectedHeader, create_agent_card_signer

    header: ProtectedHeader = {"kid": kid, "alg": alg, "jku": jku, "typ": "JOSE"}
    signer = create_agent_card_signer(signing_key=key, protected_header=header)
    return signer(card)


def signer_from_key_file(
    path: str, *, kid: str, alg: str = "ES256", jku: str | None = None
) -> Callable[[AgentCard], AgentCard]:
    """Build a card-signing function that reads its key from ``path``.

    The file holds a PEM private key (asymmetric algorithms) or a raw shared
    secret (HMAC). The key is read once, as bytes so a binary secret does not
    trip a decode error, and the key/algorithm pair is validated up front by
    signing a throwaway card, so a misconfigured key fails at startup rather
    than on the first discovery request.
    """
    key = Path(path).read_bytes().strip()
    if not key:
        raise ValueError(f"signing key file is empty: {path}")

    def signer(card: AgentCard) -> AgentCard:
        return sign_card(card, key=key, kid=kid, alg=alg, jku=jku)

    try:
        signer(build_card("https://validation.invalid/"))
    except Exception as e:  # noqa: BLE001 - any failure here is a key/alg misconfig
        raise ValueError(
            f"cannot sign agent card with the given key and {alg}: {e}"
        ) from e
    return signer
