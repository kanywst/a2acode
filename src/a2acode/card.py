"""Agent card construction.

The card advertises the coding agent's abilities as discrete A2A skills so that
calling agents can route to it deliberately rather than treating it as an opaque
chat box. The name identifies which agent is fronted; the skills are the same
generation/refactor/debug/review/test/explain set regardless of backend.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    SecurityRequirement,
)
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol

# The id used for the bearer scheme in the card's security_schemes map and in
# each security requirement that references it.
BEARER_SCHEME = "bearer"

try:
    # The agent card version tracks the package version, read from installed
    # metadata so there is one source of truth (pyproject) and it cannot drift.
    VERSION = _package_version("a2acode")
except PackageNotFoundError:  # running from a source tree without an install
    VERSION = "0.0.0"

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
    require_auth: bool = False,
) -> AgentCard:
    card = AgentCard(
        name=name,
        description=description
        or "A coding agent on an A2A mesh: generation, refactoring, "
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
    if require_auth:
        # Declare an HTTP bearer scheme and require it, so a caller learns from
        # the card that it must present a token before sending a task. A2A keeps
        # the credential at the HTTP layer; the card only advertises the
        # requirement.
        card.security_schemes[BEARER_SCHEME].http_auth_security_scheme.scheme = "bearer"
        requirement = SecurityRequirement()
        # Reference the bearer scheme with no extra scopes (an empty scope list).
        requirement.schemes[BEARER_SCHEME].SetInParent()
        card.security_requirements.append(requirement)
    return card


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
