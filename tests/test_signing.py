"""Agent card signing.

Uses HMAC (HS256) with a shared secret so the round trip needs no key
generation; the same code path applies an asymmetric PEM key in production.
"""

from __future__ import annotations

import pytest
from a2a.utils.signing import (
    InvalidSignaturesError,
    create_signature_verifier,
)

from a2acode.card import build_card, sign_card, signer_from_key_file
from a2acode.server import build_app

SECRET = "a-shared-secret-at-least-32-bytes-long!!"


def _verifier(secret: str = SECRET):
    return create_signature_verifier(
        key_provider=lambda kid, jku: secret, algorithms=["HS256"]
    )


def test_sign_card_round_trips():
    card = build_card("http://localhost:9100/")
    signed = sign_card(card, key=SECRET, kid="k1", alg="HS256")

    assert len(signed.signatures) == 1
    _verifier()(signed)  # does not raise


def test_verify_rejects_wrong_key():
    signed = sign_card(build_card("http://x/"), key=SECRET, kid="k1", alg="HS256")
    with pytest.raises(InvalidSignaturesError):
        _verifier("a-different-secret-also-32-bytes-long!")(signed)


def test_signer_from_key_file(tmp_path):
    key_file = tmp_path / "card.key"
    key_file.write_text(SECRET + "\n")

    signer = signer_from_key_file(str(key_file), kid="k1", alg="HS256")
    signed = signer(build_card("http://localhost:9100/"))

    _verifier()(signed)


def test_build_app_signs_the_served_card(tmp_path):
    from starlette.testclient import TestClient

    from a2acode.backends import make_backend

    key_file = tmp_path / "card.key"
    key_file.write_text(SECRET)
    signer = signer_from_key_file(str(key_file), kid="k1", alg="HS256")

    app = build_app(make_backend("echo"), url="http://x/", card_signer=signer)

    # Fetch the card the server actually publishes and confirm the signer was
    # applied: an unsigned card would have no signatures here.
    resp = TestClient(app).get("/.well-known/agent-card.json")
    assert resp.status_code == 200
    signatures = resp.json()["signatures"]
    assert len(signatures) == 1
    assert signatures[0]["protected"] and signatures[0]["signature"]


def test_served_card_verifies_after_compat_fields():
    from a2a.client.card_resolver import parse_agent_card
    from starlette.testclient import TestClient

    from a2acode.backends import make_backend

    app = build_app(
        make_backend("echo"),
        url="http://x/",
        card_signer=lambda card: sign_card(card, key=SECRET, kid="k1", alg="HS256"),
    )
    served = TestClient(app).get("/.well-known/agent-card.json").json()
    # The server merges v0.3 fields into the JSON after signing; a v1.0 client
    # must still verify the signature over what it parses back out.
    assert served["url"] == "http://x/"
    _verifier()(parse_agent_card(served))


def test_signer_rejects_empty_key_file(tmp_path):
    key_file = tmp_path / "empty.key"
    key_file.write_text("   \n")
    with pytest.raises(ValueError, match="empty"):
        signer_from_key_file(str(key_file), kid="k1", alg="HS256")


def test_signer_fails_fast_on_bad_key_alg(tmp_path):
    # A shared secret is not a valid ES256 key; this must fail when the signer
    # is built (at startup), not later on the first request.
    key_file = tmp_path / "card.key"
    key_file.write_text(SECRET)
    with pytest.raises(ValueError, match="cannot sign"):
        signer_from_key_file(str(key_file), kid="k1", alg="ES256")
