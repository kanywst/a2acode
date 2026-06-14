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

from a2claude.card import build_card, sign_card, signer_from_key_file
from a2claude.server import build_app

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
    key_file = tmp_path / "card.key"
    key_file.write_text(SECRET)
    signer = signer_from_key_file(str(key_file), kid="k1", alg="HS256")

    # The signer is applied to the card the server publishes.
    from a2claude.backends import make_backend

    app = build_app(make_backend("echo"), url="http://x/", card_signer=signer)
    assert app.routes
