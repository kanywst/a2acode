"""Caller authentication.

Exercised over the real ASGI app with a TestClient so the middleware, the route
exemptions, and the card declaration are all checked together.
"""

from __future__ import annotations

from starlette.testclient import TestClient

from a2acode.backends import make_backend
from a2acode.card import build_card
from a2acode.server import build_app

TOKEN = "s3cret-bearer-token"


def _client(auth_token=None):
    app = build_app(make_backend("echo"), url="http://x/", auth_token=auth_token)
    return TestClient(app)


def test_card_declares_bearer_scheme_when_auth_required():
    card = build_card("http://x/", require_auth=True)
    assert card.security_schemes["bearer"].http_auth_security_scheme.scheme == "bearer"
    assert any("bearer" in req.schemes for req in card.security_requirements)


def test_card_has_no_security_when_auth_disabled():
    card = build_card("http://x/")
    assert not card.security_schemes
    assert not card.security_requirements


def test_card_endpoint_is_public_even_with_auth():
    # Discovery must work without a credential: a caller reads the card to learn
    # which scheme to use before it has a token.
    resp = _client(TOKEN).get("/.well-known/agent-card.json")
    assert resp.status_code == 200
    assert "bearer" in resp.json()["securitySchemes"]


def test_task_endpoint_rejects_missing_token():
    resp = _client(TOKEN).post("/", json={"jsonrpc": "2.0", "id": 1, "method": "x"})
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


def test_task_endpoint_rejects_wrong_token():
    resp = _client(TOKEN).post(
        "/",
        json={"jsonrpc": "2.0", "id": 1, "method": "x"},
        headers={"Authorization": "Bearer nope"},
    )
    assert resp.status_code == 401


def test_correct_token_passes_auth():
    # A valid token gets past the middleware; the request reaches the handler
    # (a malformed RPC body then fails there, which is not a 401).
    resp = _client(TOKEN).post(
        "/",
        json={"jsonrpc": "2.0", "id": 1, "method": "x"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert resp.status_code != 401


def test_no_auth_by_default():
    resp = _client().post("/", json={"jsonrpc": "2.0", "id": 1, "method": "x"})
    assert resp.status_code != 401


def test_middleware_rejects_empty_token():
    import pytest

    from a2acode.auth import BearerAuthMiddleware

    with pytest.raises(ValueError, match="empty"):
        BearerAuthMiddleware(lambda *a: None, token="   ")
