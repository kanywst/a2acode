"""Caller authentication.

A server that answers on behalf of other agents should be able to require a
credential. This is a pure-ASGI middleware (not ``BaseHTTPMiddleware``) so it
passes the request straight through to the inner app when authorized, leaving
streaming and server-sent events untouched; it only short-circuits with a 401
when a token is missing or wrong.

The agent card stays public: a caller fetches it to learn the auth scheme
*before* it has a credential, so discovery paths under ``/.well-known/`` are
exempt while the task endpoints are protected.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Awaitable, Callable

Receive = Callable[[], Awaitable[dict]]
Send = Callable[[dict], Awaitable[None]]
ASGIApp = Callable[[dict, Receive, Send], Awaitable[None]]

_PUBLIC_PREFIXES = ("/.well-known/",)


class BearerAuthMiddleware:
    """Require ``Authorization: Bearer <token>`` on non-discovery requests."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        token: str,
        public_prefixes: tuple[str, ...] = _PUBLIC_PREFIXES,
    ) -> None:
        if not token.strip():
            raise ValueError("auth token must not be empty")
        self.app = app
        # Compare SHA-256 digests rather than the tokens themselves: the
        # constant-time compare is then always over a fixed 32 bytes, so it
        # cannot leak the token length, and the raw secret is not kept around.
        self._token_digest = hashlib.sha256(token.encode("utf-8")).digest()
        self._public = public_prefixes

    async def __call__(self, scope: dict, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self._is_public(scope.get("path", "")):
            await self.app(scope, receive, send)
            return
        if self._authorized(scope):
            await self.app(scope, receive, send)
            return
        await self._reject(send)

    def _is_public(self, path: str) -> bool:
        return any(path.startswith(p) for p in self._public)

    def _authorized(self, scope: dict) -> bool:
        # Scan the headers list for the one we need instead of materializing a
        # dict on every request.
        raw = b""
        for key, value in scope.get("headers") or []:
            if key == b"authorization":
                raw = value
                break
        # split(None, 1) tolerates extra whitespace between scheme and token.
        parts = raw.split(None, 1)
        if len(parts) != 2 or parts[0].lower() != b"bearer":
            return False
        presented = hashlib.sha256(parts[1].strip()).digest()
        return hmac.compare_digest(presented, self._token_digest)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = b'{"error": "unauthorized"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b"Bearer"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
