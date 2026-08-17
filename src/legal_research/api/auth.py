"""A shared-secret gate on the API.

The backend was reachable from the open internet with no authentication at all
(Tailscale Funnel on :8447), and every route that drives the model pipeline was
open to anyone holding the URL. This closes that.

**What this is honest about.** A shared key authenticates whoever holds it. It
does not identify users, it cannot be revoked per-client, and it is only as
secret as the least careful place it is stored. That is the right weight of
mechanism for a single-author research tool, and it is the wrong one for a
service with real users -- if this ever grows past one person, replace it rather
than adding roles to it.

**The browser problem, stated plainly.** The frontend is a static bundle served
from a public URL. A key compiled into it is readable by anyone who opens dev
tools, so `NEXT_PUBLIC_*`-style embedding turns this gate into decoration. The
two designs that actually work are: the user pastes the key into the UI and it
lives in their browser's storage, or a server-side proxy holds the key and the
bundle never sees it. This module supports both by accepting a header; it cannot
make either choice for the frontend.

Off by default. An empty key disables the gate entirely, because the common case
is a loopback dev server where a mandatory secret is friction with no benefit --
but :func:`warn_if_unprotected` exists so that choice is never silent on a host
that is actually reachable.
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp

__all__ = ["ApiKeyMiddleware", "EXEMPT_PATHS", "warn_if_unprotected"]

#: Routes that answer without a key.
#:
#: ``/api/health`` only, and only because a liveness probe that needs a secret
#: is a liveness probe nobody wires up. It returns a fixed ``{"status": "ok"}``
#: and discloses nothing. ``/api/config`` is deliberately *not* here: it lists
#: every model the host is running, which is exactly the reconnaissance a public
#: endpoint should not hand out.
EXEMPT_PATHS = frozenset({"/api/health"})


class ApiKeyMiddleware:
    """Require a shared key on ``/api/*``.

    Accepts ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``. The first is
    what most clients and proxies already know how to send; the second is easier
    to paste into a browser fetch, and refusing it would push people toward
    putting the key in a query string, where it lands in every access log.
    """

    def __init__(self, app: ASGIApp, key: str) -> None:
        self.app = app
        self.key = key

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[no-untyped-def]
        if scope["type"] != "http" or not self.key:
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        if not _protected(request):
            await self.app(scope, receive, send)
            return

        if _presented(request) is None or not secrets.compare_digest(
            _presented(request) or "", self.key
        ):
            response = JSONResponse(
                {"detail": "missing or invalid API key"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def _protected(request: Request) -> bool:
    """Whether this request must carry a key.

    ``OPTIONS`` is exempt because it is the CORS preflight, which browsers send
    *without* credentials by design. Rejecting it would make every
    cross-origin request fail at the preflight with a CORS error rather than a
    401, and the real cause -- a missing key on the request that follows --
    would never reach the developer.
    """

    if request.method == "OPTIONS":
        return False
    path = request.url.path
    return path.startswith("/api/") and path not in EXEMPT_PATHS


def _presented(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, credential = header.partition(" ")
    if scheme.lower() == "bearer" and credential:
        return credential
    return request.headers.get("x-api-key") or None


def warn_if_unprotected(host: str, key: str, emit: Callable[[str], None] = print) -> bool:
    """Say so, loudly, when an unauthenticated API is bound off-loopback.

    Returns whether a warning was emitted, so this is testable without capturing
    stdout. Deliberately not an exception: refusing to start would be the wrong
    call for someone deliberately running open on a private LAN, and a backend
    that will not boot is a backend people work around.

    Note the limit of the check. Binding to 127.0.0.1 is *not* proof of safety
    here -- a Tailscale Funnel or any reverse proxy can publish a loopback port
    to the internet, which is exactly how this API came to be publicly reachable
    while bound to localhost. This catches the obvious case; it cannot see the
    proxy in front of it.
    """

    if key:
        return False
    if host in {"127.0.0.1", "::1", "localhost"}:
        return False
    emit(
        f"WARNING: serving on {host} with no LRG_API_KEY set — every /api route "
        f"is open to anyone who can reach this host. Set LRG_API_KEY to require "
        f"a key, or bind to 127.0.0.1."
    )
    return True


AsyncHandler = Callable[[Request], Awaitable[JSONResponse]]
