"""Request identifier propagation for logs and responses."""

from __future__ import annotations

import uuid
from contextvars import ContextVar

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"
_MAX_REQUEST_ID_LENGTH = 200

request_id_var: ContextVar[str] = ContextVar("acf_request_id", default="")


def new_request_id() -> str:
    """Generate a fresh request identifier."""
    return uuid.uuid4().hex


def current_request_id() -> str:
    """Return the request id bound to the current context (empty outside requests)."""
    return request_id_var.get()


class RequestIDMiddleware:
    """Accept or generate ``X-Request-ID`` and echo it on the response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = incoming.strip() if incoming else ""
        if not request_id:
            request_id = new_request_id()
        request_id = request_id[:_MAX_REQUEST_ID_LENGTH]

        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        token = request_id_var.set(request_id)

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            request_id_var.reset(token)
