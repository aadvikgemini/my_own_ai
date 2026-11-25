"""FastAPI backend for Mini AI Assistant."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.chat_service import ChatService, ChatServiceError
from backend.config import Settings, load_settings
from backend.conversation_store import (
    ConversationNotFoundError,
    ConversationStore,
    QuestionLimitReachedError,
)
from backend.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    NewConversationResponse,
)
from backend.session_state import decode_session_state, encode_session_state

FRONTEND_INDEX_PATH = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("mini_ai_assistant.api")


class InMemoryRateLimiter:
    """Simple sliding-window rate limiter suitable only for local development."""

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: int,
        max_clients: int = 1024,
    ) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._max_clients = max_clients
        self._buckets: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Record a request if under limit, else return (False, retry_after_seconds)."""
        now = time.monotonic()
        cutoff = now - self._window_seconds

        with self._lock:
            self._prune_stale_keys(cutoff)
            timestamps = self._buckets.setdefault(key, deque())
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self._max_requests:
                retry_after = max(1, math.ceil(timestamps[0] + self._window_seconds - now))
                return False, retry_after

            timestamps.append(now)
            return True, 0

    def _prune_stale_keys(self, cutoff: float) -> None:
        if len(self._buckets) < self._max_clients:
            return
        stale_keys = [
            client_key
            for client_key, queue in self._buckets.items()
            if not queue or queue[-1] <= cutoff
        ]
        for client_key in stale_keys:
            del self._buckets[client_key]


class RequestSizeLimitMiddleware:
    """ASGI middleware enforcing a strict request body byte limit."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        content_length_raw = headers.get(b"content-length")
        if content_length_raw is not None:
            try:
                content_length = int(content_length_raw.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                response = JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                )
                await response(scope, receive, send)
                return

            if content_length < 0:
                response = JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                )
                await response(scope, receive, send)
                return

            if content_length > self.max_body_bytes:
                response = JSONResponse(
                    status_code=413,
                    content={"detail": "Request body exceeds maximum allowed size."},
                )
                await response(scope, receive, send)
                return

        if scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        received_bytes = 0
        more_body = True
        disconnected = False

        while more_body:
            message = await receive()
            msg_type = message.get("type")
            if msg_type == "http.disconnect":
                disconnected = True
                break
            if msg_type != "http.request":
                continue

            chunk = message.get("body", b"")
            received_bytes += len(chunk)
            if received_bytes > self.max_body_bytes:
                response = JSONResponse(
                    status_code=413,
                    content={"detail": "Request body exceeds maximum allowed size."},
                )
                await response(scope, receive, send)
                return

            if chunk:
                chunks.append(chunk)
            more_body = bool(message.get("more_body", False))

        full_body = b"".join(chunks)
        body_sent = False

        async def replay_receive() -> Message:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                if disconnected and not full_body:
                    return {"type": "http.disconnect"}
                return {"type": "http.request", "body": full_body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


def _get_client_key(request: Request) -> str:
    """Identify local client IP directly without trusting spoofable forwarding headers."""
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _enforce_rate_limit(
    limiter: InMemoryRateLimiter, request: Request, scope_name: str
) -> None:
    allowed, retry_after = limiter.check(f"{scope_name}:{_get_client_key(request)}")
    if not allowed:
        logger.warning("Rate limit exceeded on %s", scope_name)
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment before trying again.",
            headers={"Retry-After": str(retry_after)},
        )


def create_app(
    settings: Settings | None = None,
    chat_service: ChatService | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    resolved_settings = settings or load_settings()
    resolved_chat_service = chat_service or ChatService(resolved_settings)
    store = ConversationStore(
        max_questions=resolved_settings.max_questions_per_conversation,
        max_history_messages=resolved_settings.max_history_messages,
        max_history_chars=resolved_settings.max_history_chars,
        max_conversations=resolved_settings.max_conversations,
    )
    rate_limiter = InMemoryRateLimiter(
        max_requests=resolved_settings.rate_limit_max_requests,
        window_seconds=resolved_settings.rate_limit_window_seconds,
    )

    application = FastAPI(
        title="Mini AI Assistant Backend",
        version="1.0.0",
        description="Local FastAPI backend mediating Streamlit UI and OpenAI Responses API.",
    )
    application.state.settings = resolved_settings
    application.state.store = store
    application.state.chat_service = resolved_chat_service
    application.state.rate_limiter = rate_limiter

    application.add_middleware(
        RequestSizeLimitMiddleware,
        max_body_bytes=resolved_settings.max_request_body_bytes,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Session-State"],
        expose_headers=["X-Session-State"],
    )

    @application.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Summarize validation error without echoing raw input or stack traces.
        first_error = exc.errors()[0] if exc.errors() else {}
        loc_parts = [str(part) for part in first_error.get("loc", ()) if part != "body"]
        field_name = ".".join(loc_parts) if loc_parts else "request"
        error_msg = str(first_error.get("msg", "Invalid input.")).removeprefix(
            "Value error, "
        )
        logger.warning("Request validation failed on field=%s", field_name)
        return JSONResponse(
            status_code=422,
            content={"detail": f"Invalid {field_name}: {error_msg}"},
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error("Unhandled server error: %s", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected internal server error occurred."},
        )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_web_ui() -> HTMLResponse:
        if FRONTEND_INDEX_PATH.is_file():
            return HTMLResponse(
                content=FRONTEND_INDEX_PATH.read_text(encoding="utf-8")
            )
        return HTMLResponse(
            content="<h1>Mini AI Assistant API</h1><p>Backend is running.</p>"
        )

    @application.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.post(
        "/api/chat/new",
        response_model=NewConversationResponse,
        responses={429: {"model": ErrorResponse}},
    )
    async def new_conversation(
        request: Request, response: Response
    ) -> NewConversationResponse:
        _enforce_rate_limit(rate_limiter, request, "new_chat")
        snapshot = store.create_conversation()
        response.headers["X-Session-State"] = encode_session_state(
            secret=resolved_settings.session_secret,
            conversation_id=snapshot.conversation_id,
            questions_used=0,
            messages=[],
        )
        logger.info("Created new conversation")
        return NewConversationResponse(
            conversation_id=snapshot.conversation_id,
            question_limit=snapshot.question_limit,
            questions_used=snapshot.questions_used,
            questions_remaining=snapshot.questions_remaining,
        )

    @application.post(
        "/api/chat",
        response_model=ChatResponse,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
    )
    async def chat(
        request: Request,
        response: Response,
        payload: ChatRequest,
        x_session_state: str | None = Header(default=None, alias="X-Session-State"),
    ) -> ChatResponse:
        _enforce_rate_limit(rate_limiter, request, "chat")

        if len(payload.message) > resolved_settings.max_message_length:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Message exceeds the maximum length of "
                    f"{resolved_settings.max_message_length} characters."
                ),
            )

        try:
            history = store.reserve_turn(payload.conversation_id)
        except ConversationNotFoundError:
            restored = (
                decode_session_state(
                    secret=resolved_settings.session_secret,
                    token=x_session_state,
                    expected_conversation_id=payload.conversation_id,
                    max_questions=resolved_settings.max_questions_per_conversation,
                    max_history_messages=resolved_settings.max_history_messages,
                )
                if x_session_state
                else None
            )
            if restored is None:
                logger.info("Chat request rejected: unknown conversation")
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found. Please start a new chat.",
                ) from None

            restored_q, restored_messages = restored
            store.restore_conversation(
                payload.conversation_id, restored_q, restored_messages
            )
            try:
                history = store.reserve_turn(payload.conversation_id)
            except QuestionLimitReachedError:
                logger.info("Chat request rejected: question limit reached")
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Question limit of {resolved_settings.max_questions_per_conversation} "
                        "reached for this conversation. Please start a new chat."
                    ),
                ) from None
        except QuestionLimitReachedError:
            logger.info("Chat request rejected: question limit reached")
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Question limit of {resolved_settings.max_questions_per_conversation} "
                    "reached for this conversation. Please start a new chat."
                ),
            ) from None

        try:
            reply_text = await run_in_threadpool(
                resolved_chat_service.generate_reply, history, payload.message
            )
        except ChatServiceError as exc:
            store.abort_turn(payload.conversation_id)
            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.user_message,
            ) from None
        except Exception as exc:
            store.abort_turn(payload.conversation_id)
            logger.error("Unexpected error during chat processing: %s", type(exc).__name__)
            raise HTTPException(
                status_code=500,
                detail="An unexpected internal server error occurred.",
            ) from None

        try:
            questions_used, questions_remaining = store.commit_turn(
                payload.conversation_id, payload.message, reply_text
            )
        except ConversationNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Conversation expired before completion. Please start a new chat.",
            ) from None

        exported = store.export_bounded_state(payload.conversation_id)
        if exported is not None:
            exported_q, exported_messages = exported
            response.headers["X-Session-State"] = encode_session_state(
                secret=resolved_settings.session_secret,
                conversation_id=payload.conversation_id,
                questions_used=exported_q,
                messages=exported_messages,
            )

        logger.info(
            "Chat turn completed (questions_used=%d/%d)",
            questions_used,
            resolved_settings.max_questions_per_conversation,
        )
        return ChatResponse(
            conversation_id=payload.conversation_id,
            response=reply_text,
            questions_used=questions_used,
            questions_remaining=questions_remaining,
            question_limit=resolved_settings.max_questions_per_conversation,
        )

    return application


app = create_app()
