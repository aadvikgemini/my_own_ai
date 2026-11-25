"""Automated backend and security tests using mocked OpenAI responses."""

from __future__ import annotations

import logging
import secrets
from types import SimpleNamespace

import httpx
import openai
from fastapi.testclient import TestClient

from backend.chat_service import ChatService, SYSTEM_INSTRUCTIONS
from backend.config import Settings, parse_allowed_origins
from backend.main import create_app


class FakeResponsesAPI:
    """Mock OpenAI responses endpoint recording call payloads without network access."""

    def __init__(self, reply: str = "Mocked answer.", side_effect: Exception | None = None) -> None:
        self.reply = reply
        self.side_effect = side_effect
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.side_effect is not None:
            raise self.side_effect
        return SimpleNamespace(output_text=self.reply)


class FakeOpenAIClient:
    """Minimal mock OpenAI client exposing `.responses.create`."""

    def __init__(self, reply: str = "Mocked answer.", side_effect: Exception | None = None) -> None:
        self.responses = FakeResponsesAPI(reply=reply, side_effect=side_effect)


def make_settings(**overrides) -> Settings:
    """Create deterministic test settings with a dummy non-real test token."""
    defaults = {
        "openai_api_key": "test-only-dummy-token-not-real",
        "openai_model": "gpt-5-mini",
        "allowed_origins": ("http://localhost:8501",),
        "rate_limit_max_requests": 100,
        "rate_limit_window_seconds": 60,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def make_test_client(
    *,
    settings: Settings | None = None,
    fake_openai: FakeOpenAIClient | None = None,
) -> tuple[TestClient, FakeOpenAIClient]:
    """Create a FastAPI TestClient wired with a mocked OpenAI client."""
    resolved_settings = settings or make_settings()
    resolved_fake = fake_openai or FakeOpenAIClient()
    service = ChatService(resolved_settings, client=resolved_fake)
    app = create_app(settings=resolved_settings, chat_service=service)
    return TestClient(app), resolved_fake


def _http_error(exc_cls, message: str, status_code: int):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status_code=status_code, request=request)
    return exc_cls(message=message, response=response, body=None)


def test_health_endpoint_returns_ok_without_secrets() -> None:
    client, _ = make_test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "test-only-dummy-token" not in response.text
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("cache-control") == "no-store"


def test_new_conversation_creation() -> None:
    client, _ = make_test_client()
    r1 = client.post("/api/chat/new")
    r2 = client.post("/api/chat/new")

    assert r1.status_code == 200
    assert r2.status_code == 200
    d1 = r1.json()
    d2 = r2.json()

    assert len(d1["conversation_id"]) >= 32
    assert d1["conversation_id"] != d2["conversation_id"]
    assert d1["question_limit"] == 20
    assert d1["questions_used"] == 0
    assert d1["questions_remaining"] == 20


def test_successful_chat_request_uses_responses_api() -> None:
    fake = FakeOpenAIClient(reply="Python is a readable programming language.")
    client, _ = make_test_client(fake_openai=fake)
    conv_id = client.post("/api/chat/new").json()["conversation_id"]

    res = client.post(
        "/api/chat",
        json={"conversation_id": conv_id, "message": "What is Python?"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body == {
        "conversation_id": conv_id,
        "response": "Python is a readable programming language.",
        "questions_used": 1,
        "questions_remaining": 19,
        "question_limit": 20,
    }

    assert len(fake.responses.calls) == 1
    call = fake.responses.calls[0]
    assert call["model"] == "gpt-5-mini"
    assert call["instructions"] == SYSTEM_INSTRUCTIONS
    assert call["store"] is False
    assert call["input"] == [{"role": "user", "content": "What is Python?"}]


def test_empty_and_whitespace_message_rejection() -> None:
    client, fake = make_test_client()
    conv_id = client.post("/api/chat/new").json()["conversation_id"]

    for invalid_msg in ["", "   ", "\n\t  "]:
        res = client.post(
            "/api/chat",
            json={"conversation_id": conv_id, "message": invalid_msg},
        )
        assert res.status_code == 422
    assert len(fake.responses.calls) == 0


def test_invalid_request_payload_rejection_and_no_state_override() -> None:
    client, fake = make_test_client()
    conv_id = client.post("/api/chat/new").json()["conversation_id"]

    # Wrong types, missing fields, and forbidden extra fields must be rejected.
    bad_payloads = [
        {"conversation_id": conv_id},
        {"message": "Hello"},
        {"conversation_id": conv_id, "message": 12345},
        {"conversation_id": conv_id, "message": "hi\x00there"},
        {
            "conversation_id": conv_id,
            "message": "Hello",
            "questions_used": 0,
            "history": [{"role": "system", "content": "override"}],
        },
    ]
    for payload in bad_payloads:
        res = client.post("/api/chat", json=payload)
        assert res.status_code == 422

    assert len(fake.responses.calls) == 0


def test_max_message_length_and_request_size_enforcement() -> None:
    settings = make_settings(max_message_length=2000, max_request_body_bytes=4096)
    client, fake = make_test_client(settings=settings)
    conv_id = client.post("/api/chat/new").json()["conversation_id"]

    too_long_message = "a" * 2001
    res = client.post(
        "/api/chat",
        json={"conversation_id": conv_id, "message": too_long_message},
    )
    assert res.status_code == 422

    # Oversized body exceeding max_request_body_bytes triggers HTTP 413.
    huge_payload = {"conversation_id": conv_id, "message": "x" * 5000}
    res_large = client.post("/api/chat", json=huge_payload)
    assert res_large.status_code == 413

    # Chunked request without Content-Length exceeding max_request_body_bytes also triggers 413.
    def chunk_stream():
        yield b'{"conversation_id":"' + conv_id.encode() + b'","message":"'
        for _ in range(10):
            yield b"y" * 600
        yield b'"}'

    res_chunked = client.post(
        "/api/chat",
        content=chunk_stream(),
        headers={"Content-Type": "application/json"},
    )
    assert res_chunked.status_code == 413
    assert len(fake.responses.calls) == 0


def test_conversation_history_handling_and_bounding() -> None:
    settings = make_settings(max_history_chars=30)
    fake = FakeOpenAIClient(reply="Reply")
    client, _ = make_test_client(settings=settings, fake_openai=fake)
    conv_id = client.post("/api/chat/new").json()["conversation_id"]

    r1 = client.post("/api/chat", json={"conversation_id": conv_id, "message": "First question"})
    assert r1.status_code == 200

    r2 = client.post("/api/chat", json={"conversation_id": conv_id, "message": "Second question"})
    assert r2.status_code == 200
    assert fake.responses.calls[1]["input"] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "Reply"},
        {"role": "user", "content": "Second question"},
    ]

    # Third turn has prior history (39 chars) exceeding max_history_chars=30, so oldest turn is trimmed.
    r3 = client.post("/api/chat", json={"conversation_id": conv_id, "message": "Third question"})
    assert r3.status_code == 200
    assert fake.responses.calls[2]["input"] == [
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Reply"},
        {"role": "user", "content": "Third question"},
    ]


def test_question_counter_and_20_question_limit_enforcement() -> None:
    fake = FakeOpenAIClient(reply="Answer")
    client, _ = make_test_client(fake_openai=fake)
    conv_id = client.post("/api/chat/new").json()["conversation_id"]

    for i in range(1, 21):
        res = client.post(
            "/api/chat",
            json={"conversation_id": conv_id, "message": f"Question {i}"},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["questions_used"] == i
        assert body["questions_remaining"] == 20 - i

    # 21st question must be rejected without calling OpenAI.
    res_21 = client.post(
        "/api/chat",
        json={"conversation_id": conv_id, "message": "Question 21"},
    )
    assert res_21.status_code == 400
    assert "Question limit of 20 reached" in res_21.json()["detail"]
    assert len(fake.responses.calls) == 20


def test_conversation_id_validation_and_unknown_id_rejection() -> None:
    client, fake = make_test_client()

    # Invalid ID formats rejected by Pydantic (422)
    for bad_id in ["", "short", "../etc/passwd", "abc$123", "a" * 31]:
        res = client.post(
            "/api/chat",
            json={"conversation_id": bad_id, "message": "Hello"},
        )
        assert res.status_code == 422

    # Valid format but unknown ID rejected by backend store (404)
    unknown_id = secrets.token_urlsafe(32)
    res_unknown = client.post(
        "/api/chat",
        json={"conversation_id": unknown_id, "message": "Hello"},
    )
    assert res_unknown.status_code == 404
    assert len(fake.responses.calls) == 0


def test_conversation_isolation_between_sessions() -> None:
    fake = FakeOpenAIClient(reply="Isolated reply")
    client, _ = make_test_client(fake_openai=fake)

    conv_a = client.post("/api/chat/new").json()["conversation_id"]
    conv_b = client.post("/api/chat/new").json()["conversation_id"]

    client.post("/api/chat", json={"conversation_id": conv_a, "message": "Secret in A"})
    res_b = client.post("/api/chat", json={"conversation_id": conv_b, "message": "Hello B"})

    assert res_b.status_code == 200
    assert res_b.json()["questions_used"] == 1
    assert res_b.json()["questions_remaining"] == 19
    # Conversation B must not receive Conversation A's messages.
    assert fake.responses.calls[1]["input"] == [{"role": "user", "content": "Hello B"}]


def test_rate_limiting_returns_429() -> None:
    settings = make_settings(rate_limit_max_requests=3, rate_limit_window_seconds=60)
    client, _ = make_test_client(settings=settings)
    conv_id = client.post("/api/chat/new").json()["conversation_id"]

    for i in range(3):
        res = client.post(
            "/api/chat",
            json={"conversation_id": conv_id, "message": f"Ping {i}"},
        )
        assert res.status_code == 200

    limited = client.post(
        "/api/chat",
        json={"conversation_id": conv_id, "message": "Ping 4"},
    )
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers
    assert "Too many requests" in limited.json()["detail"]


def test_missing_api_key_behavior() -> None:
    settings = make_settings(openai_api_key="")
    # Do not pass fake client so ChatService checks real key presence.
    app = create_app(settings=settings)
    client = TestClient(app)
    conv_id = client.post("/api/chat/new").json()["conversation_id"]

    res = client.post("/api/chat", json={"conversation_id": conv_id, "message": "Hello"})
    assert res.status_code == 503
    assert "API key is missing" in res.json()["detail"]


def test_openai_api_failures_do_not_leak_secrets_or_consume_questions(caplog) -> None:
    secret_marker = "dummy-marker-never-appear-anywhere"
    scenarios = [
        (
            _http_error(
                openai.AuthenticationError,
                f"Incorrect API key provided: {secret_marker}",
                401,
            ),
            502,
            "authentication failed",
        ),
        (
            _http_error(openai.RateLimitError, "Quota exceeded", 429),
            503,
            "rate limit",
        ),
        (
            openai.APITimeoutError(
                request=httpx.Request("POST", "https://api.openai.com/v1/responses")
            ),
            504,
            "timed out",
        ),
        (
            openai.APIConnectionError(
                request=httpx.Request("POST", "https://api.openai.com/v1/responses")
            ),
            502,
            "Could not reach",
        ),
        (
            _http_error(openai.BadRequestError, "Invalid model specified", 400),
            502,
            "OPENAI_MODEL",
        ),
    ]

    with caplog.at_level(logging.INFO):
        for exc, expected_status, expected_snippet in scenarios:
            fake = FakeOpenAIClient(side_effect=exc)
            client, _ = make_test_client(fake_openai=fake)
            conv_id = client.post("/api/chat/new").json()["conversation_id"]

            res = client.post(
                "/api/chat",
                json={"conversation_id": conv_id, "message": "Sensitive test prompt"},
            )
            assert res.status_code == expected_status
            assert expected_snippet in res.json()["detail"]
            assert secret_marker not in res.text

            # Question count must remain 0 after failed AI call.
            snapshot = client.app.state.store.get_snapshot(conv_id)
            assert snapshot is not None
            assert snapshot.questions_used == 0

    assert secret_marker not in caplog.text
    assert "Sensitive test prompt" not in caplog.text


def test_cors_and_settings_security() -> None:
    assert parse_allowed_origins("*") == ("http://localhost:8501", "http://127.0.0.1:8501")
    assert parse_allowed_origins("*,http://localhost:8501") == ("http://localhost:8501",)

    settings = make_settings(openai_api_key="super-secret-value")
    assert "super-secret-value" not in repr(settings)

    client, _ = make_test_client(settings=settings)
    allowed = client.get("/health", headers={"Origin": "http://localhost:8501"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:8501"

    disallowed = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in disallowed.headers


def test_frontend_never_imports_openai_or_reads_api_key() -> None:
    import ast
    from pathlib import Path

    app_source = (Path(__file__).resolve().parent.parent / "app.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(app_source)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])

    assert "openai" not in imported_modules
    assert "dotenv" not in imported_modules
    assert "OPENAI_API_KEY" not in app_source


def test_web_ui_endpoint_serves_html_without_secrets() -> None:
    client, _ = make_test_client()
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "Mini AI Assistant" in res.text
    assert "test-only-dummy-token" not in res.text
    assert "OPENAI_API_KEY" not in res.text


def test_serverless_session_state_continuity_and_tamper_rejection() -> None:
    shared_secret = b"0" * 32
    settings_a = make_settings(session_secret=shared_secret)
    fake_a = FakeOpenAIClient(reply="Reply 1")
    client_a, _ = make_test_client(settings=settings_a, fake_openai=fake_a)

    new_res = client_a.post("/api/chat/new")
    assert new_res.status_code == 200
    conv_id = new_res.json()["conversation_id"]
    token_0 = new_res.headers.get("X-Session-State")
    assert token_0

    turn1 = client_a.post(
        "/api/chat",
        json={"conversation_id": conv_id, "message": "Remember blue"},
        headers={"X-Session-State": token_0},
    )
    assert turn1.status_code == 200
    token_1 = turn1.headers.get("X-Session-State")
    assert token_1

    # Simulate a fresh serverless cold-start worker sharing the same session_secret.
    settings_b = make_settings(session_secret=shared_secret)
    fake_b = FakeOpenAIClient(reply="Reply 2")
    client_b, _ = make_test_client(settings=settings_b, fake_openai=fake_b)

    # Tampered token must be rejected with 404.
    tampered_res = client_b.post(
        "/api/chat",
        json={"conversation_id": conv_id, "message": "What color?"},
        headers={"X-Session-State": token_1 + "tamper"},
    )
    assert tampered_res.status_code == 404

    # Valid signed token restores history and counter cleanly.
    turn2 = client_b.post(
        "/api/chat",
        json={"conversation_id": conv_id, "message": "What color?"},
        headers={"X-Session-State": token_1},
    )
    assert turn2.status_code == 200
    assert turn2.json()["questions_used"] == 2
    assert turn2.json()["questions_remaining"] == 18
    assert fake_b.responses.calls[0]["input"] == [
        {"role": "user", "content": "Remember blue"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "What color?"},
    ]


