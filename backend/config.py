"""Backend configuration loaded strictly from environment variables."""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = BACKEND_DIR / ".env"

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = (
    "http://localhost:8501",
    "http://127.0.0.1:8501",
)
MAX_QUESTIONS_PER_CONVERSATION = 20
MAX_MESSAGE_LENGTH = 2000
MAX_REQUEST_BODY_BYTES = 16 * 1024  # 16 KB
MAX_HISTORY_MESSAGES = MAX_QUESTIONS_PER_CONVERSATION * 2
MAX_HISTORY_CHARS = 24_000
MAX_CONVERSATIONS = 1000
OPENAI_TIMEOUT_SECONDS = 30.0
OPENAI_MAX_OUTPUT_TOKENS = 2048
RATE_LIMIT_MAX_REQUESTS = 30
RATE_LIMIT_WINDOW_SECONDS = 60


def parse_allowed_origins(raw_origins: str | None) -> tuple[str, ...]:
    """Parse and validate comma-separated CORS origins, rejecting wildcards."""
    if not raw_origins or not raw_origins.strip():
        return DEFAULT_ALLOWED_ORIGINS

    origins: list[str] = []
    for item in raw_origins.split(","):
        origin = item.strip().rstrip("/")
        if not origin or origin == "*":
            continue
        parsed = urlparse(origin)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            origins.append(f"{parsed.scheme}://{parsed.netloc}")

    return tuple(dict.fromkeys(origins)) or DEFAULT_ALLOWED_ORIGINS


@dataclass(frozen=True)
class Settings:
    """Application settings for the FastAPI backend."""

    openai_api_key: str = field(default="", repr=False)
    openai_model: str = DEFAULT_MODEL
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    session_secret: bytes = field(
        default_factory=lambda: secrets.token_bytes(32), repr=False
    )
    max_questions_per_conversation: int = MAX_QUESTIONS_PER_CONVERSATION
    max_message_length: int = MAX_MESSAGE_LENGTH
    max_request_body_bytes: int = MAX_REQUEST_BODY_BYTES
    max_history_messages: int = MAX_HISTORY_MESSAGES
    max_history_chars: int = MAX_HISTORY_CHARS
    max_conversations: int = MAX_CONVERSATIONS
    openai_timeout_seconds: float = OPENAI_TIMEOUT_SECONDS
    openai_max_output_tokens: int = OPENAI_MAX_OUTPUT_TOKENS
    rate_limit_max_requests: int = RATE_LIMIT_MAX_REQUESTS
    rate_limit_window_seconds: int = RATE_LIMIT_WINDOW_SECONDS


def load_settings(*, load_env_file: bool = True, env_path: Path = DEFAULT_ENV_PATH) -> Settings:
    """Load backend settings from backend/.env and process environment."""
    if load_env_file and env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_MODEL
    raw_origins = os.getenv("ALLOWED_ORIGIN") or os.getenv("ALLOWED_ORIGINS")
    origins = list(parse_allowed_origins(raw_origins))

    for vercel_env_var in (
        "VERCEL_URL",
        "VERCEL_BRANCH_URL",
        "VERCEL_PROJECT_PRODUCTION_URL",
    ):
        host = os.getenv(vercel_env_var, "").strip().removeprefix("https://").removeprefix("http://").strip("/")
        if host and "*" not in host:
            vercel_origin = f"https://{host}"
            if vercel_origin not in origins:
                origins.append(vercel_origin)

    raw_session_secret = os.getenv("SESSION_SECRET", "").strip()
    session_secret = (
        hashlib.sha256(raw_session_secret.encode("utf-8")).digest()
        if raw_session_secret
        else secrets.token_bytes(32)
    )

    return Settings(
        openai_api_key=api_key,
        openai_model=model,
        allowed_origins=tuple(origins),
        session_secret=session_secret,
    )
