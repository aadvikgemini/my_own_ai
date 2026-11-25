"""Pydantic request and response models for API validation."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

from backend.config import MAX_MESSAGE_LENGTH

CONVERSATION_ID_REGEX = re.compile(r"^[A-Za-z0-9_-]{32,128}$")


class HealthResponse(BaseModel):
    """Health check response payload."""

    status: Literal["ok"] = "ok"


class NewConversationResponse(BaseModel):
    """Response returned when creating a new conversation."""

    conversation_id: str
    question_limit: int
    questions_used: int
    questions_remaining: int


class ChatRequest(BaseModel):
    """Validated user chat request payload."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: StrictStr
    message: StrictStr

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not CONVERSATION_ID_REGEX.fullmatch(cleaned):
            raise ValueError("Invalid conversation ID format.")
        return cleaned

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("Message contains unsupported control characters.")
        if len(value) > MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"Message exceeds the maximum length of {MAX_MESSAGE_LENGTH} characters."
            )
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty.")
        return cleaned


class ChatResponse(BaseModel):
    """Validated assistant reply payload."""

    conversation_id: str
    response: str
    questions_used: int
    questions_remaining: int
    question_limit: int


class ErrorResponse(BaseModel):
    """Standardized safe error response payload."""

    detail: str
