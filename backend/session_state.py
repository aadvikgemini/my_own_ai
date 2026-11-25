"""HMAC-signed compressed state helper for serverless cold-start continuity.

This complements (and never replaces) backend validation: tokens are verified
with HMAC-SHA256 before decompression, and in-memory state takes precedence
within warm workers to prevent replay of older counters.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import zlib
from typing import Any

MAX_SESSION_HEADER_CHARS = 32_768
MAX_DECOMPRESSED_BYTES = 65_536


def encode_session_state(
    *,
    secret: bytes,
    conversation_id: str,
    questions_used: int,
    messages: list[dict[str, str]],
) -> str:
    """Serialize, compress, and HMAC-sign conversation state for continuity."""
    payload_dict = {
        "cid": conversation_id,
        "q": questions_used,
        "m": messages,
    }
    raw = json.dumps(payload_dict, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    compressed = zlib.compress(raw, level=6)
    encoded_payload = base64.urlsafe_b64encode(compressed).decode("ascii").rstrip("=")
    signature = hmac.new(secret, encoded_payload.encode("ascii"), hashlib.sha256).digest()
    encoded_sig = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return f"{encoded_payload}.{encoded_sig}"


def decode_session_state(
    *,
    secret: bytes,
    token: str,
    expected_conversation_id: str,
    max_questions: int,
    max_history_messages: int,
) -> tuple[int, list[dict[str, str]]] | None:
    """Verify HMAC signature before decompressing and validating session state."""
    if not token or len(token) > MAX_SESSION_HEADER_CHARS or "." not in token:
        return None

    parts = token.split(".", 1)
    if len(parts) != 2:
        return None
    encoded_payload, encoded_sig = parts

    expected_sig = hmac.new(
        secret, encoded_payload.encode("ascii"), hashlib.sha256
    ).digest()
    expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).decode("ascii").rstrip("=")
    if not hmac.compare_digest(encoded_sig, expected_sig_b64):
        return None

    try:
        padded = encoded_payload + "=" * (-len(encoded_payload) % 4)
        compressed = base64.urlsafe_b64decode(padded.encode("ascii"))
        decompressor = zlib.decompressobj()
        decompressed = decompressor.decompress(compressed, MAX_DECOMPRESSED_BYTES + 1)
        if len(decompressed) > MAX_DECOMPRESSED_BYTES or decompressor.unconsumed_tail:
            return None
        data: Any = json.loads(decompressed.decode("utf-8"))
    except Exception:
        return None

    if not isinstance(data, dict):
        return None
    if data.get("cid") != expected_conversation_id:
        return None

    questions_used = data.get("q")
    messages = data.get("m")
    if not isinstance(questions_used, int) or isinstance(questions_used, bool):
        return None
    if questions_used < 0 or questions_used > max_questions:
        return None
    if not isinstance(messages, list) or len(messages) > max_history_messages:
        return None

    validated_messages: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            return None
        validated_messages.append({"role": role, "content": content})

    return questions_used, validated_messages
