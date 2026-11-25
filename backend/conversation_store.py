"""Thread-safe in-memory conversation store with bounded history and turn limits."""

from __future__ import annotations

import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass, field


class ConversationNotFoundError(Exception):
    """Raised when a conversation ID does not exist on the backend."""


class QuestionLimitReachedError(Exception):
    """Raised when a conversation has reached its maximum question allowance."""


@dataclass(frozen=True)
class ConversationSnapshot:
    """Public summary of a conversation's state."""

    conversation_id: str
    question_limit: int
    questions_used: int
    questions_remaining: int


@dataclass
class _Conversation:
    """Internal mutable state for a single conversation."""

    conversation_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    questions_used: int = 0
    pending_turns: int = 0


class ConversationStore:
    """In-memory store isolating conversations by CSPRNG-generated IDs."""

    def __init__(
        self,
        *,
        max_questions: int = 20,
        max_history_messages: int = 40,
        max_history_chars: int = 24_000,
        max_conversations: int = 1000,
    ) -> None:
        self._max_questions = max_questions
        self._max_history_messages = max_history_messages
        self._max_history_chars = max_history_chars
        self._max_conversations = max_conversations
        self._conversations: OrderedDict[str, _Conversation] = OrderedDict()
        self._lock = threading.Lock()

    def create_conversation(self) -> ConversationSnapshot:
        """Create a new isolated conversation with a secure random ID."""
        with self._lock:
            while len(self._conversations) >= self._max_conversations:
                self._conversations.popitem(last=False)

            conversation_id = secrets.token_urlsafe(32)
            while conversation_id in self._conversations:
                conversation_id = secrets.token_urlsafe(32)

            self._conversations[conversation_id] = _Conversation(
                conversation_id=conversation_id
            )
            return ConversationSnapshot(
                conversation_id=conversation_id,
                question_limit=self._max_questions,
                questions_used=0,
                questions_remaining=self._max_questions,
            )

    def reserve_turn(self, conversation_id: str) -> list[dict[str, str]]:
        """Atomically reserve one question slot and return bounded prior history."""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                raise ConversationNotFoundError("Conversation not found.")

            if conv.questions_used + conv.pending_turns >= self._max_questions:
                raise QuestionLimitReachedError(
                    f"Question limit of {self._max_questions} reached for this conversation."
                )

            conv.pending_turns += 1
            self._conversations.move_to_end(conversation_id)
            return self._build_bounded_history(conv.messages)

    def commit_turn(
        self, conversation_id: str, user_message: str, assistant_reply: str
    ) -> tuple[int, int]:
        """Persist a completed Q&A turn and release its reservation."""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                raise ConversationNotFoundError("Conversation not found.")

            conv.pending_turns = max(0, conv.pending_turns - 1)
            conv.questions_used += 1
            conv.messages.append({"role": "user", "content": user_message})
            conv.messages.append({"role": "assistant", "content": assistant_reply})

            if len(conv.messages) > self._max_history_messages:
                conv.messages = conv.messages[-self._max_history_messages :]

            self._conversations.move_to_end(conversation_id)
            remaining = max(0, self._max_questions - conv.questions_used)
            return conv.questions_used, remaining

    def abort_turn(self, conversation_id: str) -> None:
        """Release a reserved slot when an upstream API call fails."""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is not None:
                conv.pending_turns = max(0, conv.pending_turns - 1)

    def restore_conversation(
        self,
        conversation_id: str,
        questions_used: int,
        messages: list[dict[str, str]],
    ) -> None:
        """Restore verified conversation state on cold-start only if not already in memory."""
        with self._lock:
            if conversation_id in self._conversations:
                return
            while len(self._conversations) >= self._max_conversations:
                self._conversations.popitem(last=False)
            bounded = self._build_bounded_history(messages)
            self._conversations[conversation_id] = _Conversation(
                conversation_id=conversation_id,
                messages=bounded,
                questions_used=min(max(0, questions_used), self._max_questions),
            )

    def export_bounded_state(
        self, conversation_id: str
    ) -> tuple[int, list[dict[str, str]]] | None:
        """Export bounded state for HMAC-signed serverless continuity."""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return None
            return conv.questions_used, self._build_bounded_history(conv.messages)

    def get_snapshot(self, conversation_id: str) -> ConversationSnapshot | None:
        """Return current counters for a conversation, or None if missing."""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return None
            return ConversationSnapshot(
                conversation_id=conv.conversation_id,
                question_limit=self._max_questions,
                questions_used=conv.questions_used,
                questions_remaining=max(0, self._max_questions - conv.questions_used),
            )

    def _build_bounded_history(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Return recent messages capped by both message count and total characters."""
        recent = messages[-self._max_history_messages :]
        selected: list[dict[str, str]] = []
        total_chars = 0

        for msg in reversed(recent):
            msg_len = len(msg["content"])
            if total_chars + msg_len > self._max_history_chars:
                # Stop at pair boundaries when possible so turns remain aligned.
                if msg["role"] == "user" and selected and selected[-1]["role"] == "assistant":
                    selected.pop()
                break
            selected.append({"role": msg["role"], "content": msg["content"]})
            total_chars += msg_len

        selected.reverse()
        if selected and selected[0]["role"] == "assistant":
            selected = selected[1:]
        return selected
