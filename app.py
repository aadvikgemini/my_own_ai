"""Streamlit frontend for Mini AI Assistant.

This interface communicates exclusively with the local FastAPI backend over HTTP
and never loads or accesses OpenAI API keys.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import httpx
import streamlit as st

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_QUESTION_LIMIT = 20
MAX_MESSAGE_LENGTH = 2000
REQUEST_TIMEOUT_SECONDS = 35.0


def get_backend_url() -> str:
    """Resolve and validate the backend base URL from environment configuration."""
    raw = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL).strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return DEFAULT_BACKEND_URL


def apply_dark_theme() -> None:
    """Apply clean, modern dark styling using static CSS only."""
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: #e6edf3;
        }
        [data-testid="stHeader"] {
            background-color: rgba(14, 17, 23, 0.85);
        }
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div {
            background-color: #0e1117 !important;
        }
        .block-container {
            max-width: 780px;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }
        .assistant-header {
            padding: 1rem 1.25rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #161b22 0%, #1c2330 100%);
            border: 1px solid #30363d;
            margin-bottom: 1rem;
        }
        .assistant-title {
            margin: 0;
            font-size: 1.75rem;
            font-weight: 700;
            color: #f0f6fc;
            letter-spacing: -0.02em;
        }
        .assistant-subtitle {
            margin: 0.35rem 0 0 0;
            font-size: 0.94rem;
            color: #9da7b3;
            line-height: 1.45;
        }
        .counter-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.45rem 0.85rem;
            border-radius: 999px;
            background-color: #161b22;
            border: 1px solid #30363d;
            color: #79c0ff;
            font-weight: 600;
            font-size: 0.9rem;
            width: 100%;
        }
        .counter-pill-limit {
            color: #ffa657;
            border-color: #6e4621;
            background-color: #231a10;
        }
        [data-testid="stChatMessage"] {
            background-color: #161b22;
            border: 1px solid #262c36;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.6rem;
        }
        [data-testid="stChatInput"] textarea {
            background-color: #161b22 !important;
            color: #f0f6fc !important;
            border-color: #30363d !important;
        }
        div.stButton > button {
            background-color: #21262d;
            color: #f0f6fc;
            border: 1px solid #30363d;
            border-radius: 10px;
            font-weight: 600;
        }
        div.stButton > button:hover {
            background-color: #30363d;
            border-color: #58a6ff;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def extract_error_detail(response: httpx.Response) -> str:
    """Safely extract a readable error message from a backend JSON response."""
    try:
        payload = response.json()
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
    except Exception:
        pass
    return f"Request failed with status code {response.status_code}."


def create_new_conversation(backend_url: str) -> tuple[bool, str | None]:
    """Request a fresh conversation ID from the FastAPI backend."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(f"{backend_url}/api/chat/new")
    except httpx.ConnectError:
        return (
            False,
            f"Cannot connect to the backend at {backend_url}. "
            "Start the FastAPI backend first, then click New Chat.",
        )
    except httpx.TimeoutException:
        return (
            False,
            "The backend timed out while starting a new conversation. Please try again.",
        )
    except httpx.HTTPError:
        return False, "An HTTP error occurred while connecting to the backend."

    if response.status_code != 200:
        return False, extract_error_detail(response)

    data = response.json()
    st.session_state.conversation_id = data["conversation_id"]
    st.session_state.question_limit = int(data.get("question_limit", DEFAULT_QUESTION_LIMIT))
    st.session_state.questions_used = int(data.get("questions_used", 0))
    st.session_state.messages = []
    st.session_state.error_message = None
    return True, None


def send_chat_message(
    backend_url: str, conversation_id: str, message: str
) -> tuple[dict | None, str | None, int | None]:
    """Send a user question to the backend and return (payload, error_message, status_code)."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(
                f"{backend_url}/api/chat",
                json={"conversation_id": conversation_id, "message": message},
            )
    except httpx.ConnectError:
        return (
            None,
            f"Cannot reach the backend at {backend_url}. "
            "Check that the FastAPI server is still running.",
            None,
        )
    except httpx.TimeoutException:
        return (
            None,
            "The request timed out waiting for a response. Please try again.",
            None,
        )
    except httpx.HTTPError:
        return None, "A network error occurred while communicating with the backend.", None

    if response.status_code != 200:
        return None, extract_error_detail(response), response.status_code

    return response.json(), None, 200


def init_session_state(backend_url: str) -> None:
    """Initialize Streamlit session state and create a backend conversation."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "questions_used" not in st.session_state:
        st.session_state.questions_used = 0
    if "question_limit" not in st.session_state:
        st.session_state.question_limit = DEFAULT_QUESTION_LIMIT
    if "error_message" not in st.session_state:
        st.session_state.error_message = None
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
        ok, err = create_new_conversation(backend_url)
        if not ok:
            st.session_state.error_message = err


def main() -> None:
    """Render the Streamlit Mini AI Assistant UI."""
    st.set_page_config(
        page_title="Mini AI Assistant",
        page_icon="🤖",
        layout="centered",
    )
    apply_dark_theme()

    backend_url = get_backend_url()
    init_session_state(backend_url)

    st.markdown(
        """
        <div class="assistant-header">
            <h1 class="assistant-title">Mini AI Assistant</h1>
            <p class="assistant-subtitle">
                Ask clear questions and explore technical concepts in beginner-friendly language.
                Conversations remember context in backend memory for up to 20 questions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    questions_used = st.session_state.questions_used
    question_limit = st.session_state.question_limit
    limit_reached = questions_used >= question_limit

    col_counter, col_action = st.columns([3, 1.3], vertical_alignment="center")
    with col_counter:
        pill_class = "counter-pill counter-pill-limit" if limit_reached else "counter-pill"
        st.markdown(
            f'<div class="{pill_class}">Questions used: {questions_used}/{question_limit}</div>',
            unsafe_allow_html=True,
        )
    with col_action:
        if st.button("New Chat", use_container_width=True):
            ok, err = create_new_conversation(backend_url)
            if not ok:
                st.session_state.error_message = err
            st.rerun()

    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    if not st.session_state.messages and not st.session_state.error_message:
        with st.chat_message("assistant"):
            st.markdown(
                "Hello! I am your Mini AI Assistant. Ask me a question below to get started."
            )

    for entry in st.session_state.messages:
        with st.chat_message(entry["role"]):
            if entry["role"] == "user":
                st.text(entry["content"])
            else:
                st.markdown(entry["content"])

    if limit_reached:
        st.warning(
            f"You have reached the {question_limit}-question limit for this conversation. "
            "Click **New Chat** above to start a fresh conversation."
        )

    prompt = st.chat_input(
        "Ask a question...",
        max_chars=MAX_MESSAGE_LENGTH,
        disabled=limit_reached or not st.session_state.conversation_id,
    )

    if prompt is not None:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            st.session_state.error_message = "Please type a question before sending."
            st.rerun()

        with st.chat_message("user"):
            st.text(cleaned_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                data, err, status_code = send_chat_message(
                    backend_url,
                    st.session_state.conversation_id,
                    cleaned_prompt,
                )

        if err is not None or data is None:
            if status_code == 404:
                st.session_state.conversation_id = None
                err = (
                    f"{err} (Backend memory resets whenever the backend server restarts.)"
                )
            elif status_code == 400 and "Question limit" in (err or ""):
                st.session_state.questions_used = st.session_state.question_limit
            st.session_state.error_message = err
            st.rerun()

        st.session_state.messages.append({"role": "user", "content": cleaned_prompt})
        st.session_state.messages.append(
            {"role": "assistant", "content": data["response"]}
        )
        st.session_state.questions_used = int(data["questions_used"])
        st.session_state.question_limit = int(
            data.get("question_limit", st.session_state.question_limit)
        )
        st.session_state.error_message = None
        st.rerun()


if __name__ == "__main__":
    main()
