"""Streamlit frontend for Mini AI Assistant.

Communicates exclusively with the FastAPI backend over HTTP and never loads,
stores, or exposes OpenAI API keys.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import httpx
import streamlit as st

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_QUESTION_LIMIT = 20
MAX_MESSAGE_LENGTH = 2000
HEALTH_TIMEOUT_SECONDS = 2.0
NEW_CHAT_TIMEOUT_SECONDS = 10.0
CHAT_TIMEOUT_SECONDS = 35.0

EXAMPLE_PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "Explain a concept in simple terms.",
        "Explain how an API works in simple, beginner-friendly terms.",
    ),
    (
        "Help me understand a Python problem.",
        "Help me understand a Python problem: what is the difference between a list and a dictionary, and when should I use each?",
    ),
    (
        "Give me a step-by-step explanation of a topic.",
        "Give me a step-by-step explanation of how Python virtual environments work.",
    ),
)


def get_backend_url() -> str:
    """Resolve and validate the backend base URL from environment configuration."""
    raw = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL).strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return DEFAULT_BACKEND_URL


def apply_custom_css() -> None:
    """Apply clean, minimal dark styling using organized static CSS."""
    st.markdown(
        """
        <style>
        :root {
            --bg-app: #0b0e14;
            --bg-surface: #121721;
            --bg-elevated: #171d29;
            --accent: #38bdf8;
            --accent-hover: #0ea5e9;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border-subtle: rgba(148, 163, 184, 0.16);
        }

        .stApp {
            background-color: var(--bg-app);
            color: var(--text-primary);
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            overflow-x: hidden;
        }

        #MainMenu,
        footer {
            visibility: hidden;
        }

        [data-testid="stHeader"] {
            background-color: rgba(11, 14, 20, 0.9);
            backdrop-filter: blur(8px);
        }

        .block-container {
            max-width: 780px;
            padding-top: 1.25rem;
            padding-bottom: 1.5rem;
        }

        /* Compact top header */
        .top-header-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            flex-wrap: wrap;
            padding: 0.15rem 0;
        }

        .brand-mark {
            width: 34px;
            height: 34px;
            border-radius: 10px;
            background-color: var(--bg-elevated);
            border: 1px solid rgba(56, 189, 248, 0.35);
            color: var(--accent);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            font-weight: 700;
            flex-shrink: 0;
        }

        .brand-title {
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.01em;
            margin: 0;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.22rem 0.6rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 500;
            border: 1px solid var(--border-subtle);
            background-color: var(--bg-surface);
            color: var(--text-secondary);
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            display: inline-block;
        }

        .status-online .status-dot {
            background-color: #34d399;
        }

        .status-online {
            color: #a7f3d0;
            border-color: rgba(52, 211, 153, 0.25);
        }

        .status-offline .status-dot {
            background-color: #f59e0b;
        }

        .status-offline {
            color: #fcd34d;
            border-color: rgba(245, 158, 11, 0.28);
        }

        .header-divider {
            border: 0;
            border-top: 1px solid var(--border-subtle);
            margin: 0.65rem 0 1.25rem 0;
        }

        /* Centered welcome screen */
        .welcome-panel {
            text-align: center;
            padding: 2.5rem 1.25rem 1.5rem 1.25rem;
            margin: 0.5rem auto 1rem auto;
            background-color: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
        }

        .welcome-heading {
            font-size: 1.55rem;
            font-weight: 650;
            color: var(--text-primary);
            letter-spacing: -0.02em;
            margin: 0 0 0.55rem 0;
        }

        .welcome-subtext {
            font-size: 0.96rem;
            color: var(--text-secondary);
            line-height: 1.55;
            max-width: 520px;
            margin: 0 auto;
        }

        .prompts-caption {
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 1rem 0 0.5rem 0;
            text-align: center;
        }

        /* Chat messages */
        [data-testid="stChatMessage"] {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 0.95rem 1.15rem;
            margin-bottom: 0.75rem;
        }

        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li {
            color: var(--text-primary);
            line-height: 1.6;
            font-size: 0.96rem;
        }

        [data-testid="stChatMessage"] pre {
            background-color: var(--bg-app) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 10px !important;
        }

        [data-testid="stChatMessage"] code {
            color: #7dd3fc;
        }

        /* Buttons */
        div.stButton > button {
            background-color: var(--bg-elevated);
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            font-size: 0.88rem;
            font-weight: 500;
            min-height: 2.55rem;
            padding: 0.45rem 0.9rem;
            transition: border-color 0.15s ease, background-color 0.15s ease;
        }

        div.stButton > button:hover:not(:disabled) {
            background-color: #1e2636;
            border-color: var(--accent);
            color: #ffffff;
        }

        div.stButton > button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* Bottom pinned input area */
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div {
            background-color: var(--bg-app) !important;
        }

        [data-testid="stBottom"] > div {
            border-top: 1px solid rgba(148, 163, 184, 0.12);
            padding-top: 0.65rem;
        }

        [data-testid="stChatInput"] {
            background-color: var(--bg-surface) !important;
            border: 1px solid rgba(148, 163, 184, 0.24) !important;
            border-radius: 14px !important;
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: var(--accent) !important;
        }

        [data-testid="stChatInput"] textarea {
            background-color: transparent !important;
            color: var(--text-primary) !important;
            font-size: 0.95rem !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--text-secondary) !important;
        }

        [data-testid="stChatInputSubmitButton"] {
            background-color: var(--accent) !important;
            color: #0b0e14 !important;
            border-radius: 10px !important;
        }

        [data-testid="stChatInputSubmitButton"]:hover:not(:disabled) {
            background-color: var(--accent-hover) !important;
        }

        [data-testid="stChatInputSubmitButton"]:disabled {
            background-color: #1e293b !important;
            color: #64748b !important;
        }

        .input-meta-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.25rem 0.25rem 0 0.25rem;
            font-size: 0.78rem;
            color: var(--text-secondary);
            flex-wrap: wrap;
        }

        .question-counter {
            font-weight: 500;
            color: var(--text-secondary);
        }

        .question-counter-limit {
            color: #fbbf24;
        }

        @media (max-width: 640px) {
            .block-container {
                padding-top: 0.85rem;
                padding-left: 0.85rem;
                padding-right: 0.85rem;
            }

            .welcome-panel {
                padding: 1.75rem 1rem 1.25rem 1rem;
            }

            .welcome-heading {
                font-size: 1.32rem;
            }
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


def check_backend_health(backend_url: str) -> bool:
    """Check whether the FastAPI backend is reachable and healthy."""
    try:
        with httpx.Client(timeout=HEALTH_TIMEOUT_SECONDS) as client:
            response = client.get(f"{backend_url}/health")
        return response.status_code == 200 and response.json().get("status") == "ok"
    except Exception:
        return False


def create_new_conversation(backend_url: str) -> tuple[bool, str | None]:
    """Create a fresh backend conversation and reset local chat state only on success."""
    try:
        with httpx.Client(timeout=NEW_CHAT_TIMEOUT_SECONDS) as client:
            response = client.post(f"{backend_url}/api/chat/new")
    except httpx.ConnectError:
        return (
            False,
            f"Backend unavailable at {backend_url}. Start the FastAPI server and try again.",
        )
    except httpx.TimeoutException:
        return (
            False,
            "The backend timed out while starting a new conversation. Please try again.",
        )
    except httpx.HTTPError:
        return False, "A network error occurred while starting a new conversation."

    if response.status_code != 200:
        return False, extract_error_detail(response)

    data = response.json()
    st.session_state.conversation_id = data["conversation_id"]
    st.session_state.session_state_token = response.headers.get("X-Session-State")
    st.session_state.question_limit = int(
        data.get("question_limit", DEFAULT_QUESTION_LIMIT)
    )
    st.session_state.questions_used = int(data.get("questions_used", 0))
    st.session_state.messages = []
    st.session_state.error_message = None
    st.session_state.pending_prompt = None
    st.session_state.failed_prompt = None
    st.session_state.is_generating = False
    return True, None


def send_chat_message(
    backend_url: str,
    conversation_id: str,
    message: str,
    session_state_token: str | None = None,
) -> tuple[dict | None, str | None, int | None, str | None]:
    """Send a user question to FastAPI and return (data, error, status_code, next_token)."""
    headers: dict[str, str] = {}
    if session_state_token:
        headers["X-Session-State"] = session_state_token

    try:
        with httpx.Client(timeout=CHAT_TIMEOUT_SECONDS) as client:
            response = client.post(
                f"{backend_url}/api/chat",
                json={"conversation_id": conversation_id, "message": message},
                headers=headers,
            )
    except httpx.ConnectError:
        return (
            None,
            f"Cannot reach the backend at {backend_url}. "
            "Check that the FastAPI server is running, then retry.",
            None,
            None,
        )
    except httpx.TimeoutException:
        return (
            None,
            "The assistant timed out while generating a response. Please try again.",
            None,
            None,
        )
    except httpx.HTTPError:
        return (
            None,
            "A network error occurred while communicating with the backend.",
            None,
            None,
        )

    if response.status_code != 200:
        return None, extract_error_detail(response), response.status_code, None

    next_token = response.headers.get("X-Session-State") or session_state_token
    return response.json(), None, 200, next_token


def init_session_state(backend_url: str) -> bool:
    """Initialize session state keys and return whether the backend is connected."""
    defaults: dict[str, object] = {
        "messages": [],
        "questions_used": 0,
        "question_limit": DEFAULT_QUESTION_LIMIT,
        "error_message": None,
        "pending_prompt": None,
        "failed_prompt": None,
        "is_generating": False,
        "conversation_id": None,
        "session_state_token": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    backend_connected = check_backend_health(backend_url)
    if backend_connected and st.session_state.conversation_id is None:
        with st.spinner("Connecting to Mini AI Assistant..."):
            ok, err = create_new_conversation(backend_url)
        if not ok:
            st.session_state.error_message = err
    elif not backend_connected and st.session_state.conversation_id is None:
        st.session_state.error_message = (
            "Backend server is currently unavailable. Start FastAPI with "
            "`uvicorn backend.main:app --host 127.0.0.1 --port 8000` and click New Chat."
        )

    return backend_connected


def queue_prompt(prompt_text: str) -> None:
    """Queue a validated non-empty prompt for generation and trigger rerun."""
    cleaned = prompt_text.strip()
    if not cleaned:
        st.session_state.error_message = "Please enter a question before sending."
        st.rerun()

    st.session_state.pending_prompt = cleaned
    st.session_state.is_generating = True
    st.session_state.error_message = None
    st.session_state.failed_prompt = None
    st.rerun()


def render_header(backend_url: str, backend_connected: bool) -> None:
    """Render the compact top header with brand icon, status badge, and New Chat."""
    with st.container():
        col_brand, col_action = st.columns([3.3, 1.1], vertical_alignment="center")
        with col_brand:
            status_cls = "status-online" if backend_connected else "status-offline"
            status_label = "Connected" if backend_connected else "Backend offline"
            st.markdown(
                f"""
                <div class="top-header-brand">
                    <div class="brand-mark" aria-hidden="true">&#10022;</div>
                    <h1 class="brand-title">Mini AI Assistant</h1>
                    <span class="status-badge {status_cls}">
                        <span class="status-dot"></span>
                        {status_label}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_action:
            if st.button(
                "New Chat",
                key="header_new_chat",
                use_container_width=True,
                disabled=st.session_state.is_generating,
            ):
                ok, err = create_new_conversation(backend_url)
                if not ok:
                    st.session_state.error_message = err
                st.rerun()

        st.markdown('<hr class="header-divider" />', unsafe_allow_html=True)


def render_welcome_screen(*, can_submit: bool) -> None:
    """Render the centered empty-state welcome message and 3 clickable prompts."""
    st.markdown(
        """
        <div class="welcome-panel">
            <h2 class="welcome-heading">Your personal AI assistant</h2>
            <p class="welcome-subtext">
                Ask a question, explore an idea, or get help with something you're learning.
            </p>
        </div>
        <div class="prompts-caption">Try an example</div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(len(EXAMPLE_PROMPTS))
    for idx, (label, full_prompt) in enumerate(EXAMPLE_PROMPTS):
        with cols[idx]:
            if st.button(
                label,
                key=f"example_prompt_{idx}",
                use_container_width=True,
                disabled=not can_submit,
            ):
                queue_prompt(full_prompt)


def render_chat_history() -> None:
    """Render all completed messages using Streamlit native chat containers."""
    for entry in st.session_state.messages:
        avatar = (
            ":material/person:"
            if entry["role"] == "user"
            else ":material/auto_awesome:"
        )
        with st.chat_message(entry["role"], avatar=avatar):
            st.markdown(entry["content"])


def process_pending_prompt(backend_url: str) -> None:
    """Execute the pending backend request while displaying a loading indicator."""
    pending = st.session_state.pending_prompt
    if not pending:
        st.session_state.is_generating = False
        return

    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(pending)

    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        with st.spinner("Thinking..."):
            data, err, status_code, next_token = send_chat_message(
                backend_url,
                st.session_state.conversation_id,
                pending,
                st.session_state.session_state_token,
            )

    st.session_state.is_generating = False
    st.session_state.pending_prompt = None

    if err is not None or data is None:
        if status_code == 404:
            st.session_state.conversation_id = None
            st.session_state.session_state_token = None
            st.session_state.failed_prompt = None
            st.session_state.error_message = (
                f"{err} (In-memory conversations clear when the backend restarts. "
                "Click New Chat to start a fresh conversation.)"
            )
        elif status_code == 400 and "Question limit" in err:
            st.session_state.questions_used = st.session_state.question_limit
            st.session_state.failed_prompt = None
            st.session_state.error_message = err
        else:
            st.session_state.failed_prompt = pending
            st.session_state.error_message = err
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": pending})
    st.session_state.messages.append(
        {"role": "assistant", "content": data["response"]}
    )
    st.session_state.questions_used = int(data["questions_used"])
    st.session_state.question_limit = int(
        data.get("question_limit", st.session_state.question_limit)
    )
    st.session_state.session_state_token = next_token
    st.session_state.failed_prompt = None
    st.session_state.error_message = None
    st.rerun()


def render_feedback_states(
    backend_url: str, *, backend_connected: bool, limit_reached: bool
) -> None:
    """Render error notices, retry controls, and 20-question limit guidance."""
    if not backend_connected and st.session_state.conversation_id is not None:
        st.warning(
            "The backend server appears offline. Restart FastAPI (`uvicorn backend.main:app --host 127.0.0.1 --port 8000`) to continue."
        )

    if st.session_state.error_message:
        st.error(st.session_state.error_message)

        if (
            st.session_state.failed_prompt
            and st.session_state.conversation_id is not None
            and not limit_reached
            and not st.session_state.is_generating
        ):
            if st.button("Retry last question", key="retry_last_btn"):
                queue_prompt(st.session_state.failed_prompt)

    if limit_reached:
        col_msg, col_btn = st.columns([3.2, 1.2], vertical_alignment="center")
        with col_msg:
            st.info(
                f"You have reached the {st.session_state.question_limit}-question limit "
                "for this conversation. Start a new chat to ask more questions."
            )
        with col_btn:
            if st.button(
                "Start New Chat",
                key="limit_new_chat",
                use_container_width=True,
                disabled=st.session_state.is_generating,
            ):
                ok, err = create_new_conversation(backend_url)
                if not ok:
                    st.session_state.error_message = err
                st.rerun()


def render_bottom_input(
    *, backend_connected: bool, limit_reached: bool
) -> str | None:
    """Render the pinned multiline chat input, shortcut hint, and question counter."""
    questions_used = st.session_state.questions_used
    question_limit = st.session_state.question_limit
    input_disabled = (
        not backend_connected
        or st.session_state.conversation_id is None
        or limit_reached
        or st.session_state.is_generating
    )

    placeholder = (
        "Question limit reached — click New Chat to continue"
        if limit_reached
        else "Ask a question..."
    )
    counter_class = (
        "question-counter question-counter-limit"
        if limit_reached
        else "question-counter"
    )
    hint_text = (
        "Generating response..."
        if st.session_state.is_generating
        else "Enter to send · Shift+Enter for new line"
    )

    bottom_container = getattr(st, "bottom", None) or st.container()
    with bottom_container:
        prompt = st.chat_input(
            placeholder,
            max_chars=MAX_MESSAGE_LENGTH,
            disabled=input_disabled,
        )
        st.markdown(
            f"""
            <div class="input-meta-row">
                <span>{hint_text}</span>
                <span class="{counter_class}">{questions_used} / {question_limit} questions</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return prompt


def main() -> None:
    """Render the complete Mini AI Assistant Streamlit interface."""
    st.set_page_config(
        page_title="Mini AI Assistant",
        page_icon="✦",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    apply_custom_css()

    backend_url = get_backend_url()
    backend_connected = init_session_state(backend_url)

    questions_used = st.session_state.questions_used
    question_limit = st.session_state.question_limit
    limit_reached = questions_used >= question_limit
    can_submit = (
        backend_connected
        and st.session_state.conversation_id is not None
        and not limit_reached
        and not st.session_state.is_generating
    )

    render_header(backend_url, backend_connected)

    submitted_prompt = render_bottom_input(
        backend_connected=backend_connected,
        limit_reached=limit_reached,
    )

    with st.container():
        if not st.session_state.messages and not st.session_state.pending_prompt:
            render_welcome_screen(can_submit=can_submit)
        else:
            render_chat_history()

        if st.session_state.is_generating and st.session_state.pending_prompt:
            process_pending_prompt(backend_url)

        render_feedback_states(
            backend_url,
            backend_connected=backend_connected,
            limit_reached=limit_reached,
        )

    if submitted_prompt is not None and not st.session_state.is_generating:
        queue_prompt(submitted_prompt)


if __name__ == "__main__":
    main()
