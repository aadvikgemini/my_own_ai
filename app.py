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
        "Explain a concept simply",
        "Explain how an API works in simple, beginner-friendly terms.",
    ),
    (
        "Python list vs dictionary",
        "Help me understand the difference between a Python list and dictionary, and when to use each.",
    ),
    (
        "Step-by-step walkthrough",
        "Give me a clear step-by-step explanation of how Python virtual environments work.",
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
    """Apply Flowstate-inspired near-black glass UI and ambient background glow."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Onest:wght@400;500;600;700&display=swap');

        :root {
            --bg-app: #04050c;
            --bg-glass: rgba(14, 18, 32, 0.64);
            --bg-glass-strong: rgba(18, 23, 42, 0.78);
            --bg-glass-user: rgba(30, 41, 74, 0.56);
            --text-primary: #eef0f6;
            --text-secondary: #b9becf;
            --accent-cyan: #22d3ee;
            --accent-blue: #38bdf8;
            --accent-violet: #8b5cf6;
            --accent-magenta: #d946ef;
            --border-glass: rgba(255, 255, 255, 0.1);
            --border-glass-strong: rgba(255, 255, 255, 0.16);
        }

        html,
        body,
        .stApp {
            background-color: var(--bg-app) !important;
            color: var(--text-primary);
            font-family: 'Onest', Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            overflow-x: hidden;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        section.main {
            background: transparent !important;
        }

        /* Subtle slow-moving ambient cyan-to-violet fluid glow behind UI */
        .ambient-bg {
            position: fixed;
            inset: 0;
            z-index: -1;
            pointer-events: none;
            overflow: hidden;
            background: radial-gradient(circle at 50% -10%, rgba(56, 189, 248, 0.08), transparent 55%);
        }

        .ambient-orb {
            position: absolute;
            border-radius: 999px;
            filter: blur(90px);
            opacity: 0.42;
            will-change: transform;
        }

        .orb-cyan {
            width: 52vw;
            height: 52vw;
            max-width: 620px;
            max-height: 620px;
            top: -16%;
            left: -10%;
            background: radial-gradient(circle, rgba(34, 211, 238, 0.28) 0%, rgba(56, 189, 248, 0.12) 48%, transparent 72%);
            animation: driftCyan 24s ease-in-out infinite alternate;
        }

        .orb-violet {
            width: 54vw;
            height: 54vw;
            max-width: 660px;
            max-height: 660px;
            bottom: -18%;
            right: -12%;
            background: radial-gradient(circle, rgba(139, 92, 246, 0.28) 0%, rgba(217, 70, 239, 0.11) 46%, transparent 74%);
            animation: driftViolet 30s ease-in-out infinite alternate;
        }

        @keyframes driftCyan {
            0% { transform: translate3d(0, 0, 0) scale(1); }
            100% { transform: translate3d(9vw, 6vh, 0) scale(1.08); }
        }

        @keyframes driftViolet {
            0% { transform: translate3d(0, 0, 0) scale(1); }
            100% { transform: translate3d(-8vw, -7vh, 0) scale(1.1); }
        }

        @keyframes fadeUp {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (prefers-reduced-motion: reduce) {
            .ambient-orb,
            .welcome-panel,
            [data-testid="stChatMessage"] {
                animation: none !important;
            }
        }

        #MainMenu,
        footer {
            visibility: hidden;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }

        .block-container {
            position: relative;
            z-index: 1;
            max-width: 760px;
            padding-top: 1.15rem;
            padding-bottom: 1.75rem;
        }

        /* Quiet Glass Header */
        [data-testid="stHorizontalBlock"]:has(.brand-left) {
            padding: 0.6rem 0.9rem;
            border-radius: 16px;
            background: var(--bg-glass);
            border: 1px solid var(--border-glass);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.38);
            margin-bottom: 0.75rem;
        }

        .brand-left {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            min-width: 0;
        }

        .brand-mark {
            width: 32px;
            height: 32px;
            border-radius: 10px;
            background: linear-gradient(135deg, rgba(34, 211, 238, 0.2), rgba(139, 92, 246, 0.25));
            border: 1px solid rgba(255, 255, 255, 0.16);
            color: var(--accent-cyan);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            font-weight: 700;
            flex-shrink: 0;
        }

        .brand-title {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.015em;
            white-space: nowrap;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            padding: 0.26rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 500;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-glass);
            color: var(--text-secondary);
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            display: inline-block;
        }

        .status-ready {
            color: #d8f8ee;
            border-color: rgba(52, 211, 153, 0.28);
        }

        .status-ready .status-dot {
            background-color: #34d399;
            box-shadow: 0 0 8px rgba(52, 211, 153, 0.65);
        }

        .status-offline {
            color: #fde68a;
            border-color: rgba(245, 158, 11, 0.3);
        }

        .status-offline .status-dot {
            background-color: #f59e0b;
        }

        /* Welcome Screen */
        .welcome-panel {
            text-align: center;
            padding: 2.6rem 1.5rem 2rem;
            margin: 1.4rem auto 1rem;
            border-radius: 22px;
            background: var(--bg-glass);
            border: 1px solid var(--border-glass);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.07);
            animation: fadeUp 0.4s ease-out;
        }

        .welcome-heading {
            margin: 0 0 0.6rem;
            font-size: 1.7rem;
            font-weight: 600;
            letter-spacing: -0.025em;
            color: var(--text-primary);
        }

        .welcome-subtext {
            margin: 0 auto;
            max-width: 490px;
            font-size: 0.96rem;
            line-height: 1.6;
            color: var(--text-secondary);
        }

        .prompts-caption {
            text-align: center;
            font-size: 0.76rem;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 0.85rem 0 0.55rem;
            opacity: 0.85;
        }

        /* Chat Bubbles */
        [data-testid="stChatMessage"] {
            background: var(--bg-glass) !important;
            border: 1px solid var(--border-glass) !important;
            border-radius: 18px !important;
            padding: 1rem 1.2rem !important;
            margin-bottom: 0.8rem !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
            animation: fadeUp 0.25s ease-out;
        }

        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li {
            color: var(--text-primary);
            line-height: 1.65;
            font-size: 0.96rem;
        }

        [data-testid="stChatMessage"] pre {
            background: rgba(4, 5, 12, 0.85) !important;
            border: 1px solid var(--border-glass) !important;
            border-radius: 12px !important;
        }

        [data-testid="stChatMessage"] code {
            color: #67e8f9;
        }

        /* Buttons */
        div.stButton > button {
            background: var(--bg-glass-strong);
            color: var(--text-primary);
            border: 1px solid var(--border-glass);
            border-radius: 999px;
            font-family: 'Onest', Inter, sans-serif;
            font-size: 0.86rem;
            font-weight: 500;
            min-height: 2.5rem;
            padding: 0.45rem 1rem;
            backdrop-filter: blur(10px);
            transition: border-color 0.16s ease, background-color 0.16s ease, transform 0.12s ease;
        }

        div.stButton > button:hover:not(:disabled) {
            background: rgba(34, 211, 238, 0.12);
            border-color: rgba(56, 189, 248, 0.55);
            color: #ffffff;
        }

        div.stButton > button:focus-visible {
            outline: 2px solid var(--accent-cyan) !important;
            outline-offset: 2px !important;
        }

        div.stButton > button:disabled {
            opacity: 0.45;
            cursor: not-allowed;
        }

        /* Flowstate-inspired Glass Composer */
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div {
            background: linear-gradient(to top, rgba(4, 5, 12, 0.96) 65%, rgba(4, 5, 12, 0)) !important;
            position: relative;
            z-index: 2;
        }

        [data-testid="stBottom"] > div {
            padding-top: 0.75rem;
            padding-bottom: 0.65rem;
        }

        [data-testid="stChatInput"] {
            background: rgba(16, 20, 36, 0.76) !important;
            border: 1px solid rgba(255, 255, 255, 0.14) !important;
            border-radius: 22px !important;
            backdrop-filter: blur(18px) !important;
            -webkit-backdrop-filter: blur(18px) !important;
            box-shadow: 0 14px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
            padding: 0.2rem 0.3rem !important;
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: rgba(34, 211, 238, 0.6) !important;
            box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.14), 0 14px 40px rgba(0, 0, 0, 0.55) !important;
        }

        [data-testid="stChatInput"] textarea {
            background: transparent !important;
            color: var(--text-primary) !important;
            font-family: 'Onest', Inter, sans-serif !important;
            font-size: 0.95rem !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--text-secondary) !important;
            opacity: 0.8;
        }

        [data-testid="stChatInputSubmitButton"] {
            background: linear-gradient(135deg, #22d3ee 0%, #3b82f6 55%, #8b5cf6 100%) !important;
            color: #04050c !important;
            border-radius: 999px !important;
            border: none !important;
            box-shadow: 0 4px 16px rgba(56, 189, 248, 0.35) !important;
        }

        [data-testid="stChatInputSubmitButton"]:hover:not(:disabled) {
            filter: brightness(1.08);
        }

        [data-testid="stChatInputSubmitButton"]:disabled {
            background: rgba(255, 255, 255, 0.08) !important;
            color: rgba(185, 190, 207, 0.45) !important;
            box-shadow: none !important;
        }

        .input-meta-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.35rem 0.5rem 0;
            font-size: 0.78rem;
            color: var(--text-secondary);
            flex-wrap: wrap;
        }

        .question-counter {
            font-weight: 500;
            color: var(--text-secondary);
        }

        .question-counter-limit {
            color: #fde68a;
        }

        @media (max-width: 640px) {
            .block-container {
                padding-top: 0.75rem;
                padding-left: 0.85rem;
                padding-right: 0.85rem;
            }

            [data-testid="stHorizontalBlock"]:has(.brand-left) {
                padding: 0.55rem 0.75rem;
            }

            .welcome-panel {
                padding: 1.9rem 1.1rem 1.5rem;
                margin-top: 0.75rem;
            }

            .welcome-heading {
                font-size: 1.38rem;
            }
        }
        </style>
        <div class="ambient-bg" aria-hidden="true">
            <div class="ambient-orb orb-cyan"></div>
            <div class="ambient-orb orb-violet"></div>
        </div>
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
    """Render the quiet glass header with brand mark, Ready status, and New Chat."""
    with st.container():
        col_brand, col_status, col_action = st.columns(
            [2.6, 1.0, 1.1], vertical_alignment="center"
        )
        with col_brand:
            st.markdown(
                """
                <div class="brand-left">
                    <div class="brand-mark" aria-hidden="true">&#10022;</div>
                    <h1 class="brand-title">Mini AI Assistant</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_status:
            status_cls = "status-ready" if backend_connected else "status-offline"
            status_label = "Ready" if backend_connected else "Offline"
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-end;">
                    <span class="status-pill {status_cls}">
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


def render_welcome_screen(*, can_submit: bool) -> None:
    """Render the centered empty-state welcome heading and starter prompts."""
    st.markdown(
        """
        <div class="welcome-panel">
            <h2 class="welcome-heading">Your personal AI assistant</h2>
            <p class="welcome-subtext">
                Ask clear questions, explore ideas, or get beginner-friendly explanations in seconds.
            </p>
        </div>
        <div class="prompts-caption">Starter prompts</div>
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
    """Render the glass-style composer bar, shortcut hint, and remaining count."""
    questions_used = st.session_state.questions_used
    question_limit = st.session_state.question_limit
    questions_remaining = max(0, question_limit - questions_used)
    input_disabled = (
        not backend_connected
        or st.session_state.conversation_id is None
        or limit_reached
        or st.session_state.is_generating
    )

    placeholder = (
        "Question limit reached — click New Chat to continue"
        if limit_reached
        else "Ask anything..."
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
                <span class="{counter_class}">{questions_used} / {question_limit} questions &middot; {questions_remaining} left</span>
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
