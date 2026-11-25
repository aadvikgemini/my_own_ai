# Mini AI Assistant

A beginner-friendly, security-focused AI chatbot built with **Streamlit** (frontend), **FastAPI** (backend), and the official **OpenAI Python SDK** using the **Responses API**.

## Architecture

```text
Streamlit UI (app.py) ──HTTP──▶ FastAPI Backend (backend/main.py) ──HTTPS──▶ OpenAI / Groq Responses API
```

- **Frontend (`app.py`)**: Provides a dark, minimal product UI featuring a compact status header (`Connected` / `Backend offline`, `New Chat`), a centered welcome screen with three clickable starter prompts, native Markdown/code-block chat bubbles, a pinned multiline chat input (`Enter` to send, `Shift+Enter` for newline), and a live `X / 20 questions` counter synced from the backend. It never accesses `OPENAI_API_KEY` or imports `openai`.
- **Backend (`backend/`)**: Validates all requests with Pydantic, enforces the **20-question limit per conversation**, manages isolated conversations in backend memory, applies in-memory rate limiting and CORS restrictions, and calls the Responses API using a backend-only API key.

> **Important:** Conversations are stored **only in backend memory** for local execution. **Restarting the FastAPI backend clears all stored conversations.**

---

## Project Structure

```text
mini-ai-assistant/
├── app.py                      # Redesigned Streamlit chat UI
├── frontend/
│   └── index.html              # Lightweight web UI served at GET / (for Vercel)
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI endpoints, middleware, rate limiter
│   ├── config.py               # Environment settings & limits
│   ├── models.py               # Pydantic request/response schemas
│   ├── chat_service.py         # OpenAI Responses API client & error handling
│   ├── conversation_store.py   # Thread-safe in-memory conversation store
│   ├── session_state.py        # HMAC-signed continuity helper
│   └── .env.example            # Template for backend environment variables
├── tests/
│   ├── __init__.py
│   └── test_api.py             # Automated pytest suite (backend + Streamlit UI tests)
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .vercelignore
└── README.md
```

---

## macOS Setup Guide (Step-by-Step)

Open the **Terminal** app on your Mac (`Cmd + Space`, type `Terminal`, press Enter), then navigate into the `mini-ai-assistant` folder.

### 1. Check your Python version

Check which Python 3 version is installed (Python 3.11 or 3.12+ is recommended):

```bash
python3 --version
```

If macOS shows an older version such as `Python 3.9.x`, check if Homebrew Python is installed:

```bash
/opt/homebrew/bin/python3.11 --version
```

*(If needed, you can install a modern Python version with Homebrew: `brew install python@3.12`.)*

### 2. Create a virtual environment

```bash
# If python3 is 3.11 or 3.12+:
python3 -m venv .venv

# Or if using Homebrew Python 3.11 specifically:
/opt/homebrew/bin/python3.11 -m venv .venv
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

Your terminal prompt will show `(.venv)` at the beginning. Activate `.venv` in each new terminal window you open for this project.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Create your `.env` file

Copy the example template to create your backend `.env` file:

```bash
cp backend/.env.example backend/.env
```

### 6. Add your API key

Open `backend/.env` in any text editor (for example, `nano backend/.env`) and configure your credentials:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5-mini
OPENAI_BASE_URL=
ALLOWED_ORIGIN=http://localhost:8501
```

- `OPENAI_API_KEY` *(required)*: Your personal OpenAI API key (or a Groq API key if using Groq's OpenAI-compatible endpoint).
- `OPENAI_MODEL` *(optional)*: Defaults to `gpt-5-mini` for OpenAI (or `openai/gpt-oss-20b` when using Groq).
- `OPENAI_BASE_URL` *(optional)*: Leave blank for OpenAI, or set `https://api.groq.com/openai/v1` for Groq.
- `ALLOWED_ORIGIN` *(optional)*: Defaults to `http://localhost:8501,http://127.0.0.1:8501`.

Save and close the file. Never commit `backend/.env` to Git.

### 7. Start the FastAPI backend (Terminal Window 1)

In your first Terminal window (inside `mini-ai-assistant` with `.venv` activated):

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Leave this terminal running. You can verify the backend at `http://127.0.0.1:8000/health`, which returns `{"status":"ok"}`.

### 8. Start the Streamlit frontend (Terminal Window 2)

Open a **second** Terminal window inside `mini-ai-assistant`, activate `.venv`, and start Streamlit:

```bash
source .venv/bin/activate
streamlit run app.py
```

Streamlit will open in your browser (usually `http://localhost:8501`).

### 9. Use the chatbot

1. Check the header status badge (`Connected`) to confirm Streamlit can reach FastAPI.
2. Click one of the three starter prompts on the welcome screen, or type your own question at the bottom and press **Enter** (`Shift+Enter` for a new line).
3. Track your remaining allowance in the bottom-right counter (`X / 20 questions`).
4. When you reach `20 / 20 questions`—or want a clean topic—click **New Chat** to start a new backend conversation.

---

## Running Automated Tests

Automated tests use mocked upstream API responses so they **do not** spend API credits:

```bash
source .venv/bin/activate
pytest -v
```

---

## Troubleshooting Common Errors

- **Header shows `Backend offline` / `Backend server is currently unavailable...`**
  - Ensure Terminal Window 1 is running `uvicorn backend.main:app --host 127.0.0.1 --port 8000`, then click **New Chat**.
- **Error: `OpenAI API key is missing...`**
  - Verify `backend/.env` exists inside `backend/` with `OPENAI_API_KEY=...`, then restart `uvicorn`.
- **Error: `OpenAI authentication failed...`**
  - Confirm your key in `backend/.env` is valid and matches your configured `OPENAI_BASE_URL` / `OPENAI_MODEL`.
- **Error: `Conversation not found. Please start a new chat.`**
  - Conversations live in backend memory only; click **New Chat** after restarting FastAPI.
- **Error: `Question limit of 20 reached...`**
  - Click **New Chat** or **Start New Chat** to begin a fresh conversation.
- **Error: `Too many requests...` (HTTP 429)**
  - The local sliding-window rate limiter (30 requests per 60 seconds) was triggered. Wait a few seconds and click **Retry last question**.
- **Error: `Address already in use` on port 8000 or 8501**
  - Stop any previously running process with `Ctrl + C`, or specify another port.

---

## Security Notes & Limitations

- **Local single-user design:** Intended for local personal use on `127.0.0.1` / `localhost`.
- **Rate limiting:** The backend uses an in-memory sliding-window rate limiter suitable for local development; **this is not production-grade protection against distributed attacks**.
- **Public deployment warning:** Do **not** expose the backend publicly without HTTPS/TLS, authentication, persistent/distributed rate limiting, and reverse-proxy hardening.
