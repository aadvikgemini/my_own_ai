# Mini AI Assistant

A beginner-friendly, security-focused AI chatbot built with **Streamlit** (frontend), **FastAPI** (backend), and the official **OpenAI Python SDK** using the **Responses API**.

## Architecture

```text
Streamlit UI (app.py) ──HTTP──▶ FastAPI Backend (backend/main.py) ──HTTPS──▶ OpenAI Responses API
```

- **Frontend (`app.py`)**: Renders a dark-themed chat UI, tracks display history in `st.session_state`, and sends HTTP requests to the FastAPI backend. It never touches `OPENAI_API_KEY` or imports `openai`.
- **Backend (`backend/`)**: Validates all requests with Pydantic, enforces the **20-question limit per conversation**, manages isolated conversations in backend memory, applies in-memory rate limiting and CORS restrictions, and calls OpenAI using a backend-only API key.

> **Important:** Conversations are stored **only in backend memory** for this first local version. **Restarting the FastAPI backend clears all stored conversations.**

---

## Project Structure

```text
mini-ai-assistant/
├── app.py                      # Streamlit user interface
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI endpoints, middleware, rate limiter
│   ├── config.py               # Environment settings & limits
│   ├── models.py               # Pydantic request/response schemas
│   ├── chat_service.py         # OpenAI Responses API client & error handling
│   ├── conversation_store.py   # Thread-safe in-memory conversation store
│   └── .env.example            # Template for backend environment variables
├── tests/
│   ├── __init__.py
│   └── test_api.py             # Automated pytest suite (mocked OpenAI calls)
├── requirements.txt
├── .gitignore
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

A **virtual environment** is an isolated folder that keeps this project's Python packages separate from the rest of your Mac.

Run one of the following commands inside `mini-ai-assistant`:

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

Your terminal prompt should now show `(.venv)` at the beginning. *(You will need to activate `.venv` in each new terminal window you open for this project.)*

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Create your `.env` file

Copy the example template file to create your real backend environment file:

```bash
cp backend/.env.example backend/.env
```

### 6. Add your OpenAI API key

Open `backend/.env` in any text editor (for example, `nano backend/.env`) and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5-mini
ALLOWED_ORIGIN=http://localhost:8501
```

- `OPENAI_API_KEY` *(required)*: Your personal OpenAI API key.
- `OPENAI_MODEL` *(optional)*: Defaults to `gpt-5-mini` if left blank.
- `ALLOWED_ORIGIN` *(optional)*: Defaults to `http://localhost:8501,http://127.0.0.1:8501`.

Save and close the file. Never share or commit `backend/.env` to Git.

### 7. Start the FastAPI backend (Terminal Window 1)

In your first Terminal window (inside `mini-ai-assistant` with `.venv` activated):

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Leave this window running. You can verify it is healthy by opening `http://127.0.0.1:8000/health` in your browser, which should show `{"status":"ok"}`.

### 8. Start the Streamlit frontend (Terminal Window 2)

Open a **second** Terminal window, `cd` into `mini-ai-assistant`, activate `.venv`, and run Streamlit:

```bash
source .venv/bin/activate
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) and open it in your web browser.

### 9. Use the chatbot

1. Type a question in the input box at the bottom and press **Enter** (or click the send button).
2. Watch the **Questions used: X/20** counter at the top increment with each completed question.
3. Ask follow-up questions—the assistant remembers earlier messages in the current conversation.
4. Once you reach **20/20** questions (or whenever you want a fresh topic), click **New Chat** to start a new conversation with a new conversation ID.

---

## Running Automated Tests

Automated tests use mocked OpenAI responses so they **do not** require a real API key or spend API credits:

```bash
source .venv/bin/activate
pytest -v
```

---

## 10. Troubleshooting Common Errors

- **Error: `OpenAI API key is missing...`**
  - Make sure you created `backend/.env` (inside the `backend/` folder, not the root folder) and added `OPENAI_API_KEY=...`.
  - Restart the `uvicorn` backend after saving `backend/.env`.
- **Error: `OpenAI authentication failed...`**
  - Double-check that your key in `backend/.env` has no extra quotes or spaces and has not been revoked in the OpenAI dashboard.
- **Error: `Cannot connect to the backend at http://127.0.0.1:8000`**
  - Confirm Terminal Window 1 is still running `uvicorn backend.main:app --host 127.0.0.1 --port 8000`.
- **Error: `Conversation not found. Please start a new chat.`**
  - Conversations live in backend memory only. If you restarted the FastAPI backend while the browser tab was open, click **New Chat** in Streamlit.
- **Error: `Question limit of 20 reached...`**
  - Each conversation allows up to 20 questions. Click **New Chat** to start a new conversation.
- **Error: `Too many requests...` (HTTP 429)**
  - You triggered the local rate limiter (30 requests per 60 seconds). Wait a few seconds and try again.
- **Error: `Address already in use` on port 8000 or 8501**
  - Another process is already using that port. Stop the previous server with `Ctrl + C`, or run on another port (updating `BACKEND_URL` / `ALLOWED_ORIGIN` if needed).

---

## Security Notes & Limitations

- **Local single-user design:** This version is intended strictly for local personal use on `127.0.0.1` / `localhost`.
- **Rate limiting:** The backend uses an in-memory sliding-window rate limiter to reduce accidental abuse during local development. **This is not production-grade protection against distributed attacks** (it resets on restart and applies only to a single local process).
- **Public deployment warning:** Do **not** expose this backend publicly without adding HTTPS/TLS, authentication, persistent/distributed rate limiting, and reverse-proxy hardening.
