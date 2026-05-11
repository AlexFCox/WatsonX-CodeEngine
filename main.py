"""
SalesBud – watsonx Orchestrate Chat Proxy
Deployed to IBM Code Engine (group13, eu-de)

Flow:
  POST /chat  →  get IAM token  →  exchange for WXO token  →  call Orchestrate chat API  →  return response
  POST /chat/new  →  create a new thread, return thread_id
"""

import os
import time
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="SalesBud Proxy", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow your Vercel frontend (and localhost for dev) to call this API
ALLOWED_ORIGINS = [
    "https://your-salesbud-app.vercel.app",   # ← replace with your actual Vercel URL
    "http://localhost:5173",                   # Vite dev server
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config from environment variables (set in Code Engine) ────────────────────
IBM_API_KEY          = os.environ["IBM_API_KEY"]           # Your IBM Cloud IAM API key
WXO_INSTANCE_GUID    = os.environ["WXO_INSTANCE_GUID"]     # 64926151-85dd-4264-90c8-53185f9e93ff
WXO_AGENT_ID         = os.environ["WXO_AGENT_ID"]          # a4f6e8b7-be3c-4c81-97f6-91a131a03b74
WXO_HOST             = os.environ.get(
    "WXO_HOST", "https://api.eu-de.watson-orchestrate.cloud.ibm.com"
)

IAM_TOKEN_URL        = "https://iam.cloud.ibm.com/identity/token"
WXO_INSTANCE_URL     = f"{WXO_HOST}/instances/{WXO_INSTANCE_GUID}"

# ── Token cache (in-memory, reused across requests in the same container) ─────
_token_cache: dict = {"token": None, "expires_at": 0}


def get_iam_token() -> str:
    """Exchange IBM API key for an IAM access token. Cached for 55 minutes."""
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"]:
        return _token_cache["token"]

    resp = httpx.post(
        IAM_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "apikey": IBM_API_KEY,
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        },
        timeout=15,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"IAM token error: {resp.text}")

    data = resp.json()
    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = now + 3300  # tokens valid 60 min, refresh at 55
    return _token_cache["token"]


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None   # pass back the thread_id from previous turns


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


class NewThreadResponse(BaseModel):
    thread_id: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat/new", response_model=NewThreadResponse)
def new_thread():
    """
    Call this once at the start of a session to get a thread_id.
    Pass that thread_id in every subsequent /chat call.
    """
    token = get_iam_token()
    resp = httpx.post(
        f"{WXO_INSTANCE_URL}/v1/threads",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "agent_id": WXO_AGENT_ID,
            "title": "SalesBud session",
        },
        timeout=20,
    )
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=502, detail=f"Thread creation failed: {resp.text}")

    thread_id = resp.json().get("id") or resp.json().get("thread_id")
    return {"thread_id": thread_id}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Send a message to the SalesBud Orchestrator and return its reply.
    Pass thread_id from a previous /chat/new call to maintain conversation state.
    If no thread_id is provided, a new thread is created automatically.
    """
    token = get_iam_token()

    # Auto-create thread if not provided
    thread_id = req.thread_id
    if not thread_id:
        thread_resp = new_thread()
        thread_id = thread_resp.thread_id

    # Build messages array — Orchestrate uses OpenAI-style format
    messages = [{"role": "user", "content": req.message}]

    resp = httpx.post(
        f"{WXO_INSTANCE_URL}/v1/orchestrate/{WXO_AGENT_ID}/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "messages": messages,
            "thread_id": thread_id,
            "stream": False,
        },
        timeout=120,   # Orchestrate multi-agent flows can take 60-90s
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Orchestrate error: {resp.text}")

    data = resp.json()

    # Extract the assistant reply — structure mirrors OpenAI chat completions
    try:
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        reply = str(data)   # fallback: return raw if structure differs

    # thread_id may come back in the response or stay as-is
    returned_thread_id = data.get("thread_id", thread_id)

    return {"reply": reply, "thread_id": returned_thread_id}