"""
SalesBud – watsonx Orchestrate Chat Proxy
Deployed to IBM Code Engine (group13, eu-de)

Uses /chat/completions with X-IBM-THREAD-ID header for session persistence.
thread_id is returned in the response and passed back on subsequent requests.
Orchestrate maintains flow state within a thread — no need to replay history.
"""

import os
import sys
import time
import httpx
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="SalesBud Proxy", version="2.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://watson-x-planted-frontend.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
IBM_API_KEY       = os.environ["IBM_API_KEY"]
WXO_INSTANCE_GUID = os.environ["WXO_INSTANCE_GUID"]
WXO_AGENT_ID      = os.environ["WXO_AGENT_ID"]
WXO_BASE_URL      = f"https://api.eu-de.watson-orchestrate.cloud.ibm.com/instances/{WXO_INSTANCE_GUID}"
IAM_TOKEN_URL     = "https://iam.cloud.ibm.com/identity/token"

# ── Token cache ───────────────────────────────────────────────────────────────
_token_cache: dict = {"token": None, "expires_at": 0}


def get_iam_token() -> str:
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
    _token_cache["expires_at"] = now + 3300
    return _token_cache["token"]


# ── Models ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str                     # latest user message only
    thread_id: Optional[str] = None  # None on first message


class ChatResponse(BaseModel):
    reply: str
    thread_id: Optional[str] = None  # return to frontend for next request


# ── SSE helpers ───────────────────────────────────────────────────────────────
FLOW_NOISE_PHRASES = [
    "a new flow has started",
    "this chat session is currently dedicated to the flow",
    "will resume once the flow is complete",
]

def is_flow_noise(text: str) -> bool:
    low = text.lower()
    return any(phrase in low for phrase in FLOW_NOISE_PHRASES)


def _call_agent(message: str, thread_id: Optional[str]) -> tuple[str, Optional[str]]:
    """
    Send a single user message to WXO via /chat/completions.
    Passes thread_id as X-IBM-THREAD-ID header for session continuity.
    Returns (reply_text, thread_id).
    """
    token = get_iam_token()

    # Only send the latest user message — thread maintains history
    payload = {
        "messages": [
            {"role": "user", "content": message}
        ],
        "stream": True,
    }

    url = f"{WXO_BASE_URL}/v1/orchestrate/{WXO_AGENT_ID}/chat/completions"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Pass thread_id as header to continue existing session
    if thread_id:
        headers["X-IBM-THREAD-ID"] = thread_id

    reply_text = ""
    returned_thread_id = thread_id
    final_texts = []

    with httpx.stream(
        "POST",
        url,
        headers=headers,
        json=payload,
        timeout=180,
    ) as response:
        if response.status_code != 200:
            body = response.read().decode()
            print(f"[ERROR] Orchestrate {response.status_code}: {body}", file=sys.stderr, flush=True)
            raise HTTPException(
                status_code=502,
                detail=f"Orchestrate error {response.status_code}: {body}"
            )

        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue

            raw = line[len("data: "):]
            if raw.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue

            obj = chunk.get("object", "")
            print(f"[SSE] object={obj!r}", file=sys.stderr, flush=True)

            # Capture thread_id from response
            if not returned_thread_id:
                tid = chunk.get("thread_id")
                if tid:
                    returned_thread_id = tid
                    print(f"[SSE] thread_id captured: {tid}", file=sys.stderr, flush=True)

            # Completed message — collect non-noise replies
            if obj == "thread.message.completed":
                try:
                    text = chunk["data"]["message"]["content"][0]["text"]
                    print(f"[SSE] completed: {text[:120]!r}", file=sys.stderr, flush=True)
                    if not is_flow_noise(text):
                        final_texts.append(text)
                except (KeyError, IndexError):
                    pass
                continue

            # Delta accumulation fallback
            try:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    reply_text += delta
            except (KeyError, IndexError):
                continue

    # Return last non-noise completed message, or delta fallback
    result = final_texts[-1] if final_texts else reply_text.strip()
    return result, returned_thread_id


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply, thread_id = _call_agent(req.message, req.thread_id)
    return {"reply": reply, "thread_id": thread_id}