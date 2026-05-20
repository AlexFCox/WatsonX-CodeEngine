"""
SalesBud – watsonx Orchestrate Chat Proxy
Deployed to IBM Code Engine (group13, eu-de)

Uses /chat/completions with X-IBM-THREAD-ID header for session persistence.
When a flow fires, we poll the same thread until we get a real reply.
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

app = FastAPI(title="SalesBud Proxy", version="2.1.0")

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

MAX_POLL_ATTEMPTS = 10
POLL_DELAY_S      = 2.0

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
    message: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    thread_id: Optional[str] = None


# ── SSE helpers ───────────────────────────────────────────────────────────────
FLOW_NOISE_PHRASES = [
    "a new flow has started",
    "this chat session is currently dedicated to the flow",
    "will resume once the flow is complete",
]

def is_flow_noise(text: str) -> bool:
    return any(p in text.lower() for p in FLOW_NOISE_PHRASES)


def _single_request(message: str, thread_id: Optional[str]) -> tuple[str, Optional[str]]:
    """
    Make one SSE request to Orchestrate.
    Returns (reply_text, thread_id).
    reply_text may be flow noise — caller decides what to do.
    """
    token = get_iam_token()

    payload = {
        "messages": [{"role": "user", "content": message}],
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if thread_id:
        headers["X-IBM-THREAD-ID"] = thread_id

    url = f"{WXO_BASE_URL}/v1/orchestrate/{WXO_AGENT_ID}/chat/completions"

    reply_text      = ""
    returned_tid    = thread_id
    final_texts     = []

    with httpx.stream("POST", url, headers=headers, json=payload, timeout=180) as response:
        if response.status_code != 200:
            body = response.read().decode()
            print(f"[ERROR] {response.status_code}: {body}", file=sys.stderr, flush=True)
            raise HTTPException(status_code=502, detail=f"Orchestrate error {response.status_code}: {body}")

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
            print(f"[SSE] {obj}", file=sys.stderr, flush=True)

            # Capture thread_id
            if not returned_tid:
                tid = chunk.get("thread_id")
                if tid:
                    returned_tid = tid
                    print(f"[SSE] thread_id={tid}", file=sys.stderr, flush=True)

            # Completed message
            if obj == "thread.message.completed":
                try:
                    text = chunk["data"]["message"]["content"][0]["text"]
                    print(f"[SSE] completed: {text[:100]!r}", file=sys.stderr, flush=True)
                    final_texts.append(text)
                except (KeyError, IndexError):
                    pass
                continue

            # Delta fallback
            try:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    reply_text += delta
            except (KeyError, IndexError):
                continue

    result = final_texts[-1] if final_texts else reply_text.strip()
    return result, returned_tid


def _call_agent(message: str, thread_id: Optional[str]) -> tuple[str, Optional[str]]:
    """
    Call Orchestrate and poll on the same thread if a flow is in progress.
    On flow noise: wait POLL_DELAY_S seconds, then send an empty ping on
    the same thread to resume — Orchestrate will continue the flow.
    """
    # First real request with the user's message
    reply, tid = _single_request(message, thread_id)

    if not is_flow_noise(reply):
        return reply, tid

    # Flow is running — poll the same thread with empty ping
    print(f"[POLL] Flow detected, polling thread {tid}", file=sys.stderr, flush=True)

    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        time.sleep(POLL_DELAY_S)
        print(f"[POLL] attempt {attempt}/{MAX_POLL_ATTEMPTS}", file=sys.stderr, flush=True)

        # Send empty message to resume/check the flow on the same thread
        poll_reply, tid = _single_request("", tid)

        if poll_reply and not is_flow_noise(poll_reply):
            print(f"[POLL] got real reply on attempt {attempt}", file=sys.stderr, flush=True)
            return poll_reply, tid

        print(f"[POLL] still noise or empty, retrying...", file=sys.stderr, flush=True)

    # Give up
    return "Still processing — please wait a moment and try again.", tid


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.1.0"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply, thread_id = _call_agent(req.message, req.thread_id)
    return {"reply": reply, "thread_id": thread_id}