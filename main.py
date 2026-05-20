"""
SalesBud – watsonx Orchestrate Chat Proxy
Deployed to IBM Code Engine (group13, eu-de)

Uses the /api/v1/orchestrate/runs/stream endpoint with thread_id for session persistence.
Orchestrate maintains conversation state — no need to replay full history.
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
    message: str                  # latest user message only
    thread_id: Optional[str] = None  # None on first message, returned from prior response


class ChatResponse(BaseModel):
    reply: str
    thread_id: Optional[str] = None  # return to frontend for next request


# ── SSE helper ────────────────────────────────────────────────────────────────
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
    Send a single user message to WXO via /runs/stream endpoint.
    Returns (reply_text, thread_id).
    thread_id is returned by Orchestrate and must be sent on subsequent requests
    to maintain session state.
    """
    token = get_iam_token()

    payload = {
        "agent_id": WXO_AGENT_ID,
        "message": {
            "role": "user",
            "content": message,
        },
    }

    # Include thread_id if we have one (continues existing session)
    if thread_id:
        payload["thread_id"] = thread_id

    url = f"{WXO_BASE_URL}/api/v1/orchestrate/runs/stream"

    reply_text = ""
    returned_thread_id = thread_id  # keep existing if not returned

    with httpx.stream(
        "POST",
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
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
            if not line:
                continue

            print(f"[SSE raw] {line[:200]}", file=sys.stderr, flush=True)

            if not line.startswith("data: "):
                continue

            raw = line[len("data: "):]
            if raw.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue

            event = chunk.get("event", "")
            data  = chunk.get("data", {})

            print(f"[SSE] event={event!r}", file=sys.stderr, flush=True)

            # Extract thread_id whenever it appears
            if isinstance(data, dict):
                tid = data.get("thread_id") or data.get("id")
                if tid and not returned_thread_id:
                    returned_thread_id = tid
                    print(f"[SSE] thread_id captured: {tid}", file=sys.stderr, flush=True)

            # Message completed — extract text
            if event == "message.completed":
                try:
                    content = data.get("message", {}).get("content", [])
                    if isinstance(content, list) and content:
                        text = content[0].get("text", "")
                    elif isinstance(content, str):
                        text = content
                    else:
                        text = ""

                    print(f"[SSE] message.completed text: {text[:120]!r}", file=sys.stderr, flush=True)

                    if text and not is_flow_noise(text):
                        reply_text = text

                except (KeyError, IndexError, TypeError) as e:
                    print(f"[SSE] parse error: {e}", file=sys.stderr, flush=True)
                continue

            # run.completed — stream is done
            if event in ("run.completed", "done"):
                break

            # flow.slot.listen — flow is waiting for input, keep reading
            if event == "flow.slot.listen":
                print(f"[SSE] flow.slot.listen — flow waiting for input", file=sys.stderr, flush=True)
                continue

            # Delta text accumulation (fallback)
            if event == "message.delta":
                try:
                    delta = data.get("delta", {}).get("content", "")
                    if isinstance(delta, str):
                        reply_text += delta
                    elif isinstance(delta, list) and delta:
                        reply_text += delta[0].get("text", "")
                except (KeyError, TypeError):
                    pass

    return reply_text.strip(), returned_thread_id


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Send a single user message to SalesBud Orchestrator.
    Pass thread_id from previous response to continue the session.
    Returns reply text and thread_id for next request.
    """
    reply, thread_id = _call_agent(req.message, req.thread_id)
    return {"reply": reply, "thread_id": thread_id}