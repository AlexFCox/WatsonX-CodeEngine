"""
SalesBud – watsonx Orchestrate Chat Proxy
Deployed to IBM Code Engine (group13, eu-de)

Orchestrate does not support thread_id continuation.
We send the full conversation history on every request instead.
"""

import os
import sys
import time
import httpx
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SalesBud Proxy", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://watson-x-planted-frontend.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
IBM_API_KEY        = os.environ["IBM_API_KEY"]
WXO_INSTANCE_GUID  = os.environ["WXO_INSTANCE_GUID"]
WXO_AGENT_ID       = os.environ["WXO_AGENT_ID"]
WXO_BASE_URL       = f"https://api.eu-de.watson-orchestrate.cloud.ibm.com/instances/{WXO_INSTANCE_GUID}"
IAM_TOKEN_URL      = "https://iam.cloud.ibm.com/identity/token"

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
class Message(BaseModel):
    role: str      # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]   # full conversation history including latest user message


class ChatResponse(BaseModel):
    reply: str


# ── SSE helper ────────────────────────────────────────────────────────────────
# Flow notification strings from Orchestrate that should NOT be returned to the user.
# These are internal status messages emitted when a tool/flow starts or resumes.
FLOW_NOISE = (
    "a new flow has started",
    "this chat session is currently dedicated to the flow",
    "will resume once the flow is complete",
)

def _is_flow_noise(text: str) -> bool:
    low = text.lower()
    return any(phrase in low for phrase in FLOW_NOISE)


def _call_agent(agent_id: str, messages: list[Message]) -> str:
    """
    Send full conversation history to a given WXO agent.
    Returns the final completed reply text, skipping flow noise events.

    Key behaviour:
    - Collects ALL thread.message.completed events, not just the first.
    - Skips any whose text is a flow status notification.
    - Returns the last non-noise completed message text.
    - Falls back to accumulated delta text if no completed message found.
    """
    token = get_iam_token()

    payload = {
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }

    url = f"{WXO_BASE_URL}/v1/orchestrate/{agent_id}/chat/completions"

    delta_text  = ""   # accumulated from streaming deltas
    final_texts = []   # all thread.message.completed texts (excluding noise)

    with httpx.stream(
        "POST",
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,   # flows can take time — give them 3 minutes
    ) as response:
        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Orchestrate error {response.status_code}: {response.read().decode()}"
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

            # Log every event type for debugging
            print(f"[SSE] object={obj!r}", file=sys.stderr, flush=True)

            # Completed message event — collect but skip flow noise
            if obj == "thread.message.completed":
                try:
                    text = chunk["data"]["message"]["content"][0]["text"]
                    print(f"[SSE] completed text preview: {text[:120]!r}", file=sys.stderr, flush=True)
                    if not _is_flow_noise(text):
                        final_texts.append(text)
                        delta_text = ""  # reset delta accumulation
                except (KeyError, IndexError):
                    pass
                # Do NOT break — keep reading for more events after flow completes
                continue

            # Delta streaming chunks
            try:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    delta_text += delta
            except (KeyError, IndexError):
                continue

    # Return the last non-noise completed message, or fall back to deltas
    if final_texts:
        return final_texts[-1]
    return delta_text.strip()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Send full conversation history to SalesBud Orchestrator.
    Orchestrate does not persist threads, so we replay the full history each time.
    """
    reply = _call_agent(WXO_AGENT_ID, req.messages)
    return {"reply": reply}