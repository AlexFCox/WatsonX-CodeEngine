"""
SalesBud – watsonx Orchestrate Chat Proxy
Deployed to IBM Code Engine (group13, eu-de)

The Orchestrate API streams responses as SSE chunks.
This proxy collects all chunks and returns a single JSON response to the frontend.
"""

import os
import time
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json

app = FastAPI(title="SalesBud Proxy", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "https://watson-x-planted-frontend.vercel.app", 
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
IBM_API_KEY       = os.environ["IBM_API_KEY"]
WXO_INSTANCE_GUID = os.environ["WXO_INSTANCE_GUID"]  # 946df986-9572-455e-bc1f-be9c5c5ec40e
WXO_AGENT_ID      = os.environ["WXO_AGENT_ID"]        # 155d8a0d-0649-4f95-bb39-c1f5389c4685
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
    message: str
    thread_id: Optional[str] = None  # pass back on subsequent turns to maintain context


class ChatResponse(BaseModel):
    reply: str
    thread_id: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Send a message to SalesBud Orchestrator.
    Handles the SSE streaming response and returns the full reply as JSON.
    Pass thread_id from previous responses to maintain conversation state.
    """
    token = get_iam_token()

    # Build payload — include thread_id if continuing a conversation
    payload = {
        "messages": [{"role": "user", "content": req.message}],
    }
    if req.thread_id:
        payload["thread_id"] = req.thread_id

    url = f"{WXO_BASE_URL}/v1/orchestrate/{WXO_AGENT_ID}/chat/completions"

    full_text = ""
    thread_id = req.thread_id

    with httpx.stream(
        "POST",
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
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

            # Grab thread_id from any chunk that has it
            if not thread_id and chunk.get("thread_id"):
                thread_id = chunk["thread_id"]

            # The completed message has the full text — prefer that
            if chunk.get("object") == "thread.message.completed":
                try:
                    full_text = chunk["data"]["message"]["content"][0]["text"]
                except (KeyError, IndexError):
                    pass
                break

            # Otherwise accumulate delta chunks
            try:
                delta = chunk["choices"][0]["delta"].get("content", "")
                full_text += delta
            except (KeyError, IndexError):
                continue

    return {"reply": full_text, "thread_id": thread_id}