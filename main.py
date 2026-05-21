"""
SalesBud – watsonx Orchestrate Chat Proxy v3.1
Deployed to IBM Code Engine (group13, eu-de)

CONFIRMED API FACTS (cross-referenced against IBM ADK docs + working v2.2):

  Chat endpoint (cloud production):
    POST /api/v1/orchestrate/{agent_id}/chat/completions
    Header: Authorization: Bearer <iam_token>
    Header: X-IBM-THREAD-ID: <thread_id>   ← session continuity, optional first turn
    Body:   {"messages": [{"role": "user", "content": "..."}], "stream": true}
    Response: SSE stream, each line "data: {...}"
              chunk.object = "thread.message.delta"     ← streaming text
              chunk.object = "thread.message.completed" ← final full text
              chunk.thread_id                            ← capture and reuse
              "[DONE]"                                   ← stream end marker

  Run status endpoint (for async_wait polling):
    GET /api/v1/orchestrate/runs/{run_id}
    Header: Authorization: Bearer <iam_token>
    Response: {"status": "pending|running|completed|async_wait|async_completed|failed|cancelled"}

  IAM token exchange (standard IBM Cloud):
    POST https://iam.cloud.ibm.com/identity/token
    Body: grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=<key>

ASYNC FLOW HANDLING (Approach B):
  When a Flow-backed tool is called, the SSE stream ends without
  thread.message.completed. The proxy detects this (empty reply after [DONE])
  and polls GET /runs/{run_id} until async_completed, then sends a follow-up
  message to resume the agent.

  Run IDs are extracted from SSE chunks that contain chunk.id or chunk.run_id.
"""

import os
import sys
import time
import json
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="SalesBud Proxy", version="3.1.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://watson-x-planted-frontend.*\.vercel\.app",
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
IBM_API_KEY       = os.environ["IBM_API_KEY"]
WXO_INSTANCE_GUID = os.environ.get("WXO_INSTANCE_GUID", "946df986-9572-455e-bc1f-be9c5c5ec40e")

# Cloud production base — confirmed working in v2.2
# Endpoint: /api/v1/orchestrate/{agent_id}/chat/completions
WXO_BASE_URL      = "https://eu-de.watson-orchestrate.cloud.ibm.com"

# Agent IDs
AUTH_AGENT_ID     = os.environ.get("AUTH_AGENT_ID",     "6c98d390-dfc7-4b8f-b11e-bf36dd148c80")
ORCHESTRATOR_ID   = os.environ.get("ORCHESTRATOR_ID",   "b2340dd1-11f7-4234-b073-125d85d78c98")

IAM_URL           = "https://iam.cloud.ibm.com/identity/token"

# Async loop settings
POLL_INTERVAL_S   = 1.5   # seconds between status polls
POLL_MAX_ATTEMPTS = 40    # 40 × 1.5s = 60s max wait per async tool
ASYNC_LOOP_MAX    = 5     # safety valve: max tool-call rounds per user turn

# ── IAM TOKEN CACHE ───────────────────────────────────────────────────────────
_token_cache: dict = {}

async def get_iam_token() -> str:
    now = time.time()
    if _token_cache.get("token") and now < _token_cache.get("expires_at", 0) - 30:
        return _token_cache["token"]

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            IAM_URL,
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": IBM_API_KEY,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        data = resp.json()
        _token_cache["token"]      = data["access_token"]
        _token_cache["expires_at"] = now + data.get("expires_in", 3600)
        print(f"[proxy] IAM token refreshed", flush=True)
        return _token_cache["token"]

# ── PYDANTIC MODELS ───────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    thread_id: Optional[str] = None
    owner_id: Optional[str] = None
    user_email: Optional[str] = None

class AuthRequest(BaseModel):
    message: str
    history: list[Message] = []
    thread_id: Optional[str] = None

# ── FLOW NOISE FILTER ─────────────────────────────────────────────────────────
# The agent emits internal status messages when calling flows. Filter them out
# so they don't appear as the final reply to the user.
FLOW_NOISE_PATTERNS = [
    "starting flow", "flow started", "flow instance",
    "running tool", "calling tool", "tool call",
    "fetching", "looking up", "searching for",
    "please wait", "one moment",
]

def is_flow_noise(text: str) -> bool:
    t = text.lower().strip()
    if len(t) < 5:
        return True
    return any(p in t for p in FLOW_NOISE_PATTERNS)

# ── CORE: SSE STREAM CONSUMER ─────────────────────────────────────────────────
async def call_agent_stream(
    agent_id: str,
    messages: list[dict],
    thread_id: Optional[str],
) -> dict:
    """
    POST to /api/v1/orchestrate/{agent_id}/chat/completions with stream=true.
    Consumes the SSE stream and returns:
      {
        "reply":     str,          # assembled final text (may be empty on async_wait)
        "thread_id": str | None,   # capture and reuse next turn
        "run_id":    str | None,   # needed for polling async runs
        "finished":  bool,         # True = got thread.message.completed or [DONE] with text
        "async_wait": bool,        # True = stream ended without a final answer
      }
    """
    token  = await get_iam_token()
    url    = f"{WXO_BASE_URL}/api/v1/orchestrate/{agent_id}/chat/completions"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    if thread_id:
        headers["X-IBM-THREAD-ID"] = thread_id

    payload = {
        "messages": messages,
        "stream":   True,
    }

    reply_chunks: list[str] = []
    final_texts:  list[str] = []
    returned_thread_id      = thread_id
    run_id                  = None

    print(f"[proxy] → POST {url} thread={thread_id}", flush=True)

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:

                if resp.status_code != 200:
                    body = await resp.aread()
                    print(f"[proxy] WxO error {resp.status_code}: {body[:300]}", flush=True)
                    return {
                        "reply": f"WxO error {resp.status_code}: {body[:200].decode()}",
                        "thread_id": returned_thread_id,
                        "run_id": None,
                        "finished": False,
                        "async_wait": False,
                    }

                async for raw_line in resp.aiter_lines():
                    if not raw_line.strip():
                        continue

                    # Strip SSE "data: " prefix
                    line = raw_line
                    if line.startswith("data:"):
                        line = line[5:].strip()

                    if not line or line == "[DONE]":
                        break

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    obj = chunk.get("object", "")
                    print(f"[proxy] SSE object={obj!r}", flush=True)

                    # ── Capture thread_id ────────────────────────────────
                    if not returned_thread_id:
                        tid = chunk.get("thread_id")
                        if tid:
                            returned_thread_id = tid
                            print(f"[proxy] thread_id={tid}", flush=True)

                    # ── Capture run_id (needed for async polling) ────────
                    if not run_id:
                        run_id = (
                            chunk.get("run_id") or
                            chunk.get("id") or
                            chunk.get("data", {}).get("run_id")
                        )

                    # ── Final completed message ──────────────────────────
                    if obj == "thread.message.completed":
                        try:
                            text = chunk["data"]["message"]["content"][0]["text"]
                            print(f"[proxy] completed: {text[:120]!r}", flush=True)
                            if not is_flow_noise(text):
                                final_texts.append(text)
                        except (KeyError, IndexError, TypeError):
                            pass
                        continue

                    # ── Streaming delta ──────────────────────────────────
                    if obj in ("thread.message.delta", "chat.completion.chunk"):
                        try:
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                reply_chunks.append(delta)
                        except (KeyError, IndexError, TypeError):
                            pass
                        continue

    except httpx.ReadTimeout:
        print("[proxy] SSE stream timed out", flush=True)
    except Exception as e:
        print(f"[proxy] SSE exception: {e}", flush=True)

    # Prefer thread.message.completed text; fall back to accumulated deltas
    reply = (final_texts[-1] if final_texts else "".join(reply_chunks)).strip()

    # async_wait = stream ended with no usable reply
    async_wait = not reply

    return {
        "reply":      reply,
        "thread_id":  returned_thread_id,
        "run_id":     run_id,
        "finished":   bool(reply),
        "async_wait": async_wait,
    }

# ── ASYNC WAIT POLLING ────────────────────────────────────────────────────────
async def poll_run_until_done(run_id: str, token: str) -> Optional[dict]:
    """
    Poll GET /api/v1/orchestrate/runs/{run_id} until status leaves async_wait.
    Returns the final run dict, or None on timeout.
    """
    if not run_id:
        print("[proxy] poll: no run_id", flush=True)
        return None

    url     = f"{WXO_BASE_URL}/api/v1/orchestrate/runs/{run_id}"
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(POLL_MAX_ATTEMPTS):
        await asyncio.sleep(POLL_INTERVAL_S)
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code != 200:
                    print(f"[proxy] poll attempt {attempt+1}: HTTP {resp.status_code}", flush=True)
                    continue

                data   = resp.json()
                status = data.get("status", "unknown")
                print(f"[proxy] poll attempt {attempt+1}: status={status}", flush=True)

                if status in ("async_completed", "completed"):
                    return data
                if status in ("failed", "cancelled", "expired"):
                    print(f"[proxy] run terminal status: {status}", flush=True)
                    return data

        except Exception as e:
            print(f"[proxy] poll exception attempt {attempt+1}: {e}", flush=True)

    print(f"[proxy] poll timed out after {POLL_MAX_ATTEMPTS} attempts", flush=True)
    return None

# ── MAIN AGENT TURN (with async_wait loop) ────────────────────────────────────
async def run_agent_turn(
    agent_id: str,
    user_message: str,
    history: list[Message],
    thread_id: Optional[str],
) -> dict:
    """
    Execute one user turn end-to-end, handling async_wait loops.

    Strategy:
      1. Build messages list (history + new user message).
         On first turn: full history so agent has context.
         On subsequent turns (thread established): just the new message —
         WxO thread maintains state.
      2. Stream the response.
      3. If async_wait detected: poll for run completion, then send
         a follow-up "please continue" message on the same thread.
      4. Repeat up to ASYNC_LOOP_MAX times.

    Returns { reply, thread_id, async_rounds }
    """

    # Build initial messages list
    # First turn: include history so agent has full context
    # Subsequent turns: thread_id carries state, just send latest message
    if thread_id:
        messages = [{"role": "user", "content": user_message}]
    else:
        messages = [{"role": m.role, "content": m.content} for m in history]
        messages.append({"role": "user", "content": user_message})

    current_thread_id = thread_id
    async_rounds      = 0

    for loop in range(ASYNC_LOOP_MAX + 1):
        result = await call_agent_stream(
            agent_id  = agent_id,
            messages  = messages,
            thread_id = current_thread_id,
        )

        current_thread_id = result["thread_id"] or current_thread_id

        # ── Clean completion ──────────────────────────────────────────────
        if result["finished"]:
            return {
                "reply":        result["reply"],
                "thread_id":    current_thread_id,
                "async_rounds": async_rounds,
            }

        # ── async_wait — a flow tool was called ───────────────────────────
        if result["async_wait"]:
            async_rounds += 1
            run_id = result.get("run_id")
            print(f"[proxy] async_wait on loop {loop}, run_id={run_id}", flush=True)

            token       = await get_iam_token()
            completed   = await poll_run_until_done(run_id, token)

            if not completed:
                return {
                    "reply":        "A background tool is taking too long. Please try again.",
                    "thread_id":    current_thread_id,
                    "async_rounds": async_rounds,
                }

            status = completed.get("status", "")
            if status in ("failed", "cancelled", "expired"):
                return {
                    "reply":        f"A background tool ended with status: {status}. Please try again.",
                    "thread_id":    current_thread_id,
                    "async_rounds": async_rounds,
                }

            # Tool completed — send a follow-up on the same thread to resume
            print(f"[proxy] flow completed, resuming agent on thread={current_thread_id}", flush=True)
            messages = [{"role": "user", "content": "Please continue."}]
            continue

        # ── Unexpected empty reply (not async_wait, not finished) ─────────
        break

    return {
        "reply":        "Something went wrong. Please try again.",
        "thread_id":    current_thread_id,
        "async_rounds": async_rounds,
    }

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.1.0"}


@app.post("/auth")
async def auth(req: AuthRequest):
    """Auth route — calls Auth Agent to look up Salesforce User ID."""
    try:
        result = await run_agent_turn(
            agent_id    = AUTH_AGENT_ID,
            user_message = req.message,
            history     = req.history,
            thread_id   = req.thread_id,
        )
        updated_history = [m.dict() for m in req.history] + [
            {"role": "user",      "content": req.message},
            {"role": "assistant", "content": result["reply"]},
        ]
        return {
            "reply":     result["reply"],
            "thread_id": result["thread_id"],
            "history":   updated_history,
        }
    except Exception as e:
        print(f"[proxy] /auth error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(req: ChatRequest):
    """Main chat route — calls Orchestrator V2 with async_wait loop."""
    try:
        result = await run_agent_turn(
            agent_id    = ORCHESTRATOR_ID,
            user_message = req.message,
            history     = req.history,
            thread_id   = req.thread_id,
        )
        updated_history = [m.dict() for m in req.history] + [
            {"role": "user",      "content": req.message},
            {"role": "assistant", "content": result["reply"]},
        ]
        return {
            "reply":        result["reply"],
            "thread_id":    result["thread_id"],
            "async_rounds": result["async_rounds"],
            "history":      updated_history,
        }
    except Exception as e:
        print(f"[proxy] /chat error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))