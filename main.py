"""
SalesBud – watsonx Orchestrate Chat Proxy v2.3
Deployed to IBM Code Engine (group13, eu-de)

When flow noise is detected, proxy polls GET /runs/{run_id}/events
until run.completed appears, then extracts the real message.
Tool-agnostic — works for any number of async tools firing simultaneously.
"""

import os
import sys
import re
import time
import httpx
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="SalesBud Proxy", version="2.3.0")

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

MAX_POLL_ATTEMPTS = 20   # poll run events up to 20 times
POLL_DELAY_S      = 2.0  # 2s between polls = up to 40s total wait

# ── Token cache ───────────────────────────────────────────────────────────────
_token_cache: dict = {"token": None, "expires_at": 0}


def get_iam_token() -> str:
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"]:
        return _token_cache["token"]
    resp = httpx.post(
        IAM_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"apikey": IBM_API_KEY, "grant_type": "urn:ibm:params:oauth:grant-type:apikey"},
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


# ── Helpers ───────────────────────────────────────────────────────────────────
FLOW_NOISE_PHRASES = [
    "a new flow has started",
    "this chat session is currently dedicated to the flow",
    "will resume once the flow is complete",
]

def is_flow_noise(text: str) -> bool:
    return any(p in text.lower() for p in FLOW_NOISE_PHRASES)


def _poll_run_events(run_id: str) -> Optional[str]:
    """
    Poll GET /api/v1/orchestrate/runs/{run_id}/events until run.completed.
    Returns the final message.completed text, or None if timed out.
    This works for ALL async tools simultaneously — no per-tool logic needed.
    """
    token = get_iam_token()
    url = f"{WXO_BASE_URL}/api/v1/orchestrate/runs/{run_id}/events"
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        time.sleep(POLL_DELAY_S)
        print(f"[POLL] run events attempt {attempt}/{MAX_POLL_ATTEMPTS} for run_id={run_id}", file=sys.stderr, flush=True)

        try:
            resp = httpx.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                print(f"[POLL] events error {resp.status_code}: {resp.text[:200]}", file=sys.stderr, flush=True)
                continue

            events = resp.json()
            if not isinstance(events, list):
                continue

            print(f"[POLL] got {len(events)} events", file=sys.stderr, flush=True)

            # Check if run is complete
            run_complete = any(
                e.get("event") in ("run.completed", "done")
                for e in events
            )

            # Extract last message.completed text that isn't flow noise
            message_text = None
            for e in reversed(events):
                if e.get("event") == "message.completed":
                    try:
                        data = e.get("data", {})
                        content = data.get("message", {}).get("content", [])
                        if isinstance(content, list) and content:
                            text = content[0].get("text", "")
                        elif isinstance(content, str):
                            text = content
                        else:
                            text = ""

                        if text and not is_flow_noise(text):
                            message_text = text
                            break
                    except (KeyError, IndexError, TypeError):
                        continue

            if run_complete and message_text:
                print(f"[POLL] run complete, got message: {message_text[:100]!r}", file=sys.stderr, flush=True)
                return message_text

            if run_complete and not message_text:
                print(f"[POLL] run complete but no usable message found", file=sys.stderr, flush=True)
                return None

        except Exception as e:
            print(f"[POLL] error: {e}", file=sys.stderr, flush=True)

    print(f"[POLL] timed out after {MAX_POLL_ATTEMPTS} attempts", file=sys.stderr, flush=True)
    return None


def _single_request(message: str, thread_id: Optional[str]) -> tuple[str, Optional[str], Optional[str]]:
    """
    One SSE request. Returns (reply_text, thread_id, run_id).
    Captures run_id from SSE stream for polling.
    """
    token = get_iam_token()
    payload = {"messages": [{"role": "user", "content": message}], "stream": True}
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if thread_id:
        headers["X-IBM-THREAD-ID"] = thread_id

    url = f"{WXO_BASE_URL}/v1/orchestrate/{WXO_AGENT_ID}/chat/completions"
    reply_text   = ""
    returned_tid = thread_id
    returned_rid = None  # run_id
    final_texts  = []

    with httpx.stream("POST", url, headers=headers, json=payload, timeout=180) as response:
        if response.status_code != 200:
            body = response.read().decode()
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

            # Capture thread_id and run_id
            if not returned_tid:
                tid = chunk.get("thread_id")
                if tid:
                    returned_tid = tid
                    print(f"[SSE] thread_id={tid}", file=sys.stderr, flush=True)

            if not returned_rid:
                rid = chunk.get("run_id") or chunk.get("id")
                if rid and rid != returned_tid:  # avoid confusing thread_id with run_id
                    returned_rid = rid
                    print(f"[SSE] run_id={rid}", file=sys.stderr, flush=True)

            if obj == "thread.message.completed":
                try:
                    text = chunk["data"]["message"]["content"][0]["text"]
                    print(f"[SSE] completed: {text[:100]!r}", file=sys.stderr, flush=True)
                    final_texts.append(text)
                except (KeyError, IndexError):
                    pass
                continue

            # Also check for run_id in data
            data = chunk.get("data", {})
            if isinstance(data, dict) and not returned_rid:
                rid = data.get("run_id") or data.get("id")
                if rid:
                    returned_rid = rid
                    print(f"[SSE] run_id from data={rid}", file=sys.stderr, flush=True)

            try:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    reply_text += delta
            except (KeyError, IndexError):
                continue

    result = final_texts[-1] if final_texts else reply_text.strip()
    return result, returned_tid, returned_rid


def _call_agent(message: str, thread_id: Optional[str]) -> tuple[str, Optional[str]]:
    """
    Full agent call. If flow noise detected, polls run events until complete.
    Tool-agnostic — handles any number of simultaneous async tool calls.
    """
    reply, tid, rid = _single_request(message, thread_id)

    if not is_flow_noise(reply):
        return reply, tid

    # Flow is running — poll run events endpoint
    print(f"[FLOW] Flow noise detected. thread_id={tid} run_id={rid}", file=sys.stderr, flush=True)

    if rid:
        # Best approach: poll the run events endpoint
        polled_reply = _poll_run_events(rid)
        if polled_reply:
            return polled_reply, tid
    else:
        # Fallback: no run_id captured, poll thread with empty message
        print(f"[FLOW] No run_id captured, falling back to empty message poll", file=sys.stderr, flush=True)
        for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
            time.sleep(POLL_DELAY_S)
            print(f"[POLL] empty msg attempt {attempt}", file=sys.stderr, flush=True)
            poll_reply, tid, _ = _single_request("", tid)
            if poll_reply and not is_flow_noise(poll_reply):
                return poll_reply, tid

    return "Still processing — please wait a moment and try again.", tid


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.3.0"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply, thread_id = _call_agent(req.message, req.thread_id)
    return {"reply": reply, "thread_id": thread_id}