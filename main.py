"""
SalesBud – watsonx Orchestrate Chat Proxy v2.4
Deployed to IBM Code Engine (group13, eu-de)

Enhanced SSE logging to find correct agent run_id format.
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

app = FastAPI(title="SalesBud Proxy", version="2.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://watson-x-planted-frontend.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IBM_API_KEY       = os.environ["IBM_API_KEY"]
WXO_INSTANCE_GUID = os.environ["WXO_INSTANCE_GUID"]
WXO_AGENT_ID      = os.environ["WXO_AGENT_ID"]
WXO_BASE_URL      = f"https://api.eu-de.watson-orchestrate.cloud.ibm.com/instances/{WXO_INSTANCE_GUID}"
IAM_TOKEN_URL     = "https://iam.cloud.ibm.com/identity/token"

MAX_POLL_ATTEMPTS = 20
POLL_DELAY_S      = 2.0

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


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    thread_id: Optional[str] = None


FLOW_NOISE_PHRASES = [
    "a new flow has started",
    "this chat session is currently dedicated to the flow",
    "will resume once the flow is complete",
]

def is_flow_noise(text: str) -> bool:
    return any(p in text.lower() for p in FLOW_NOISE_PHRASES)


def _try_run_events(run_id: str) -> Optional[str]:
    """Try GET /api/v1/orchestrate/runs/{run_id}/events"""
    token = get_iam_token()
    url = f"{WXO_BASE_URL}/api/v1/orchestrate/runs/{run_id}/events"
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        time.sleep(POLL_DELAY_S)
        print(f"[POLL] attempt {attempt}/{MAX_POLL_ATTEMPTS} run_id={run_id}", file=sys.stderr, flush=True)
        try:
            resp = httpx.get(url, headers=headers, timeout=30)
            print(f"[POLL] status={resp.status_code}", file=sys.stderr, flush=True)
            if resp.status_code != 200:
                print(f"[POLL] error: {resp.text[:200]}", file=sys.stderr, flush=True)
                continue

            events = resp.json()
            if not isinstance(events, list):
                print(f"[POLL] unexpected response type: {type(events)}", file=sys.stderr, flush=True)
                continue

            print(f"[POLL] got {len(events)} events", file=sys.stderr, flush=True)
            for e in events:
                print(f"[POLL] event: {e.get('event')} id={e.get('id')}", file=sys.stderr, flush=True)

            run_complete = any(e.get("event") in ("run.completed", "done") for e in events)

            message_text = None
            for e in reversed(events):
                if e.get("event") == "message.completed":
                    try:
                        data = e.get("data", {})
                        content = data.get("message", {}).get("content", [])
                        text = content[0].get("text", "") if isinstance(content, list) and content else content if isinstance(content, str) else ""
                        if text and not is_flow_noise(text):
                            message_text = text
                            break
                    except (KeyError, IndexError, TypeError):
                        continue

            if run_complete and message_text:
                print(f"[POLL] SUCCESS: {message_text[:100]!r}", file=sys.stderr, flush=True)
                return message_text

        except Exception as e:
            print(f"[POLL] exception: {e}", file=sys.stderr, flush=True)

    return None


def _single_request(message: str, thread_id: Optional[str]) -> tuple[str, Optional[str], dict]:
    """
    One SSE request. Returns (reply_text, thread_id, all_ids).
    all_ids contains every UUID-like and numeric ID found in the stream.
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
    final_texts  = []
    all_ids      = {}  # collect ALL id-like fields from every chunk

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

            # Log full chunk when it's a run step or contains a new id
            if obj in ("thread.run.step.delta", "thread.run.created", "thread.run.completed", "thread.run.step.completed"):
                print(f"[SSE] {obj} full: {json.dumps(chunk)[:300]}", file=sys.stderr, flush=True)
            else:
                print(f"[SSE] {obj}", file=sys.stderr, flush=True)

            # Collect ALL top-level id fields
            for key in ("id", "run_id", "thread_id", "assistant_id", "step_id"):
                val = chunk.get(key)
                if val:
                    all_ids[key] = val

            # Collect ids from nested data
            data = chunk.get("data", {})
            if isinstance(data, dict):
                for key in ("id", "run_id", "thread_id", "step_id", "assistant_id"):
                    val = data.get(key)
                    if val:
                        all_ids[f"data.{key}"] = val

            if not returned_tid:
                tid = chunk.get("thread_id") or (data.get("thread_id") if isinstance(data, dict) else None)
                if tid:
                    returned_tid = tid
                    print(f"[SSE] thread_id={tid}", file=sys.stderr, flush=True)

            if obj == "thread.message.completed":
                try:
                    text = chunk["data"]["message"]["content"][0]["text"]
                    print(f"[SSE] completed: {text[:100]!r}", file=sys.stderr, flush=True)
                    final_texts.append(text)
                except (KeyError, IndexError):
                    pass
                continue

            try:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    reply_text += delta
            except (KeyError, IndexError):
                continue

    result = final_texts[-1] if final_texts else reply_text.strip()
    print(f"[SSE] all captured IDs: {all_ids}", file=sys.stderr, flush=True)
    return result, returned_tid, all_ids


def _call_agent(message: str, thread_id: Optional[str]) -> tuple[str, Optional[str]]:
    reply, tid, all_ids = _single_request(message, thread_id)

    if not is_flow_noise(reply):
        return reply, tid

    print(f"[FLOW] Flow noise. all_ids={all_ids}", file=sys.stderr, flush=True)

    # Try every UUID-like ID we captured as a potential run_id
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
    candidate_ids = [v for v in all_ids.values() if uuid_pattern.match(str(v)) and v != tid]

    print(f"[FLOW] candidate run_ids: {candidate_ids}", file=sys.stderr, flush=True)

    for candidate in candidate_ids:
        result = _try_run_events(candidate)
        if result:
            return result, tid

    # Fallback — empty message poll
    print(f"[FLOW] No run_id worked, falling back to empty message poll", file=sys.stderr, flush=True)
    for attempt in range(1, 10):
        time.sleep(POLL_DELAY_S)
        poll_reply, tid, _ = _single_request("", tid)
        if poll_reply and not is_flow_noise(poll_reply):
            return poll_reply, tid

    return "Still processing — please wait and try again.", tid


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.4.0"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply, thread_id = _call_agent(req.message, req.thread_id)
    return {"reply": reply, "thread_id": thread_id}