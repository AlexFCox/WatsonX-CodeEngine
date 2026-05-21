"""
SalesBud – watsonx Orchestrate Chat Proxy v2.5
Tries multiple URL patterns for the run events endpoint.
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

app = FastAPI(title="SalesBud Proxy", version="2.5.0")

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


def _try_all_event_urls(run_id: str) -> Optional[str]:
    """
    Try every known URL pattern for the run events endpoint.
    Log status for each so we can see which one works.
    """
    token = get_iam_token()
    headers = {"Authorization": f"Bearer {token}"}

    # All URL patterns to try
    candidate_urls = [
        f"{WXO_BASE_URL}/api/v1/orchestrate/runs/{run_id}/events",
        f"{WXO_BASE_URL}/v1/orchestrate/runs/{run_id}/events",
        f"{WXO_BASE_URL}/api/v1/orchestrate/{WXO_AGENT_ID}/runs/{run_id}/events",
        f"{WXO_BASE_URL}/v1/orchestrate/{WXO_AGENT_ID}/runs/{run_id}/events",
        # Also try run status without /events
        f"{WXO_BASE_URL}/api/v1/orchestrate/runs/{run_id}",
        f"{WXO_BASE_URL}/v1/orchestrate/runs/{run_id}",
    ]

    # First probe all URLs once to find which one returns non-404
    print(f"[PROBE] Probing all URL patterns for run_id={run_id}", file=sys.stderr, flush=True)
    working_url = None
    for url in candidate_urls:
        try:
            resp = httpx.get(url, headers=headers, timeout=10)
            print(f"[PROBE] {resp.status_code} → {url}", file=sys.stderr, flush=True)
            if resp.status_code == 200:
                working_url = url
                break
        except Exception as e:
            print(f"[PROBE] error {url}: {e}", file=sys.stderr, flush=True)

    if not working_url:
        print(f"[PROBE] No working URL found", file=sys.stderr, flush=True)
        return None

    print(f"[POLL] Using URL: {working_url}", file=sys.stderr, flush=True)

    # Poll working URL until run.completed
    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        time.sleep(POLL_DELAY_S)
        print(f"[POLL] attempt {attempt}/{MAX_POLL_ATTEMPTS}", file=sys.stderr, flush=True)
        try:
            resp = httpx.get(working_url, headers=headers, timeout=30)
            if resp.status_code != 200:
                continue

            data = resp.json()

            # Handle both list (events) and object (run status) responses
            if isinstance(data, list):
                events = data
                run_complete = any(e.get("event") in ("run.completed", "done") for e in events)
                print(f"[POLL] {len(events)} events, complete={run_complete}", file=sys.stderr, flush=True)
                for e in events:
                    print(f"[POLL] event={e.get('event')}", file=sys.stderr, flush=True)

                message_text = None
                for e in reversed(events):
                    if e.get("event") == "message.completed":
                        try:
                            content = e.get("data", {}).get("message", {}).get("content", [])
                            text = content[0].get("text", "") if isinstance(content, list) and content else ""
                            if text and not is_flow_noise(text):
                                message_text = text
                                break
                        except (KeyError, IndexError, TypeError):
                            continue

                if run_complete and message_text:
                    return message_text

            elif isinstance(data, dict):
                status = data.get("status", "")
                print(f"[POLL] run status={status}", file=sys.stderr, flush=True)
                if status in ("completed", "async_completed"):
                    result = data.get("result", {})
                    print(f"[POLL] result={json.dumps(result)[:200]}", file=sys.stderr, flush=True)
                    # Try to extract message from result
                    text = result.get("output") or result.get("text") or result.get("message") or ""
                    if text and not is_flow_noise(str(text)):
                        return str(text)
                    # If no text in result, fall through to empty message poll
                    return None

        except Exception as e:
            print(f"[POLL] exception: {e}", file=sys.stderr, flush=True)

    return None


def _single_request(message: str, thread_id: Optional[str]) -> tuple[str, Optional[str], dict]:
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
    all_ids      = {}

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

            # Collect all IDs
            for key in ("id", "run_id", "thread_id", "assistant_id", "step_id"):
                val = chunk.get(key)
                if val:
                    all_ids[key] = val

            data = chunk.get("data", {})
            if isinstance(data, dict):
                for key in ("id", "run_id", "thread_id", "step_id"):
                    val = data.get(key)
                    if val:
                        all_ids[f"data.{key}"] = val

            if not returned_tid:
                tid = chunk.get("thread_id") or (data.get("thread_id") if isinstance(data, dict) else None)
                if tid:
                    returned_tid = tid

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
    print(f"[SSE] captured IDs: {all_ids}", file=sys.stderr, flush=True)
    return result, returned_tid, all_ids


def _call_agent(message: str, thread_id: Optional[str]) -> tuple[str, Optional[str]]:
    reply, tid, all_ids = _single_request(message, thread_id)

    if not is_flow_noise(reply):
        return reply, tid

    print(f"[FLOW] Noise detected. all_ids={all_ids}", file=sys.stderr, flush=True)

    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
    candidate_ids = [v for v in all_ids.values() if uuid_pattern.match(str(v)) and v != tid]
    print(f"[FLOW] candidate run_ids: {candidate_ids}", file=sys.stderr, flush=True)

    # Try events/status endpoint for each candidate
    for candidate in candidate_ids:
        result = _try_all_event_urls(candidate)
        if result:
            return result, tid

    # Fallback — empty message poll on same thread
    print(f"[FLOW] Falling back to empty message poll", file=sys.stderr, flush=True)
    for attempt in range(1, 10):
        time.sleep(POLL_DELAY_S)
        poll_reply, tid, _ = _single_request("", tid)
        if poll_reply and not is_flow_noise(poll_reply):
            return poll_reply, tid

    return "Still processing — please wait and try again.", tid


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.5.0"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply, thread_id = _call_agent(req.message, req.thread_id)
    return {"reply": reply, "thread_id": thread_id}


# ── Run status route (called by frontend poller) ──────────────────────────────
class RunStatusRequest(BaseModel):
    run_id: str
    thread_id: Optional[str] = None


class RunStatusResponse(BaseModel):
    status: str           # "pending", "running", "completed", "failed", "expired", "unknown"
    reply: Optional[str] = None
    thread_id: Optional[str] = None


@app.post("/run-status", response_model=RunStatusResponse)
def run_status(req: RunStatusRequest):
    """
    Check status of a WXO run and extract reply if completed.
    Called by frontend during flow noise polling.
    Tries multiple URL patterns and logs all attempts.
    """
    token = get_iam_token()
    headers = {"Authorization": f"Bearer {token}"}

    # URL patterns to try
    urls = [
        f"{WXO_BASE_URL}/api/v1/orchestrate/runs/{req.run_id}",
        f"{WXO_BASE_URL}/v1/orchestrate/runs/{req.run_id}",
        f"{WXO_BASE_URL}/api/v1/orchestrate/runs/{req.run_id}/events",
        f"{WXO_BASE_URL}/v1/orchestrate/runs/{req.run_id}/events",
    ]

    for url in urls:
        try:
            resp = httpx.get(url, headers=headers, timeout=15)
            print(f"[RUN-STATUS] {resp.status_code} {url}", file=sys.stderr, flush=True)

            if resp.status_code != 200:
                continue

            data = resp.json()
            print(f"[RUN-STATUS] response: {json.dumps(data)[:300]}", file=sys.stderr, flush=True)

            # Handle list response (events endpoint)
            if isinstance(data, list):
                run_complete = any(e.get("event") in ("run.completed", "done") for e in data)
                message_text = None
                for e in reversed(data):
                    if e.get("event") == "message.completed":
                        try:
                            content = e.get("data", {}).get("message", {}).get("content", [])
                            text = content[0].get("text", "") if isinstance(content, list) and content else ""
                            if text and not is_flow_noise(text):
                                message_text = text
                                break
                        except (KeyError, IndexError, TypeError):
                            continue

                if run_complete:
                    return {"status": "completed", "reply": message_text, "thread_id": req.thread_id}
                return {"status": "running", "thread_id": req.thread_id}

            # Handle object response (run status endpoint)
            elif isinstance(data, dict):
                status = data.get("status", "unknown")
                if status in ("completed", "async_completed"):
                    result = data.get("result", {})
                    text = result.get("output") or result.get("text") or result.get("message") or ""
                    return {"status": "completed", "reply": str(text) if text else None, "thread_id": req.thread_id}
                if status in ("failed", "cancelled", "expired"):
                    return {"status": status, "thread_id": req.thread_id}
                return {"status": status or "running", "thread_id": req.thread_id}

        except Exception as e:
            print(f"[RUN-STATUS] error {url}: {e}", file=sys.stderr, flush=True)

    return {"status": "unknown", "thread_id": req.thread_id}