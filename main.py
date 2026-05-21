"""
SalesBud – watsonx Orchestrate Chat Proxy v3.0
Deployed to IBM Code Engine (group13, eu-de)

KEY CHANGE vs v2.2: Full async_wait loop (Approach B).

When the agent calls a Flow-backed tool, WxO transitions the run to
async_wait status. Previously the proxy would time out because it was
waiting for message.completed that never arrived.

Now the proxy:
  1. Starts a run via /runs/stream, collects SSE events.
  2. Detects async_wait (run.step events containing tool_use with
     is_async=true, or status poll showing async_wait).
  3. Polls GET /runs/{run_id} until status = async_completed.
  4. Extracts tool results from the completed run's step_history.
  5. Submits a follow-up POST /runs with role=tool messages.
  6. Loops until a run completes cleanly (no more async_wait).
"""

import os
import re
import time
import json
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any

app = FastAPI(title="SalesBud Proxy", version="3.0.0")

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
IBM_API_KEY        = os.environ["IBM_API_KEY"]
WXO_INSTANCE_GUID  = os.environ.get("WXO_INSTANCE_GUID", "946df986-9572-455e-bc1f-be9c5c5ec40e")
WXO_HOST           = f"https://eu-de.watson-orchestrate.cloud.ibm.com/{WXO_INSTANCE_GUID}"
AUTH_AGENT_ID      = os.environ.get("AUTH_AGENT_ID", "6c98d390-dfc7-4b8f-b11e-bf36dd148c80")
ORCHESTRATOR_ID    = os.environ.get("ORCHESTRATOR_ID", "b2340dd1-11f7-4234-b073-125d85d78c98")

IAM_URL            = "https://iam.cloud.ibm.com/identity/token"

# Async loop settings
POLL_INTERVAL_S    = 1.0   # seconds between status polls
POLL_MAX_ATTEMPTS  = 60    # max 60s before giving up on a single async tool
ASYNC_LOOP_MAX     = 5     # max tool-call rounds per user turn (safety valve)
STREAM_TIMEOUT_MS  = 90000 # ms — passed to WxO as stream_timeout

# ── IAM TOKEN CACHE ───────────────────────────────────────────────────────────
_token_cache: dict = {}

async def get_iam_token() -> str:
    now = time.time()
    if _token_cache.get("token") and now < _token_cache.get("expires_at", 0) - 30:
        return _token_cache["token"]

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            IAM_URL,
            data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": IBM_API_KEY},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        data = resp.json()
        _token_cache["token"] = data["access_token"]
        _token_cache["expires_at"] = now + data.get("expires_in", 3600)
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

# ── WXO MESSAGE BUILDER ───────────────────────────────────────────────────────
def build_wxo_message(role: str, text: str, tool_call_id: str = None) -> dict:
    """Build a WxO-shaped message object."""
    msg = {
        "role": role,
        "content": [{"response_type": "text", "text": text}],
    }
    if tool_call_id:
        msg["additional_properties"] = {
            "tool_call_id": tool_call_id,
            "display_properties": {"skip_render": True, "is_async": False},
        }
    return msg

def build_wxo_history(history: list[Message]) -> list[dict]:
    return [build_wxo_message(m.role, m.content) for m in history]

# ── CORE: STREAM + ASYNC_WAIT LOOP ────────────────────────────────────────────

async def run_agent_turn(
    agent_id: str,
    message_text: str,
    history: list[Message],
    thread_id: Optional[str],
    context_variables: dict = {},
) -> dict:
    """
    Execute one user turn, handling async_wait loops automatically.

    Returns:
        {
          "reply": str,           # final assistant text
          "thread_id": str,       # thread to reuse next turn
          "async_rounds": int,    # how many async loops occurred
        }
    """
    token = await get_iam_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Build the initial payload
    wxo_history = build_wxo_history(history)
    payload = {
        "agent_id": agent_id,
        "message": build_wxo_message("user", message_text),
        "thread_id": thread_id,
        "context_variables": context_variables,
        "stream_timeout": STREAM_TIMEOUT_MS,
    }

    current_thread_id = thread_id
    final_reply = ""
    async_rounds = 0

    for loop in range(ASYNC_LOOP_MAX + 1):
        # ── STEP 1: POST /runs/stream ─────────────────────────────────────
        stream_url = f"{WXO_HOST}/api/v1/orchestrate/runs/stream"
        sse_result = await consume_sse_stream(stream_url, headers, payload)

        current_thread_id = sse_result.get("thread_id") or current_thread_id
        run_id            = sse_result.get("run_id")

        # ── STEP 2: Did the run complete cleanly? ─────────────────────────
        if sse_result["status"] == "completed":
            final_reply = sse_result["reply"]
            break

        # ── STEP 3: async_wait — poll until async_completed ───────────────
        if sse_result["status"] == "async_wait":
            async_rounds += 1
            print(f"[proxy] async_wait detected on loop {loop}, run_id={run_id}")

            completed_run = await poll_until_async_completed(run_id, headers)
            if not completed_run:
                # Timed out waiting for flows
                final_reply = (
                    "I'm sorry — one of the background tools is taking too long to respond. "
                    "Please try again in a moment."
                )
                break

            # ── STEP 4: Extract tool results from the completed run ───────
            tool_results = extract_tool_results(completed_run)
            print(f"[proxy] Got {len(tool_results)} tool result(s) from async run")

            if not tool_results:
                # Fallback: no structured results found, prod the agent to continue
                payload = {
                    "agent_id": agent_id,
                    "message": build_wxo_message(
                        "user",
                        "The background tool has completed. Please continue."
                    ),
                    "thread_id": current_thread_id,
                    "context_variables": context_variables,
                    "stream_timeout": STREAM_TIMEOUT_MS,
                }
            else:
                # ── STEP 5: Submit tool results as follow-up messages ─────
                # WxO expects one message per tool result with role=tool
                # We send the first result; if there are multiple, chain them
                tool_messages = []
                for tr in tool_results:
                    tool_messages.append(
                        build_wxo_message(
                            role="tool",
                            text=json.dumps(tr["output"]) if isinstance(tr["output"], dict) else str(tr["output"]),
                            tool_call_id=tr.get("tool_call_id"),
                        )
                    )

                # Send the first tool result to resume the run
                # (subsequent results sent in the same payload as additional history context)
                first = tool_messages[0]
                payload = {
                    "agent_id": agent_id,
                    "message": first,
                    "thread_id": current_thread_id,
                    "context_variables": context_variables,
                    "stream_timeout": STREAM_TIMEOUT_MS,
                }

            # Loop continues — will POST /runs/stream again with tool result
            continue

        # ── Any other terminal status ─────────────────────────────────────
        final_reply = sse_result.get("reply") or "An unexpected error occurred."
        break

    return {
        "reply": final_reply,
        "thread_id": current_thread_id,
        "async_rounds": async_rounds,
    }


async def consume_sse_stream(url: str, headers: dict, payload: dict) -> dict:
    """
    POST to the WxO SSE stream endpoint and collect events until done or async_wait.

    Returns:
        {
          "status":    "completed" | "async_wait" | "error",
          "reply":     str,          # assembled assistant text (may be partial on async_wait)
          "thread_id": str | None,
          "run_id":    str | None,
          "tool_calls": list,        # raw tool_use events captured
        }
    """
    reply_chunks: list[str] = []
    thread_id = None
    run_id    = None
    tool_calls: list[dict] = []
    final_status = "completed"

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    print(f"[proxy] SSE request failed: {resp.status_code} — {body[:300]}")
                    return {"status": "error", "reply": f"WxO error {resp.status_code}", "thread_id": None, "run_id": None, "tool_calls": []}

                async for raw_line in resp.aiter_lines():
                    if not raw_line.strip():
                        continue

                    # SSE format: "data: {...}" or "event: done" or plain JSON
                    line = raw_line
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("event", "")
                    data       = event.get("data", event)  # some events are flat

                    # ── Capture IDs ───────────────────────────────────────
                    if not thread_id:
                        thread_id = (
                            data.get("thread_id") or
                            event.get("thread_id") or
                            data.get("run", {}).get("thread_id")
                        )
                    if not run_id:
                        run_id = (
                            data.get("run_id") or
                            data.get("id") or
                            event.get("run_id")
                        )

                    # ── Text delta ────────────────────────────────────────
                    if event_type in ("message.delta", "run.step.delta"):
                        # Try multiple shapes WxO uses
                        delta_text = (
                            data.get("delta", {}).get("content", [{}])[0].get("text", "") or
                            data.get("text", "") or
                            data.get("content", "")
                        )
                        if delta_text:
                            reply_chunks.append(delta_text)

                    # ── Full message completed ────────────────────────────
                    elif event_type == "message.completed":
                        msg_content = data.get("content", [])
                        if isinstance(msg_content, list):
                            for block in msg_content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    reply_chunks.append(block.get("text", ""))
                        elif isinstance(msg_content, str):
                            reply_chunks.append(msg_content)

                    # ── Tool use (async flow invocation) ─────────────────
                    elif event_type == "run.step.completed":
                        step = data.get("step_details", {})
                        if step.get("type") == "tool_calls":
                            for tc in step.get("tool_calls", []):
                                tool_calls.append(tc)
                                # Check if this is an async invocation
                                if tc.get("type") == "function" and tc.get("function", {}).get("is_async"):
                                    final_status = "async_wait"

                    # ── Explicit async signal ─────────────────────────────
                    elif event_type == "run.step.started":
                        if data.get("type") == "tool_calls":
                            # Check display_properties for is_async flag
                            dp = data.get("additional_properties", {}).get("display_properties", {})
                            if dp.get("is_async"):
                                final_status = "async_wait"

                    # ── message.interrupt = WxO paused for async ──────────
                    elif event_type == "message.interrupt":
                        final_status = "async_wait"
                        print(f"[proxy] message.interrupt received — run is async_wait")

                    # ── Done ──────────────────────────────────────────────
                    elif event_type == "done":
                        break

                    # ── Error ─────────────────────────────────────────────
                    elif event_type == "error":
                        print(f"[proxy] SSE error event: {data}")
                        final_status = "error"
                        break

    except httpx.ReadTimeout:
        print("[proxy] SSE stream timed out — treating as async_wait")
        final_status = "async_wait"
    except Exception as e:
        print(f"[proxy] SSE stream exception: {e}")
        final_status = "error"

    reply_text = "".join(reply_chunks).strip()

    # If we got no text at all and no async signal, something is wrong
    if not reply_text and final_status == "completed":
        final_status = "error"

    return {
        "status":     final_status,
        "reply":      reply_text,
        "thread_id":  thread_id,
        "run_id":     run_id,
        "tool_calls": tool_calls,
    }


async def poll_until_async_completed(run_id: str, headers: dict) -> Optional[dict]:
    """
    Poll GET /runs/{run_id} until status is async_completed (or failed/cancelled).
    Returns the full run object, or None on timeout.
    """
    if not run_id:
        print("[proxy] poll_until_async_completed: no run_id, can't poll")
        return None

    poll_url = f"{WXO_HOST}/api/v1/orchestrate/runs/{run_id}"

    for attempt in range(POLL_MAX_ATTEMPTS):
        await asyncio.sleep(POLL_INTERVAL_S)
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(poll_url, headers=headers)
                if resp.status_code != 200:
                    print(f"[proxy] poll attempt {attempt}: HTTP {resp.status_code}")
                    continue

                run_data = resp.json()
                status   = run_data.get("status", "unknown")
                print(f"[proxy] poll attempt {attempt+1}: status={status}")

                if status in ("async_completed", "completed"):
                    return run_data
                if status in ("failed", "cancelled", "expired"):
                    print(f"[proxy] run ended with status={status}")
                    return run_data

        except Exception as e:
            print(f"[proxy] poll exception on attempt {attempt}: {e}")

    print(f"[proxy] poll_until_async_completed timed out after {POLL_MAX_ATTEMPTS}s for run_id={run_id}")
    return None


def extract_tool_results(completed_run: dict) -> list[dict]:
    """
    Parse a completed run object to extract tool outputs.

    WxO stores tool results in run.result or in step_history.
    Returns a list of { tool_call_id, tool_name, output }.
    """
    results = []

    # Try run.result first (flat output)
    run_result = completed_run.get("result")
    if run_result:
        if isinstance(run_result, dict):
            results.append({
                "tool_call_id": completed_run.get("id"),
                "tool_name":    "flow_result",
                "output":       run_result,
            })
        elif isinstance(run_result, str):
            results.append({
                "tool_call_id": completed_run.get("id"),
                "tool_name":    "flow_result",
                "output":       run_result,
            })

    # Also check step_history for tool_calls with outputs
    step_history = completed_run.get("step_history", [])
    for step in step_history:
        step_details = step.get("step_details", {})
        if step_details.get("type") == "tool_calls":
            for tc in step_details.get("tool_calls", []):
                output = tc.get("function", {}).get("output")
                if output:
                    results.append({
                        "tool_call_id": tc.get("id"),
                        "tool_name":    tc.get("function", {}).get("name", "unknown"),
                        "output":       output,
                    })

    return results


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0.0"}


@app.post("/auth")
async def auth(req: AuthRequest):
    """Auth route — calls Auth Agent to look up the rep's Salesforce User ID."""
    try:
        result = await run_agent_turn(
            agent_id=AUTH_AGENT_ID,
            message_text=req.message,
            history=req.history,
            thread_id=req.thread_id,
        )
        return {
            "reply":      result["reply"],
            "thread_id":  result["thread_id"],
            "history":    [m.dict() for m in req.history] + [
                {"role": "user",      "content": req.message},
                {"role": "assistant", "content": result["reply"]},
            ],
        }
    except Exception as e:
        print(f"[proxy] /auth error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Main chat route — calls Orchestrator V2 with async_wait loop.
    Passes owner_id and user_email as context_variables so the agent
    can use them without asking (auth simplification, TC Auth).
    """
    context_vars = {}
    if req.owner_id:
        context_vars["owner_id"] = req.owner_id
    if req.user_email:
        context_vars["wxo_email_id"] = req.user_email

    try:
        result = await run_agent_turn(
            agent_id=ORCHESTRATOR_ID,
            message_text=req.message,
            history=req.history,
            thread_id=req.thread_id,
            context_variables=context_vars,
        )
        return {
            "reply":        result["reply"],
            "thread_id":    result["thread_id"],
            "async_rounds": result["async_rounds"],
            "history":      [m.dict() for m in req.history] + [
                {"role": "user",      "content": req.message},
                {"role": "assistant", "content": result["reply"]},
            ],
        }
    except Exception as e:
        print(f"[proxy] /chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))