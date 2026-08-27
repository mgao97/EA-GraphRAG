#!/usr/bin/env python
"""
Local OpenAI-compatible proxy that forwards chat/completions requests to the
MiniMax-M3 /v1/responses API.

Baselines that use an OpenAI client (langchain / openai python SDK) can point
their base_url at this proxy (http://127.0.0.1:30001/v1) and keep using the
standard chat.completions interface; this proxy translates to/from MiniMax's
/v1/responses protocol.

Run:
    python minimax_proxy.py
Env (optional):
    MINIMAX_API_KEY  (defaults to the key baked in below)
    MINIMAX_PROXY_PORT (default 30001)
"""
import os
import json
import uuid
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MINIMAX_API_KEY = os.environ.get(
    "MINIMAX_API_KEY",
    "sk-cp-1kLPw64SDteGbS2G-dS6JQzCjEUJPgvB_WAy8uN20_rnUYod170Aw71mTpbcIaWCvdSmwf3rttuKVhkOuwszkSCXF9Swy2BqsnKoI21jh5PF3WUcQgTsX5s",
)
MINIMAX_URL = "https://api.minimaxi.com/v1/responses"
MODEL = "MiniMax-M3"
PORT = int(os.environ.get("MINIMAX_PROXY_PORT", "30001"))

app = FastAPI()


def _to_minimax_payload(body: dict) -> dict:
    """Convert an OpenAI chat/completions request into a MiniMax /responses request."""
    messages = body.get("messages", [])
    # Build a single text 'input' from the conversation.
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        parts.append(f"{role}: {content}")
    inp = "\n".join(parts)
    payload = {
        "model": MODEL,
        "input": inp,
    }
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("max_tokens") is not None:
        payload["max_tokens"] = body["max_tokens"]
    # request id so we can poll for the result
    payload["request_id"] = "ea_" + uuid.uuid4().hex
    return payload, inp


def _extract_text(resp_json: dict) -> str:
    """Pull the answer text out of a MiniMax /responses response."""
    # /responses returns output items with .content[].text
    out = resp_json.get("output", [])
    texts = []
    for item in out:
        for c in item.get("content", []) if isinstance(item, dict) else []:
            if isinstance(c, dict) and c.get("type") == "output_text":
                texts.append(c.get("text", ""))
            elif isinstance(c, dict) and "text" in c:
                texts.append(c.get("text", ""))
    if texts:
        return "\n".join(texts)
    # fallbacks
    if isinstance(out, list) and out and isinstance(out[0], dict) and "text" in out[0]:
        return out[0]["text"]
    return resp_json.get("text", "")


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    stream = body.get("stream", False)
    payload, _ = _to_minimax_payload(body)

    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }

    if stream:
        # Minimal streaming: do a blocking call then stream the final text.
        r = requests.post(MINIMAX_URL, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        text = _extract_text(r.json())
        async def gen():
            chunk = {
                "id": "chatcmpl-minimax",
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": text}, "finish_reason": "stop"}],
            }
            yield "data: " + json.dumps(chunk) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    r = requests.post(MINIMAX_URL, headers=headers, json=payload, timeout=600)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        return JSONResponse(
            status_code=r.status_code,
            content={"error": r.text[:500]},
        )
    text = _extract_text(r.json())
    resp = {
        "id": "chatcmpl-minimax",
        "object": "chat.completion",
        "created": 0,
        "model": body.get("model", MODEL),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return JSONResponse(resp)


@app.get("/v1/models")
async def models():
    return JSONResponse({"object": "list", "data": [{"id": MODEL, "object": "model"}]})


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    # NOTE: uvicorn requires an import string for workers/reload; use a simple
    # single-worker run with a raised concurrency limit. Synchronous endpoints
    # are served from uvicorn's threadpool, so parallel baseline calls are fine.
    uvicorn.run(
        "minimax_proxy:app",
        host="0.0.0.0",
        port=PORT,
        limit_concurrency=128,
    )
