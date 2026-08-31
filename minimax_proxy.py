#!/usr/bin/env python
"""
Local OpenAI-compatible proxy.

  * /v1/chat/completions  -> MiniMax-M3 /v1/responses (key baked in / MINIMAX_API_KEY)
  * /v1/embeddings        -> bianxie.ai text-embedding-3-small (chat/completions interface)

Embeddings provider note:
    bianxie.ai exposes `text-embedding-3-small` through its *chat/completions*
    interface (not the standard /v1/embeddings endpoint). This proxy accepts a
    standard OpenAI embeddings request

        {"model": "text-embedding-3-small", "input": ["...", "..."]}

    converts it to a chat call against https://api.bianxie.ai/v1/chat/completions,
    and parses the returned `message.content` back into a float vector, then wraps
    it in the standard OpenAI embeddings response

        {"data": [{"object": "embedding", "index": 0, "embedding": [...]}]}

    The exact text->vector parsing is isolated in `_parse_embedding_from_text`
    so it can be tuned once the real bianxie.ai response shape is confirmed.

Baselines that use an OpenAI client (langchain / openai python SDK) can point their
base_url at this proxy (http://127.0.0.1:30001/v1) and keep using the standard
chat.completions / embeddings interfaces.

Run:
    BIANXIE_API_KEY=bx-xxx python minimax_proxy.py
Env (optional):
    MINIMAX_API_KEY  (defaults to the key baked in below)
    MINIMAX_PROXY_PORT (default 30001)
    BIANXIE_API_KEY / EMBEDDINGS_API_KEY (required for /v1/embeddings)
    BIANXIE_BASE_URL (default https://api.bianxie.ai/v1)
    EMBEDDING_MODEL_NAME (default text-embedding-3-small)
"""
import os
import re
import json
import uuid
import asyncio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response

# Shared async HTTP client. Short connect timeout so a dead upstream (e.g. bianxie.ai
# not responding at TCP level) fails fast instead of hanging the event loop for 120s.
HTTP_TIMEOUT = httpx.Timeout(connect=8.0, read=60.0, write=20.0, pool=10.0)
_client = None

def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        # verify=False: 该机器 CA 证书不全，直连 api.minimax.io 会 SSL 校验失败。
        _client = httpx.AsyncClient(timeout=HTTP_TIMEOUT, verify=False)
    return _client

# 组B LLM 现走 OpenRouter 的 minimax/minimax-m3（绕开 MiniMax 官方限速）。
# 透传标准 OpenAI chat/completions 格式，不再转 MiniMax /responses 格式。
MINIMAX_API_KEY = os.environ.get(
    "MINIMAX_API_KEY",
    "sk-cp-hhAGQFUBOvc_LTDoOdLtMmPIxrcRt2TS8yf6T4XWMoYuyHNibtol_XrQa7X0994a8eq-eKw2j6Jo6pimIq0XaYjbjmEibQuzp8xJ4rDJ8_j5zdDbV5w_SjE",
)
# 组B LLM 现走 MiniMax 官网 API（api.minimaxi.com，替换原先的 OpenRouter 透传）。
# 官网同时支持 OpenAI /v1/chat/completions 格式，下游请求直接透传即可。
MINIMAX_URL = "https://api.minimaxi.com/v1/chat/completions"
OPENROUTER_MODEL = "MiniMax-M3"  # 官网模型名
MODEL = "MiniMax-M3"
PORT = int(os.environ.get("MINIMAX_PROXY_PORT", "30001"))

# ---------------------------------------------------------------------------
# Embeddings via bianxie.ai.
# bianxie.ai serves `text-embedding-3-small` through its chat/completions
# interface. We convert the standard OpenAI embeddings request into a chat call
# and parse the returned text back into a float vector.
# The key must be provided through one of these env vars; no default key.
# ---------------------------------------------------------------------------
BIANXIE_API_KEY = os.environ.get(
    "BIANXIE_API_KEY",
    os.environ.get("EMBEDDINGS_API_KEY", ""),
)
BIANXIE_BASE_URL = os.environ.get("BIANXIE_BASE_URL", "https://api.bianxie.ai/v1").rstrip("/")
BIANXIE_CHAT_URL = BIANXIE_BASE_URL + "/chat/completions"
BIANXIE_EMBED_URL = BIANXIE_BASE_URL + "/embeddings"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

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


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _strip_reasoning_from_response(raw: bytes) -> bytes:
    """把 MiniMax-M3 的 <think>...</think> 推理块从 message.content 中剥离。

    MiniMax-M3 是推理模型, 思考过程会和答案一起返回。多数 baseline(LightRAG /
    GraphRAG / EA)直接把整个 content 当作预测答案, 导致 EM 被压到 0——实测
    LightRAG hotpotqa 组B 剥离后 EM 0.000 -> 0.440, GraphRAG 0.000 -> 0.120。
    这里在代理层统一剥离, 并把推理内容保留到标准 reasoning_content 字段,
    需要 CoT 的方法仍可取用。流式响应在 chat_completions 里直接透传, 不处理。
    """
    try:
        data = json.loads(raw)
    except Exception:
        return raw
    changed = False
    for ch in data.get("choices") or []:
        if not isinstance(ch, dict):
            continue
        msg = ch.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, str) or "<think>" not in content:
            continue
        mt = _THINK_RE.search(content)
        if mt and "reasoning_content" not in msg:
            msg["reasoning_content"] = mt.group(1).strip()
        cleaned = _THINK_RE.sub("", content)
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"</?think>", "", cleaned).strip()
        msg["content"] = cleaned
        changed = True
    if not changed:
        return raw
    try:
        return json.dumps(data, ensure_ascii=False).encode()
    except Exception:
        return raw


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    stream = body.get("stream", False)
    # 透传标准 OpenAI 格式到 MiniMax 官网 API，model 替换为官网的 MiniMax-M3。
    fwd = dict(body)
    fwd["model"] = OPENROUTER_MODEL
    # 去掉 OpenRouter 专有字段（官网不接受 reasoning / HTTP-Referer / X-Title）。
    fwd.pop("reasoning", None)

    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }

    client = get_client()
    if stream:
        # 真实 SSE 透传：把 MiniMax 官网的流式响应转发给下游。
        async def gen():
            async with client.stream("POST", MINIMAX_URL, headers=headers, json=fwd, timeout=HTTP_TIMEOUT) as r:
                async for chunk in r.aiter_raw():
                    yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")

    # 非流式：对 MiniMax 官网的 429(速率限制)/5xx 做指数退避重试，
    # 把整体请求速率自然压在 Token Plan 限额内，下游无需感知限速。
    raw = None
    last_status = 200
    for attempt in range(6):
        try:
            async with client.stream("POST", MINIMAX_URL, headers=headers, json=fwd) as r:
                raw = await r.aread()
                last_status = r.status_code
                if r.status_code in (429, 500, 502, 503):
                    wait = min(2 ** attempt * 3 + 2, 60)
                    print(f"[proxy] MiniMax {r.status_code}, retry {attempt+1}/6 after {wait}s", flush=True)
                    await asyncio.sleep(wait)
                    continue
                break
        except (httpx.TransportError, httpx.TimeoutException) as e:
            wait = min(2 ** attempt * 3 + 2, 60)
            print(f"[proxy] MiniMax transport err {e}, retry {attempt+1}/6 after {wait}s", flush=True)
            await asyncio.sleep(wait)
            continue
    if raw is None or last_status >= 400:
        return Response(
            content=(raw or b"{}")[:500],
            status_code=last_status if last_status >= 400 else 502,
            media_type="application/json",
        )
    # MiniMax 官网返回标准 OpenAI chat/completions 格式，下游 openai SDK 直接解析。
    # 先剥离 <think> 推理块，避免下游 baseline 把思考过程当成预测答案。
    raw = _strip_reasoning_from_response(raw)
    return Response(content=raw, media_type="application/json")


def _parse_embedding_from_text(text: str):
    """Extract a float vector from bianxie.ai's chat `message.content`.

    bianxie.ai serves `text-embedding-3-small` through chat/completions and the
    returned content *should* be a vector (often wrapped in a JSON array or with
    surrounding prose). This parser tries, in order:
      1. a JSON array literal (possibly inside a code fence or prose)
      2. a bracketed list of numbers  [...]
      3. a plain comma/whitespace separated list of numbers
    Returns a list[float] or raises ValueError.
    """
    if text is None:
        raise ValueError("empty content")
    raw = text.strip()

    # 1) JSON array (search for the first [...] that parses as a list of numbers)
    candidates = re.findall(r"\[.*?\]", raw, flags=re.DOTALL)
    for c in candidates:
        try:
            arr = json.loads(c)
            if isinstance(arr, list) and arr and all(isinstance(x, (int, float)) for x in arr):
                return [float(x) for x in arr]
        except (json.JSONDecodeError, ValueError):
            continue

    # 2) brace-less code fence ```[...]```
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, flags=re.DOTALL)
    if fence:
        try:
            arr = json.loads(fence.group(1))
            if isinstance(arr, list) and arr and all(isinstance(x, (int, float)) for x in arr):
                return [float(x) for x in arr]
        except (json.JSONDecodeError, ValueError):
            pass

    # 3) bare numbers separated by commas or whitespace
    nums = re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", raw)
    if len(nums) >= 2:
        try:
            return [float(x) for x in nums]
        except ValueError:
            pass

    raise ValueError(f"could not parse embedding vector from bianxie response: {raw[:200]!r}")


@app.post("/v1/embeddings")
async def embeddings(req: Request):
    """OpenAI-compatible embeddings backed by bianxie.ai.

    Accepts the standard OpenAI request body:
        {"model": "text-embedding-3-small", "input": ["...", "..."]}

    Strategy (tried in order):
      1. bianxie.ai's standard /v1/embeddings endpoint (returns real vectors).
         This is preferred when available.
      2. Fallback: send each input as a chat message to /v1/chat/completions
         and parse the returned `message.content` text back into a float vector
         (in case bianxie only exposes the embedding model through chat).

    Returns the standard OpenAI embeddings response.
    """
    if not BIANXIE_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "BIANXIE_API_KEY / EMBEDDINGS_API_KEY env not set"},
        )
    body = await req.json()
    model = body.get("model", EMBEDDING_MODEL)
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]

    headers = {
        "Authorization": f"Bearer {BIANXIE_API_KEY}",
        "Content-Type": "application/json",
    }

    client = get_client()

    # ---- Attempt 1: standard embeddings endpoint ---------------------------
    # bianxie 偶发 "Server disconnected" / 5xx，这里做退避重试，
    # 否则下游(如 EA 的 RETRIEVE)拿不到向量，检索全废、EM 归零。
    embed_err = None
    for attempt in range(5):
        try:
            payload = {"model": model, "input": inputs}
            if body.get("encoding_format"):
                payload["encoding_format"] = body["encoding_format"]
            r = await client.post(BIANXIE_EMBED_URL, headers=headers, json=payload)
            if r.status_code == 200:
                return JSONResponse(r.json())
            embed_err = r.text[:300]
            if r.status_code in (429, 500, 502, 503):
                wait = min(2 ** attempt * 2 + 1, 30)
                print(f"[proxy] embeddings {r.status_code}, retry {attempt+1}/5 after {wait}s", flush=True)
                await asyncio.sleep(wait)
                continue
            break
        except (httpx.RequestError, httpx.TimeoutException) as e:
            embed_err = f"standard endpoint unreachable: {str(e)[:200]}"
            wait = min(2 ** attempt * 2 + 1, 30)
            print(f"[proxy] embeddings transport err, retry {attempt+1}/5 after {wait}s", flush=True)
            await asyncio.sleep(wait)
            continue

    # ---- Attempt 2: chat/completions fallback -----------------------------
    data = []
    try:
        for idx, text in enumerate(inputs):
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": str(text)}],
            }
            r = await client.post(BIANXIE_CHAT_URL, headers=headers, json=payload)
            r.raise_for_status()
            resp_json = r.json()
            content = resp_json["choices"][0]["message"]["content"]
            vec = _parse_embedding_from_text(content)
            data.append({"object": "embedding", "index": idx, "embedding": vec})
    except httpx.HTTPStatusError as e:
        return JSONResponse(
            status_code=r.status_code,
            content={"error": f"chat fallback failed (standard endpoint err: {embed_err}); {r.text[:300]}"},
        )
    except (httpx.RequestError, httpx.TimeoutException) as e:
        return JSONResponse(status_code=502, content={"error": str(e)[:300]})
    except (KeyError, IndexError, ValueError) as e:
        return JSONResponse(
            status_code=502,
            content={"error": f"chat fallback parse failed: {str(e)[:200]}; standard err: {embed_err}"},
        )

    return JSONResponse({
        "object": "list",
        "data": data,
        "model": model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    })


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
