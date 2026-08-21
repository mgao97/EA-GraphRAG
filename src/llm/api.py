"""OpenAI-compatible LLM backend.

Works with **any** server that exposes an OpenAI-compatible ``/v1/chat/completions``
endpoint.  This includes:

* OpenAI hosted models (``https://api.openai.com/v1`` — set ``OPENAI_API_KEY``).
* **Ollama** (default ``http://localhost:11434/v1``, no API key needed).
* **vLLM** (default ``http://localhost:8000/v1``).
* **LM Studio** (default ``http://localhost:1234/v1``).
* Any other OpenAI-compatible drop-in.

Switching from a local model to OpenAI is therefore a configuration change,
not a code change.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

from .base import BaseLLM, LLMMessage, LLMResponse, join_messages


_DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
_DEFAULT_VLLM_URL = "http://localhost:8000/v1"
_DEFAULT_LMSTUDIO_URL = "http://localhost:1234/v1"
_DEFAULT_OPENAI_URL = "https://api.openai.com/v1"


def _resolve_base_url(value: str | None, provider: str | None) -> str | None:
    """Resolve the API base URL from explicit value, provider alias, or env."""
    if value:
        return value.rstrip("/") if not value.endswith("/v1") else value
    env = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
    if env:
        return env
    if provider in {"ollama"}:
        return _DEFAULT_OLLAMA_URL
    if provider in {"vllm"}:
        return _DEFAULT_VLLM_URL
    if provider in {"lmstudio", "lm-studio"}:
        return _DEFAULT_LMSTUDIO_URL
    return None


class APILLM(BaseLLM):
    name = "api"

    def __init__(self, model: str = "gpt-4o-mini",
                  api_key: str | None = None,
                  base_url: str | None = None,
                  provider: str | None = None):
        """Parameters
        ----------
        model:
                Model name.  For Ollama use the model tag you pulled, e.g.
                ``qwen2.5:7b``.  For OpenAI use ``gpt-4o-mini``, etc.
        api_key:
                API key.  If ``None`` we fall back to ``OPENAI_API_KEY`` /
                ``LLM_API_KEY``.  Local services (Ollama / vLLM) ignore the
                key but the OpenAI client still requires a non-empty string,
                so we pass a dummy ``"sk-local"`` when no key is set.
        base_url:
                Full base URL, e.g. ``http://localhost:11434/v1``.  If
                ``None`` we derive one from ``provider`` or the
                ``OPENAI_BASE_URL`` environment variable.
        provider:
                Shortcut: ``"ollama"``, ``"vllm"``, ``"lmstudio"``, or
                ``"openai"`` (default).
        """
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package required for api backend; "
                "pip install openai to enable it"
            ) from exc

        resolved_url = _resolve_base_url(base_url, provider)
        resolved_key = (api_key
                         or os.environ.get("LLM_API_KEY")
                         or os.environ.get("OPENAI_API_KEY"))
        # Local OpenAI-compatible servers (Ollama, vLLM, LM Studio) do not
        # require a real API key.  The OpenAI client still needs a non-empty
        # string, so we substitute a placeholder.
        if not resolved_key and resolved_url and "localhost" in resolved_url:
            resolved_key = "sk-local-no-auth"

        kwargs: dict[str, Any] = {"api_key": resolved_key}
        if resolved_url:
            kwargs["base_url"] = resolved_url

        self.model = model
        self.base_url = resolved_url
        self.client = OpenAI(**kwargs)

    def chat(self, messages: Sequence[LLMMessage], temperature: float = 0.0,
              max_tokens: int = 512) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        msg = resp.choices[0].message.content or ""
        usage: dict[str, Any] = {}
        if resp.usage:
            usage = dict(resp.usage.model_dump())
        return LLMResponse(text=msg.strip(), usage=usage)

    def list_models(self) -> list[str]:
        """Return models known to the server (best-effort)."""
        try:
            models = self.client.models.list()
            return [m.id for m in getattr(models, "data", [])]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------
def ollama_llm(model: str = "qwen2.5:7b") -> APILLM:
    """Convenience constructor for Ollama."""
    return APILLM(model=model, provider="ollama")


def vllm_llm(model: str = "meta-llama/Llama-3.1-8B-Instruct") -> APILLM:
    """Convenience constructor for vLLM."""
    return APILLM(model=model, provider="vllm")


def openai_llm(model: str = "gpt-4o-mini") -> APILLM:
    """Convenience constructor for OpenAI."""
    return APILLM(model=model, provider="openai")
