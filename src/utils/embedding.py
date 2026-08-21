"""Embedding utilities with a deterministic offline backend.

The default :class:`DummyEmbedder` produces reproducible hashed bag-of-ngrams
vectors so the entire pipeline runs without any model download.  Real
backends (sentence-transformer, OpenAI-compatible) can be plugged in by
implementing :class:`BaseEmbedder` or by switching ``backend`` in
``configs/default.yaml``.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod
from typing import Iterable, List, Sequence

import numpy as np


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _hash_features(tokens: Sequence[str], dim: int) -> np.ndarray:
    """Hashed n-gram features with sub-linear TF scaling."""
    vec = np.zeros(dim, dtype=np.float32)
    if not tokens:
        return vec
    grams: List[str] = []
    grams.extend(tokens)
    if len(tokens) > 1:
        grams.extend(f"{tokens[i-1]}_{tokens[i]}" for i in range(1, len(tokens)))
    for g in grams:
        h = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if (h[4] & 1) else -1.0
        vec[idx] += sign
    vec = np.sign(vec) * np.sqrt(np.abs(vec))
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec


class BaseEmbedder(ABC):
    """Interface for pluggable embedding backends."""

    @abstractmethod
    def embed(self, texts: Iterable[str]) -> np.ndarray: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...


class DummyEmbedder(BaseEmbedder):
    """Deterministic, dependency-free embedder used for offline experiments."""

    def __init__(self, dim: int = 256, seed: int = 0):
        self._dim = int(dim)
        self._seed = int(seed)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        out = np.zeros((0, self._dim), dtype=np.float32)
        texts = list(texts)
        if not texts:
            return out
        out = np.stack([_hash_features(_tokenize(t), self._dim) for t in texts])
        return out

    def cosine(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.size == 0 or b.size == 0:
            return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
        return a @ b.T


def get_embedder(backend: str = "dummy",
                  model: str | None = None,
                  dim: int = 256,
                  base_url: str | None = None,
                  api_key: str | None = None,
                  api_key_env: str = "OPENAI_API_KEY",
                  **kwargs) -> BaseEmbedder:
    """Factory that returns an embedder based on ``backend`` name.

    ``base_url`` and ``api_key`` are forwarded to the OpenAI-compatible
    backend so the same factory works for hosted OpenAI, Ollama, vLLM, etc.
    """
    backend = (backend or "dummy").lower()
    if backend == "dummy":
        return DummyEmbedder(dim=dim, seed=kwargs.get("seed", 0))
    if backend in {"sentence_transformer", "st", "bge"}:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed; pip install "
                "sentence-transformers to use this backend"
            ) from exc
        model_name = model or "BAAI/bge-m3"

        class _STEmbedder(BaseEmbedder):
            def __init__(self, name: str):
                self._m = SentenceTransformer(name)

            @property
            def dim(self) -> int:
                return int(self._m.get_sentence_embedding_dimension())

            def embed(self, texts: Iterable[str]) -> np.ndarray:
                vecs = self._m.encode(list(texts), normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False)
                return vecs.astype(np.float32)

        return _STEmbedder(model_name)
    if backend in {"openai", "api", "ollama", "vllm"}:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package required for openai-compatible embedding "
                "backend; pip install openai to enable it"
            ) from exc
        resolved_key = api_key or os.environ.get(api_key_env)
        if not resolved_key and base_url and "localhost" in base_url:
            resolved_key = "sk-local-no-auth"
        client_kwargs = {"api_key": resolved_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        is_local = bool(base_url and "localhost" in base_url)
        model_name = (
            model
            or ("text-embedding-3-small" if not is_local else "nomic-embed-text")
        )

        class _OpenAIEmbedder(BaseEmbedder):
            def __init__(self, name: str):
                self._m = name
                self._dim_cache: int | None = None

            @property
            def dim(self) -> int:
                return int(self._dim_cache or 1536)

            def embed(self, texts: Iterable[str]) -> np.ndarray:
                texts = list(texts)
                if not texts:
                    out_dim = int(self._dim_cache or 1536)
                    return np.zeros((0, out_dim), dtype=np.float32)
                resp = client.embeddings.create(model=self._m, input=texts)
                vecs = np.stack(
                    [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]
                )
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vecs = vecs / norms
                if self._dim_cache is None:
                    self._dim_cache = int(vecs.shape[1])
                return vecs.astype(np.float32)

        return _OpenAIEmbedder(model_name)
    raise ValueError(f"Unknown embedding backend: {backend}")
