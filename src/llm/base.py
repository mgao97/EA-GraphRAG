"""LLM interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMResponse:
    text: str
    usage: dict


class BaseLLM(ABC):
    """Minimal chat-style interface."""

    name: str = "base"

    @abstractmethod
    def chat(self, messages: Sequence[LLMMessage], temperature: float = 0.0,
              max_tokens: int = 512) -> LLMResponse: ...


def join_messages(messages: Sequence[LLMMessage]) -> str:
    return "\n".join(f"[{m.role}] {m.content}" for m in messages)
