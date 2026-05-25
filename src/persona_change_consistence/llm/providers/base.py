"""Provider abstraction.

A *provider* knows how to make a single chat completion call given a key
and a model id. The scheduler owns retries, rate limits, and key rotation;
the provider only owns request shape and response parsing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ProviderError",
    "RetryableProviderError",
    "RateLimitError",
    "Provider",
]


@dataclass
class ChatRequest:
    """A unified chat request."""

    model: str  # provider-side model id (post-routing)
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    response_format: Optional[Dict[str, Any]] = None  # e.g. {"type": "json_object"}
    # Controls extended-thinking / reasoning for models that support it.
    # ``False`` (the experimental default) tells the provider to skip the
    # internal chain-of-thought so the response is a plain answer.
    # ``True`` enables it (Claude / Gemini), and ``None`` leaves the provider
    # default in place. Each provider knows how to translate this value.
    thinking: Optional[bool] = False
    # Optional explicit thinking-budget in output tokens for providers
    # (Anthropic, Gemini) that accept one. Ignored when ``thinking is False``.
    thinking_budget: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None


@dataclass
class ChatResponse:
    """A unified chat response."""

    content: str
    model: str
    finish_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    raw: Any = None  # provider-native object for advanced inspection


class ProviderError(Exception):
    """Non-retryable provider error (bad request, auth, etc.)."""


class RetryableProviderError(ProviderError):
    """Transient error worth retrying after backoff."""


class RateLimitError(RetryableProviderError):
    """Provider rate limit hit. Optionally carries a retry-after hint."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class Provider(ABC):
    """Stateless provider — one instance per provider config block."""

    name: str

    def __init__(self, *, name: str, base_url: Optional[str] = None):
        self.name = name
        self.base_url = base_url

    @abstractmethod
    def chat(self, request: ChatRequest, *, api_key: str) -> ChatResponse:
        """Run a single chat completion. Must translate provider errors to
        ProviderError / RetryableProviderError / RateLimitError."""
        raise NotImplementedError
