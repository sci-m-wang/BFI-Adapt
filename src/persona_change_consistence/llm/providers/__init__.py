"""Provider registry."""

from __future__ import annotations

from typing import Optional

from .base import (
    ChatRequest,
    ChatResponse,
    Provider,
    ProviderError,
    RateLimitError,
    RetryableProviderError,
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "Provider",
    "ProviderError",
    "RetryableProviderError",
    "RateLimitError",
    "build_provider",
]


def build_provider(*, name: str, type: str, base_url: Optional[str] = None) -> Provider:
    """Instantiate a provider from its type string."""
    if type == "openai_compat":
        from .openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(name=name, base_url=base_url)
    if type == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider(name=name, base_url=base_url)
    if type == "google":
        from .google import GoogleProvider
        return GoogleProvider(name=name, base_url=base_url)
    raise ValueError(f"Unknown provider type: {type!r}")
