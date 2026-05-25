"""Anthropic native provider.

Uses the official ``anthropic`` Python SDK. Anthropic's API differs from
OpenAI's: system prompts are passed as a top-level ``system`` parameter
(not as a message role), and ``max_tokens`` is required.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import (
    ChatRequest,
    ChatResponse,
    Provider,
    ProviderError,
    RateLimitError,
    RetryableProviderError,
)

log = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(Provider):
    def chat(self, request: ChatRequest, *, api_key: str) -> ChatResponse:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "anthropic SDK is required. Install with: pip install anthropic"
            ) from e

        client = anthropic.Anthropic(
            api_key=api_key,
            timeout=request.timeout,
        )

        system_text, user_messages = _split_system(request.messages)

        kwargs = {
            "model": request.model,
            "messages": user_messages,
            "max_tokens": request.max_tokens or _DEFAULT_MAX_TOKENS,
        }
        if system_text:
            kwargs["system"] = system_text
        # Reasoning-only models (Opus 4.7+) reject ``temperature`` and
        # ``top_p``. We silently drop them for those models so the same
        # call signature works across the whole Claude family.
        is_reasoning_only = _is_reasoning_only_model(request.model)
        if request.temperature is not None and not is_reasoning_only:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None and not is_reasoning_only:
            kwargs["top_p"] = request.top_p
        if request.stop is not None:
            kwargs["stop_sequences"] = request.stop

        # Extended-thinking control. Anthropic's default for normal models is
        # already "no extended thinking", so we only need to do something when
        # the caller explicitly asks for thinking=True. We still respect an
        # explicit thinking=False as a no-op (default behaviour).
        if request.thinking is True:
            budget = request.thinking_budget or 1024
            # Anthropic requires budget < max_tokens.
            cap = max(kwargs["max_tokens"] - 1, 1)
            if budget >= kwargs["max_tokens"]:
                budget = cap
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        # response_format / json mode: Anthropic does not have a direct flag.
        # Callers should add it via the system prompt.
        kwargs.update({k: v for k, v in (request.extra or {}).items()
                       if k not in {"system", "messages", "model", "max_tokens"}})

        try:
            resp = client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e), retry_after=_retry_after(e)) from e
        except anthropic.APITimeoutError as e:
            raise RetryableProviderError(f"timeout: {e}") from e
        except anthropic.APIConnectionError as e:
            raise RetryableProviderError(f"connection error: {e}") from e
        except anthropic.APIStatusError as e:
            status = getattr(e, "status_code", None)
            if status is not None and status >= 500:
                raise RetryableProviderError(f"server error {status}: {e}") from e
            if status == 429:
                raise RateLimitError(str(e), retry_after=_retry_after(e)) from e
            raise ProviderError(f"HTTP {status}: {e}") from e
        except Exception as e:  # pragma: no cover
            raise ProviderError(f"unexpected error: {e}") from e

        content = _extract_text(resp)
        usage = getattr(resp, "usage", None)
        return ChatResponse(
            content=content,
            model=getattr(resp, "model", request.model),
            finish_reason=getattr(resp, "stop_reason", None),
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            raw=resp,
        )


def _split_system(messages):
    """Pull system messages into a single string; return (system, user_messages)."""
    system_parts = []
    user_messages = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part.get("text", ""))
        else:
            user_messages.append(m)
    return "\n\n".join(system_parts) if system_parts else None, user_messages


def _is_reasoning_only_model(model_id: str) -> bool:
    """Anthropic reasoning-only models reject ``temperature`` / ``top_p``.

    From Claude 4.7 (Opus) onward Anthropic moved to reasoning-only models;
    earlier 4.x and 3.x families still accept the sampling parameters.
    """
    if not model_id:
        return False
    m = model_id.lower()
    # Heuristic: families known to be reasoning-only.
    return (
        m.startswith("claude-opus-4-7")
        or m.startswith("claude-opus-5")
        or m.startswith("claude-sonnet-5")
    )


def _extract_text(resp) -> str:
    blocks = getattr(resp, "content", None) or []
    parts = []
    for b in blocks:
        # Anthropic returns ContentBlock objects with a `type` and `text`.
        t = getattr(b, "type", None)
        if t == "text":
            parts.append(getattr(b, "text", "") or "")
    return "".join(parts)


def _retry_after(e) -> Optional[float]:
    resp = getattr(e, "response", None)
    if resp is None:
        return None
    headers = getattr(resp, "headers", None) or {}
    val = headers.get("retry-after") or headers.get("Retry-After")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
