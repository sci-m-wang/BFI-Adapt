"""OpenAI-compatible provider.

Covers vendors that expose the OpenAI Chat Completions API at a custom
base URL: OpenAI, DeepSeek, Qwen DashScope (compat mode), Zhipu GLM,
Moonshot, OpenRouter, Volcano Engine (Doubao), Xiaomi MiMo, SiliconFlow.

Uses the `openai` SDK as the HTTP layer to inherit good defaults and the
streaming / token-counting bits.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .base import (
    ChatRequest,
    ChatResponse,
    Provider,
    ProviderError,
    RateLimitError,
    RetryableProviderError,
)

log = logging.getLogger(__name__)


class OpenAICompatProvider(Provider):
    def __init__(self, *, name: str, base_url: str):
        super().__init__(name=name, base_url=base_url)
        if not base_url:
            raise ValueError(
                f"OpenAICompatProvider '{name}' requires a base_url."
            )

    def chat(self, request: ChatRequest, *, api_key: str) -> ChatResponse:
        try:
            from openai import OpenAI
            from openai import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                RateLimitError as SDKRateLimitError,
            )
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "openai SDK is required for OpenAI-compatible providers. "
                "Install with: pip install openai"
            ) from e

        client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=request.timeout,
        )

        kwargs: Dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
        }
        if request.temperature is not None:
            # GPT-5.1+ chat-latest variants reject ``temperature`` (and ``top_p``)
            # — the server pins them to its own defaults. Drop them silently so
            # callers can keep a uniform temperature setting across the pool.
            if not _openai_drops_sampling_params(self.name, request.model):
                kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            # OpenAI deprecated ``max_tokens`` for the GPT-5.1+ generation
            # in favor of ``max_completion_tokens``. Detect by provider + model
            # so we keep backward compatibility on every other vendor.
            if _openai_uses_max_completion_tokens(self.name, request.model):
                kwargs["max_completion_tokens"] = request.max_tokens
            else:
                kwargs["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            if not _openai_drops_sampling_params(self.name, request.model):
                kwargs["top_p"] = request.top_p
        if request.stop is not None:
            kwargs["stop"] = request.stop
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format

        # Translate the unified `thinking` flag to whatever this vendor wants.
        _apply_thinking(self.name, request, kwargs)

        # Caller-supplied extras win over our defaults.
        kwargs.update(request.extra or {})

        try:
            resp = client.chat.completions.create(**kwargs)
        except SDKRateLimitError as e:
            retry_after = _extract_retry_after(e)
            raise RateLimitError(str(e), retry_after=retry_after) from e
        except APITimeoutError as e:
            raise RetryableProviderError(f"timeout: {e}") from e
        except APIConnectionError as e:
            raise RetryableProviderError(f"connection error: {e}") from e
        except APIStatusError as e:
            status = getattr(e, "status_code", None)
            if status is not None and status >= 500:
                raise RetryableProviderError(f"server error {status}: {e}") from e
            if status == 429:
                raise RateLimitError(str(e), retry_after=_extract_retry_after(e)) from e
            raise ProviderError(f"HTTP {status}: {e}") from e
        except Exception as e:  # pragma: no cover - safety net
            raise ProviderError(f"unexpected error: {e}") from e

        return _parse_openai_response(resp)


# ---------------------------------------------------------------------------
# Thinking / reasoning translation
# ---------------------------------------------------------------------------


def _apply_thinking(provider_name: str, request: ChatRequest, kwargs: dict) -> None:
    """Translate ``request.thinking`` into the right per-vendor parameter.

    Idempotent and conservative: if the caller already set the relevant
    parameter through ``extra`` we do not overwrite it.
    """
    thinking = request.thinking
    if thinking is None:
        # Leave provider defaults alone.
        return

    name = provider_name.lower()

    # ------- Volcano Engine (Doubao) / DeepSeek / GLM family -------
    # All three accept ``thinking={"type": "enabled"|"disabled"}`` (or
    # GLM also "auto"). For reasoning-only models that refuse the flag
    # the server will respond with 400; in that case the caller can pass
    # ``thinking=None`` to suppress.
    if name in {"doubao", "deepseek", "glm"}:
        kwargs.setdefault(
            "extra_body",
            {},
        )
        # Preserve any caller-supplied extra_body fields.
        eb = dict(kwargs["extra_body"])
        eb.setdefault(
            "thinking",
            {"type": "enabled" if thinking else "disabled"},
        )
        kwargs["extra_body"] = eb
        return

    # ------- Qwen (Aliyun DashScope) -------
    # Qwen3 uses ``extra_body={"enable_thinking": bool}``.
    if name == "qwen":
        eb = dict(kwargs.get("extra_body") or {})
        eb.setdefault("enable_thinking", bool(thinking))
        kwargs["extra_body"] = eb
        return

    # ------- SiliconFlow (multi-vendor aggregator) -------
    # Per the OpenAPI spec, SiliconFlow exposes top-level ``enable_thinking``
    # and ``thinking_budget`` (NOT nested in extra_body.thinking). Only a
    # specific subset of models honors the flag — unsupported models will
    # 400-error if the field is sent. So we send it only when the requested
    # model id is in the supported list. The full list (as of 2026-05) is
    # documented at:
    # https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions
    if name == "siliconflow":
        if _siliconflow_supports_thinking(request.model):
            eb = dict(kwargs.get("extra_body") or {})
            eb.setdefault("enable_thinking", bool(thinking))
            if request.thinking_budget is not None and thinking:
                eb.setdefault("thinking_budget", int(request.thinking_budget))
            kwargs["extra_body"] = eb
        return

    # ------- Moonshot (Kimi) -------
    # Only the dedicated kimi-k2-thinking-* models do reasoning, and they
    # cannot be turned off. Plain kimi-k2-* never thinks. So we do nothing.
    if name == "moonshot":
        return

    # ------- OpenRouter -------
    # OpenRouter exposes a unified ``reasoning`` parameter — pass through.
    if name == "openrouter":
        eb = dict(kwargs.get("extra_body") or {})
        eb.setdefault(
            "reasoning",
            {"enabled": bool(thinking)},
        )
        kwargs["extra_body"] = eb
        return

    # ------- OpenAI proper -------
    # GPT-4o / GPT-4.1 do not reason; o-series and GPT-5 reasoning cannot
    # be turned off via API. Nothing to do here.
    if name == "openai":
        return

    # ------- MiMo / other vendors -------
    # No documented flag. Leave the request untouched.
    return


# SiliconFlow's documented list of chat models that accept ``enable_thinking``.
# Source: https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions
# Keep this in sync when SiliconFlow expands coverage. The keys are
# case-sensitive provider/model strings as listed in their /v1/models endpoint.
_SILICONFLOW_THINKING_SUPPORTED = frozenset({
    # GLM family
    "Pro/zai-org/GLM-5",
    "Pro/zai-org/GLM-4.7",
    "zai-org/GLM-4.6",
    "zai-org/GLM-4.5V",
    # DeepSeek family
    "deepseek-ai/DeepSeek-V3.2",
    "Pro/deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-V3.1-Terminus",
    "Pro/deepseek-ai/DeepSeek-V3.1-Terminus",
    # Qwen3 family
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B",
    # Qwen3.5 family
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-4B",
    # Tencent Hunyuan
    "tencent/Hunyuan-A13B-Instruct",
})


def _siliconflow_supports_thinking(model_id: str) -> bool:
    """Whether SiliconFlow exposes ``enable_thinking`` for this model."""
    return model_id in _SILICONFLOW_THINKING_SUPPORTED


def _openai_uses_max_completion_tokens(provider_name: str, model_id: str) -> bool:
    """Whether to send ``max_completion_tokens`` instead of ``max_tokens``.

    OpenAI deprecated ``max_tokens`` for GPT-5.1 and later generations. The
    GPT-5 first generation (``gpt-5``, ``gpt-5-chat-latest``, ``gpt-5-mini``,
    ``gpt-5-nano``) still accepts ``max_tokens``. From GPT-5.1 onward the
    server rejects it with HTTP 400.
    """
    if provider_name.lower() != "openai":
        return False
    m = model_id.lower()
    # All GPT-5.1+ variants (chat-latest, codex, mini, nano, pro, ...).
    if m.startswith(("gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5")):
        return True
    # o-series reasoning models also use max_completion_tokens, but we don't
    # route to them anyway. Cover the case in case someone adds an alias.
    if m.startswith(("o1", "o3", "o4")):
        return True
    return False


def _openai_drops_sampling_params(provider_name: str, model_id: str) -> bool:
    """Whether OpenAI rejects ``temperature`` / ``top_p`` for this model.

    GPT-5.1+ chat-latest variants pin both to server-side defaults and 400
    if a caller sends them. The first-generation GPT-5 ``gpt-5-chat-latest``
    still accepts both, so we whitelist that explicitly.
    """
    if provider_name.lower() != "openai":
        return False
    m = model_id.lower()
    if m.startswith(("gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5")):
        return True
    if m.startswith(("o1", "o3", "o4")):
        return True
    return False


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_openai_response(resp) -> ChatResponse:
    if not resp.choices:
        raise ProviderError("OpenAI-compatible response had no choices")
    choice = resp.choices[0]
    content = choice.message.content or ""
    usage = getattr(resp, "usage", None)
    return ChatResponse(
        content=content,
        model=getattr(resp, "model", "") or "",
        finish_reason=getattr(choice, "finish_reason", None),
        input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        raw=resp,
    )


def _extract_retry_after(e) -> Optional[float]:
    """Best-effort: pull Retry-After header from an openai SDK error."""
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
