"""Google Gemini provider.

Uses the official ``google-genai`` SDK (the new unified Gen AI SDK).
Falls back to ``google-generativeai`` if only that is available.
Gemini's API uses a ``contents`` list with ``role`` in {"user", "model"}.
System prompts go through a separate ``system_instruction`` config field.
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


class GoogleProvider(Provider):
    def chat(self, request: ChatRequest, *, api_key: str) -> ChatResponse:
        try:
            from google import genai
            from google.genai import types as gtypes
        except ImportError:
            return self._chat_legacy(request, api_key=api_key)

        client = genai.Client(api_key=api_key)
        system_text, contents = _split_system(request.messages)

        cfg_kwargs = {}
        if request.temperature is not None:
            cfg_kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            cfg_kwargs["max_output_tokens"] = request.max_tokens
        if request.top_p is not None:
            cfg_kwargs["top_p"] = request.top_p
        if request.stop is not None:
            cfg_kwargs["stop_sequences"] = request.stop
        if system_text:
            cfg_kwargs["system_instruction"] = system_text
        if request.response_format and request.response_format.get("type") == "json_object":
            cfg_kwargs["response_mime_type"] = "application/json"

        # Thinking control for Gemini 2.5 family. Setting thinking_budget=0
        # disables the internal "thinking" stage; -1 lets the model decide.
        # Older 1.x / 2.0 models do not have a thinking phase and silently
        # ignore the field, so it is safe to always send when requested.
        thinking_cfg = _build_gemini_thinking_config(request, gtypes)
        if thinking_cfg is not None:
            cfg_kwargs["thinking_config"] = thinking_cfg

        try:
            resp = client.models.generate_content(
                model=request.model,
                contents=_to_gemini_contents(contents, gtypes),
                config=gtypes.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None,
            )
        except Exception as e:
            self._translate_error(e)

        content = getattr(resp, "text", None) or _extract_genai_text(resp)
        usage = getattr(resp, "usage_metadata", None)
        return ChatResponse(
            content=content or "",
            model=request.model,
            finish_reason=_extract_finish_reason(resp),
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
            raw=resp,
        )

    # -------------------------------------------------------------------
    # Legacy SDK fallback (google-generativeai)
    # -------------------------------------------------------------------
    def _chat_legacy(self, request: ChatRequest, *, api_key: str) -> ChatResponse:
        try:
            import google.generativeai as legacy
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Install one of: `pip install google-genai` (preferred) or "
                "`pip install google-generativeai`."
            ) from e

        legacy.configure(api_key=api_key)
        system_text, contents = _split_system(request.messages)

        generation_config = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            generation_config["max_output_tokens"] = request.max_tokens
        if request.top_p is not None:
            generation_config["top_p"] = request.top_p
        if request.stop is not None:
            generation_config["stop_sequences"] = request.stop
        if request.response_format and request.response_format.get("type") == "json_object":
            generation_config["response_mime_type"] = "application/json"

        thinking_cfg = _build_gemini_thinking_config(request, gtypes=None)
        if thinking_cfg is not None:
            generation_config["thinking_config"] = thinking_cfg

        model = legacy.GenerativeModel(
            model_name=request.model,
            system_instruction=system_text,
        )

        legacy_contents = []
        for m in contents:
            role = "user" if m["role"] == "user" else "model"
            text = m["content"] if isinstance(m["content"], str) else str(m["content"])
            legacy_contents.append({"role": role, "parts": [text]})

        try:
            resp = model.generate_content(
                legacy_contents,
                generation_config=generation_config or None,
                request_options={"timeout": request.timeout} if request.timeout else None,
            )
        except Exception as e:
            self._translate_error(e)

        content = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        return ChatResponse(
            content=content,
            model=request.model,
            finish_reason=_extract_finish_reason(resp),
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
            raw=resp,
        )

    # -------------------------------------------------------------------
    @staticmethod
    def _translate_error(e):
        msg = str(e).lower()
        if "429" in msg or "rate" in msg or "quota" in msg:
            raise RateLimitError(str(e)) from e
        if "timeout" in msg or "deadline" in msg:
            raise RetryableProviderError(str(e)) from e
        if "503" in msg or "500" in msg or "unavailable" in msg:
            raise RetryableProviderError(str(e)) from e
        raise ProviderError(str(e)) from e


def _build_gemini_thinking_config(request, gtypes):
    """Translate our unified ``thinking`` flag to Gemini's ThinkingConfig.

    ``thinking=False`` -> ``thinking_budget=0`` (disable).
    ``thinking=True``  -> caller-supplied budget, else -1 (let the model
    decide). ``thinking=None`` -> leave provider default in place.

    Older 1.x / 2.0 models silently ignore the field, so it is safe to
    always send when the caller asked for it.
    """
    if request.thinking is None:
        return None
    if request.thinking is False:
        budget = 0
    else:
        budget = request.thinking_budget if request.thinking_budget is not None else -1
    if gtypes is None:
        return {"thinking_budget": budget}
    try:
        return gtypes.ThinkingConfig(thinking_budget=budget)
    except (AttributeError, TypeError):
        return {"thinking_budget": budget}


def _split_system(messages):
    """Pull system messages into a string and translate roles to user/model."""
    system_parts = []
    out = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            continue
        if role == "assistant":
            role = "model"
        if role not in ("user", "model"):
            role = "user"
        out.append({"role": role, "content": content})
    return ("\n\n".join(system_parts) if system_parts else None), out


def _to_gemini_contents(messages, gtypes):
    """Convert our message list to google-genai Content objects."""
    contents = []
    for m in messages:
        role = m["role"]
        if role == "assistant":
            role = "model"
        text = m["content"] if isinstance(m["content"], str) else str(m["content"])
        contents.append(
            gtypes.Content(role=role, parts=[gtypes.Part(text=text)])
        )
    return contents


def _extract_genai_text(resp) -> str:
    """Concatenate text parts across all candidates / parts."""
    parts_out = []
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                parts_out.append(text)
    return "".join(parts_out)


def _extract_finish_reason(resp) -> Optional[str]:
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        return None
    return str(getattr(cands[0], "finish_reason", None) or "")
