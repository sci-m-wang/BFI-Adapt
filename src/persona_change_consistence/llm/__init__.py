"""Multi-provider LLM scheduling for the persona-change-consistence project.

Public API:

* :class:`LLMClient`     - the main entry point.
* :func:`load_config`    - read ``configs/api_keys.yaml``.
* :class:`LLMResult`     - return type from :meth:`LLMClient.chat`.
* :class:`ProviderError` - base exception for provider failures.
"""

from .client import LLMClient, LLMResult
from .config import LLMConfig, load_config
from .providers import (
    ChatRequest,
    ChatResponse,
    ProviderError,
    RateLimitError,
    RetryableProviderError,
)
from .tracker import Tracker, load_completed_ids

__all__ = [
    "LLMClient",
    "LLMResult",
    "LLMConfig",
    "load_config",
    "ChatRequest",
    "ChatResponse",
    "ProviderError",
    "RateLimitError",
    "RetryableProviderError",
    "Tracker",
    "load_completed_ids",
]
