"""Configuration loader for the LLM scheduler.

Reads ``configs/api_keys.yaml`` and produces typed in-memory config objects.
Performs basic validation but does not connect to any provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError as e:  # pragma: no cover - explicit guidance
    raise ImportError(
        "PyYAML is required for LLM config loading. "
        "Install with: pip install pyyaml"
    ) from e


__all__ = [
    "KeyConfig",
    "ModelPricing",
    "ProviderConfig",
    "RetryConfig",
    "LLMConfig",
    "load_config",
]


@dataclass
class KeyConfig:
    """A single API key with its rate-limit configuration."""

    id: str
    key: str
    rpm: int = 0  # requests-per-minute, 0 = unlimited
    tpm: int = 0  # tokens-per-minute, 0 = unlimited
    concurrency: int = 4
    enabled: bool = True


@dataclass
class ModelPricing:
    """USD price per 1M tokens (None = unknown / free)."""

    input_per_1m: Optional[float] = None
    output_per_1m: Optional[float] = None


@dataclass
class ProviderConfig:
    """A provider — possibly with multiple keys and a model price catalog."""

    name: str
    type: str  # openai_compat | anthropic | google
    base_url: Optional[str] = None
    keys: List[KeyConfig] = field(default_factory=list)
    models: Dict[str, ModelPricing] = field(default_factory=dict)


@dataclass
class RetryConfig:
    max_attempts: int = 5
    initial_backoff: float = 1.0
    max_backoff: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: float = 0.2


@dataclass
class LLMConfig:
    """Top-level resolved configuration."""

    providers: Dict[str, ProviderConfig]
    # Routing: alias -> (provider_name, provider_model_id)
    model_routes: Dict[str, tuple]
    retry: RetryConfig
    timeout: float = 120.0
    trace_dir: Path = field(default_factory=lambda: Path("llm_traces"))

    def resolve_model(self, alias: str) -> tuple:
        """Resolve a model alias to (provider_name, provider_model_id).

        Falls back to treating ``alias`` as a direct ``provider:model`` string,
        and finally to searching providers for a model with that exact id.
        """
        if alias in self.model_routes:
            return self.model_routes[alias]
        if ":" in alias:
            provider, model_id = alias.split(":", 1)
            if provider in self.providers:
                return (provider, model_id)
        # Search by exact match in any provider's model catalog.
        for prov_name, prov in self.providers.items():
            if alias in prov.models:
                return (prov_name, alias)
        raise KeyError(
            f"Unknown model alias '{alias}'. Declare it under `models:` in "
            f"your api_keys.yaml or pass 'provider:model_id'."
        )

    def get_pricing(self, provider: str, model_id: str) -> ModelPricing:
        prov = self.providers.get(provider)
        if not prov:
            return ModelPricing()
        return prov.models.get(model_id, ModelPricing())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


_DEFAULT_PATHS = [
    Path("configs/api_keys.yaml"),
    Path.home() / ".config" / "persona_change_consistence" / "api_keys.yaml",
]


def _find_config_path(explicit: Optional[Path]) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"LLM config not found at {p}")
        return p

    env = os.environ.get("PERSONA_LLM_CONFIG")
    if env:
        p = Path(env).expanduser()
        if not p.exists():
            raise FileNotFoundError(
                f"PERSONA_LLM_CONFIG={env} but the file does not exist."
            )
        return p

    for candidate in _DEFAULT_PATHS:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate api_keys.yaml. Either:\n"
        "  - copy configs/api_keys.example.yaml to configs/api_keys.yaml, or\n"
        "  - set PERSONA_LLM_CONFIG=/path/to/api_keys.yaml, or\n"
        "  - pass `path=` explicitly to load_config()."
    )


def _parse_key(raw: dict, *, provider: str, index: int) -> KeyConfig:
    if "key" not in raw:
        raise ValueError(
            f"Provider '{provider}' key[{index}] missing required field 'key'."
        )
    key_value = raw["key"]
    # Allow ${ENV_VAR} or env: prefix expansion for keys checked into source-like config.
    if isinstance(key_value, str) and key_value.startswith("${") and key_value.endswith("}"):
        env_name = key_value[2:-1]
        key_value = os.environ.get(env_name, "")
        if not key_value:
            raise ValueError(
                f"Provider '{provider}' key[{index}] references env var "
                f"{env_name} which is not set."
            )
    elif isinstance(key_value, str) and key_value.startswith("env:"):
        env_name = key_value[len("env:"):]
        key_value = os.environ.get(env_name, "")
        if not key_value:
            raise ValueError(
                f"Provider '{provider}' key[{index}] references env var "
                f"{env_name} which is not set."
            )

    return KeyConfig(
        id=raw.get("id") or f"{provider}-{index}",
        key=key_value,
        rpm=int(raw.get("rpm", 0) or 0),
        tpm=int(raw.get("tpm", 0) or 0),
        concurrency=int(raw.get("concurrency", 4) or 4),
        enabled=bool(raw.get("enabled", True)),
    )


def _parse_pricing(raw: Optional[dict]) -> ModelPricing:
    if not raw:
        return ModelPricing()
    return ModelPricing(
        input_per_1m=raw.get("input_per_1m"),
        output_per_1m=raw.get("output_per_1m"),
    )


def _parse_provider(name: str, raw: dict) -> ProviderConfig:
    prov_type = raw.get("type")
    if prov_type not in {"openai_compat", "anthropic", "google"}:
        raise ValueError(
            f"Provider '{name}' has invalid type '{prov_type}'. "
            "Must be one of: openai_compat, anthropic, google."
        )
    base_url = raw.get("base_url")
    if prov_type == "openai_compat" and not base_url:
        raise ValueError(
            f"Provider '{name}' is openai_compat but missing 'base_url'."
        )

    keys_raw = raw.get("keys") or []
    keys = [
        _parse_key(k, provider=name, index=i)
        for i, k in enumerate(keys_raw)
        if k is not None
    ]
    enabled_keys = [k for k in keys if k.enabled and k.key and "REPLACE_ME" not in k.key]

    models_raw = raw.get("models") or {}
    models = {
        mid: _parse_pricing(p)
        for mid, p in models_raw.items()
    }

    prov = ProviderConfig(
        name=name,
        type=prov_type,
        base_url=base_url,
        keys=enabled_keys,
        models=models,
    )
    return prov


def _parse_routes(raw: dict, providers: Dict[str, ProviderConfig]) -> Dict[str, tuple]:
    routes = {}
    for alias, target in (raw or {}).items():
        if not isinstance(target, str) or ":" not in target:
            raise ValueError(
                f"Model alias '{alias}' must map to 'provider:model_id' "
                f"(got: {target!r})."
            )
        provider, model_id = target.split(":", 1)
        if provider not in providers:
            # We do not fail hard here — the provider might just be disabled
            # (e.g. all keys had REPLACE_ME). Skip the route silently.
            continue
        routes[alias] = (provider, model_id)
    return routes


def _parse_retry(raw: Optional[dict]) -> RetryConfig:
    raw = raw or {}
    return RetryConfig(
        max_attempts=int(raw.get("max_attempts", 5)),
        initial_backoff=float(raw.get("initial_backoff", 1.0)),
        max_backoff=float(raw.get("max_backoff", 60.0)),
        backoff_multiplier=float(raw.get("backoff_multiplier", 2.0)),
        jitter=float(raw.get("jitter", 0.2)),
    )


def load_config(path: Optional[Path] = None) -> LLMConfig:
    """Load and validate the LLM configuration."""
    cfg_path = _find_config_path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw_providers = raw.get("providers") or {}
    providers: Dict[str, ProviderConfig] = {}
    for name, prov_raw in raw_providers.items():
        try:
            prov = _parse_provider(name, prov_raw or {})
        except ValueError as e:
            # Skip the provider but warn loudly.
            import warnings
            warnings.warn(f"Skipping provider '{name}': {e}", stacklevel=2)
            continue
        if not prov.keys:
            # No usable keys — silently drop.
            continue
        providers[name] = prov

    routes = _parse_routes(raw.get("models") or {}, providers)

    defaults = raw.get("defaults") or {}
    retry = _parse_retry(defaults.get("retry"))
    timeout = float(defaults.get("timeout", 120))
    trace_dir = Path(defaults.get("trace_dir", "llm_traces"))

    return LLMConfig(
        providers=providers,
        model_routes=routes,
        retry=retry,
        timeout=timeout,
        trace_dir=trace_dir,
    )
