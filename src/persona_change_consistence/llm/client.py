"""Unified LLM client.

Usage:

    from persona_change_consistence.llm import LLMClient

    client = LLMClient(experiment="multimodel_v1")
    out = client.chat(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.0,
    )
    print(out.content)

    # Parallel batch
    results = client.batch_chat(
        [
            {"model": "claude-sonnet-4", "messages": [...], "call_id": "p1"},
            {"model": "deepseek-v3",     "messages": [...], "call_id": "p2"},
        ],
        max_workers=8,
    )

    client.print_summary()
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from .config import LLMConfig, load_config
from .providers import (
    ChatRequest,
    ChatResponse,
    Provider,
    ProviderError,
    build_provider,
)
from .scheduler import KeyPool, Scheduler
from .tracker import Tracker, load_completed_ids

log = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """What :meth:`LLMClient.chat` returns."""

    call_id: str
    model_alias: str
    provider: str
    provider_model: str
    content: str
    finish_reason: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_usd: Optional[float]
    latency_sec: float
    attempts: int
    key_id: str
    success: bool = True
    error: Optional[str] = None


class LLMClient:
    """One client per experiment. Holds all provider+key state."""

    def __init__(
        self,
        *,
        experiment: str,
        config: Optional[LLMConfig] = None,
        config_path: Optional[Path] = None,
        trace_dir: Optional[Path] = None,
    ):
        self.experiment = experiment
        self.config = config or load_config(config_path)
        self._providers: Dict[str, Provider] = {}
        self._schedulers: Dict[str, Scheduler] = {}
        for name, prov_cfg in self.config.providers.items():
            provider = build_provider(
                name=name,
                type=prov_cfg.type,
                base_url=prov_cfg.base_url,
            )
            pool = KeyPool(prov_cfg)
            self._providers[name] = provider
            self._schedulers[name] = Scheduler(
                provider=provider, pool=pool, retry=self.config.retry,
            )

        self.tracker = Tracker(
            experiment=experiment,
            trace_dir=trace_dir or self.config.trace_dir,
            config=self.config,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """All model aliases the current configuration can route."""
        return sorted(self.config.model_routes.keys())

    def list_providers(self) -> List[str]:
        return sorted(self.config.providers.keys())

    def pool_stats(self) -> Dict[str, Any]:
        return {
            name: sched.pool.stats()
            for name, sched in self._schedulers.items()
        }

    # ------------------------------------------------------------------
    # Single call
    # ------------------------------------------------------------------

    def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[bool] = False,
        thinking_budget: Optional[int] = None,
        timeout: Optional[float] = None,
        call_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        raise_on_error: bool = True,
    ) -> LLMResult:
        provider_name, provider_model = self.config.resolve_model(model)
        sched = self._schedulers.get(provider_name)
        if sched is None:
            raise KeyError(
                f"Model alias '{model}' resolved to provider '{provider_name}' "
                f"but no usable keys exist for that provider."
            )

        req = ChatRequest(
            model=provider_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            thinking=thinking,
            thinking_budget=thinking_budget,
            extra=extra or {},
            timeout=timeout or self.config.timeout,
        )

        call_id = call_id or str(uuid.uuid4())
        t0 = time.monotonic()
        try:
            resp, attempts, key_id = sched.execute(req)
            latency = time.monotonic() - t0
            rec = self.tracker.record(
                model_alias=model,
                provider=provider_name,
                provider_model=provider_model,
                request=req,
                response=resp,
                key_id=key_id,
                attempts=attempts,
                latency_sec=latency,
                call_id=call_id,
            )
            return LLMResult(
                call_id=rec.call_id,
                model_alias=model,
                provider=provider_name,
                provider_model=provider_model,
                content=resp.content,
                finish_reason=resp.finish_reason,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost_usd=rec.total_cost_usd,
                latency_sec=latency,
                attempts=attempts,
                key_id=key_id,
            )
        except Exception as e:
            latency = time.monotonic() - t0
            self.tracker.record(
                model_alias=model,
                provider=provider_name,
                provider_model=provider_model,
                request=req,
                response=None,
                key_id="-",
                attempts=self.config.retry.max_attempts,
                latency_sec=latency,
                error=e,
                call_id=call_id,
            )
            if raise_on_error:
                raise
            return LLMResult(
                call_id=call_id,
                model_alias=model,
                provider=provider_name,
                provider_model=provider_model,
                content="",
                finish_reason=None,
                input_tokens=None,
                output_tokens=None,
                cost_usd=None,
                latency_sec=latency,
                attempts=self.config.retry.max_attempts,
                key_id="-",
                success=False,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def batch_chat(
        self,
        jobs: List[Dict[str, Any]],
        *,
        max_workers: int = 8,
        resume: bool = True,
        progress: bool = True,
        on_done: Optional[Callable[[LLMResult], None]] = None,
    ) -> List[LLMResult]:
        """Run many chat jobs in parallel.

        Each ``job`` is a dict with the same keyword args as :meth:`chat`,
        and ideally a stable ``call_id`` so re-runs can skip completed
        calls (when ``resume=True``).
        """
        skip_ids: Set[str] = set()
        if resume:
            skip_ids = load_completed_ids(self.tracker.path, only_success=True)
            if skip_ids:
                log.info(
                    "[LLMClient] resume: skipping %d already-completed calls",
                    len(skip_ids),
                )

        pending = []
        for job in jobs:
            cid = job.get("call_id")
            if cid and cid in skip_ids:
                continue
            pending.append(job)

        results: List[LLMResult] = []
        if not pending:
            log.info("[LLMClient] batch_chat: nothing to do.")
            return results

        log.info(
            "[LLMClient] batch_chat: %d jobs, %d workers, experiment=%s",
            len(pending), max_workers, self.experiment,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(self._chat_job, job): job for job in pending
            }
            done = 0
            for fut in as_completed(future_map):
                res = fut.result()
                results.append(res)
                done += 1
                if on_done is not None:
                    try:
                        on_done(res)
                    except Exception:  # pragma: no cover
                        log.exception("on_done callback failed")
                if progress and (done % 10 == 0 or done == len(pending)):
                    log.info(
                        "[LLMClient] progress: %d/%d", done, len(pending),
                    )
        return results

    def _chat_job(self, job: Dict[str, Any]) -> LLMResult:
        try:
            return self.chat(raise_on_error=False, **job)
        except Exception as e:  # pragma: no cover - defensive
            log.exception("chat job failed: %s", e)
            return LLMResult(
                call_id=job.get("call_id", str(uuid.uuid4())),
                model_alias=job.get("model", "?"),
                provider="?",
                provider_model="?",
                content="",
                finish_reason=None,
                input_tokens=None,
                output_tokens=None,
                cost_usd=None,
                latency_sec=0.0,
                attempts=0,
                key_id="-",
                success=False,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return self.tracker.summary()

    def print_summary(self) -> None:
        self.tracker.print_summary()
