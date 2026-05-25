"""Per-key scheduling: rate-limiting, concurrency, and key rotation.

A :class:`KeyPool` owns the runtime state for all keys of one provider:

* In-flight concurrency (a semaphore per key).
* Sliding-window rate limits (RPM / TPM).
* Cooldowns for keys that have been rate-limited or banned.

The scheduler picks the *least-loaded* eligible key; if none is free it
waits on the soonest-available cooldown / semaphore release.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from .config import KeyConfig, ProviderConfig, RetryConfig
from .providers import (
    ChatRequest,
    ChatResponse,
    Provider,
    ProviderError,
    RateLimitError,
    RetryableProviderError,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-key runtime state
# ---------------------------------------------------------------------------


@dataclass
class _KeyState:
    config: KeyConfig
    semaphore: threading.BoundedSemaphore
    # Sliding-window timestamps (deques of monotonic times).
    request_times: Deque[float] = field(default_factory=deque)
    # Each entry: (timestamp, tokens).
    token_events: Deque[tuple] = field(default_factory=deque)
    cooldown_until: float = 0.0
    in_flight: int = 0
    total_requests: int = 0
    total_failures: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def id(self) -> str:
        return self.config.id


# ---------------------------------------------------------------------------
# Key pool
# ---------------------------------------------------------------------------


class KeyPool:
    """Manages the lifecycle of all keys for a single provider."""

    def __init__(self, provider_cfg: ProviderConfig):
        self.provider_cfg = provider_cfg
        self._keys: List[_KeyState] = [
            _KeyState(
                config=k,
                semaphore=threading.BoundedSemaphore(max(1, k.concurrency)),
            )
            for k in provider_cfg.keys
        ]
        if not self._keys:
            raise ValueError(
                f"KeyPool for provider '{provider_cfg.name}' has no usable keys."
            )
        self._cv = threading.Condition()

    # ------- selection -------
    def _eligible(self, now: float) -> List[_KeyState]:
        eligible = []
        for k in self._keys:
            if k.cooldown_until > now:
                continue
            if not _within_rpm(k, now):
                continue
            # We do not pre-check TPM (token usage is best-effort post-hoc).
            eligible.append(k)
        return eligible

    def _pick(self, now: float) -> Optional[_KeyState]:
        eligible = self._eligible(now)
        if not eligible:
            return None
        # Prefer least in-flight (load balance), tiebreak by lowest total reqs.
        eligible.sort(key=lambda k: (k.in_flight, k.total_requests))
        return eligible[0]

    def _earliest_available(self, now: float) -> float:
        candidates = []
        for k in self._keys:
            if k.cooldown_until > now:
                candidates.append(k.cooldown_until)
            rpm_wait = _rpm_wait_until(k, now)
            if rpm_wait is not None:
                candidates.append(rpm_wait)
        return min(candidates) if candidates else now + 0.5

    # ------- public API -------
    def acquire(self, *, timeout: Optional[float] = None) -> _KeyState:
        """Block until a key is free, then mark it in-flight."""
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        with self._cv:
            while True:
                now = time.monotonic()
                k = self._pick(now)
                if k is not None:
                    # Try to acquire the semaphore non-blocking.
                    if k.semaphore.acquire(blocking=False):
                        k.in_flight += 1
                        return k
                # Wait until next eligibility window or semaphore release.
                wait_until = self._earliest_available(now)
                wait_for = max(0.05, min(2.0, wait_until - now))
                if deadline is not None:
                    remain = deadline - now
                    if remain <= 0:
                        raise TimeoutError("KeyPool.acquire timed out")
                    wait_for = min(wait_for, remain)
                self._cv.wait(timeout=wait_for)

    def release(self, k: _KeyState, *, tokens: int = 0, success: bool = True) -> None:
        with self._cv:
            now = time.monotonic()
            with k.lock:
                k.in_flight = max(0, k.in_flight - 1)
                k.total_requests += 1
                if not success:
                    k.total_failures += 1
                k.request_times.append(now)
                if tokens > 0:
                    k.token_events.append((now, tokens))
            k.semaphore.release()
            self._cv.notify_all()

    def cooldown(self, k: _KeyState, seconds: float) -> None:
        """Park this key for `seconds` (e.g. after a 429)."""
        with self._cv:
            new_until = time.monotonic() + max(0.0, seconds)
            if new_until > k.cooldown_until:
                k.cooldown_until = new_until
            log.warning(
                "[KeyPool:%s] cooldown key=%s for %.1fs",
                self.provider_cfg.name, k.id, seconds,
            )
            self._cv.notify_all()

    def stats(self) -> Dict:
        return {
            "provider": self.provider_cfg.name,
            "keys": [
                {
                    "id": k.id,
                    "in_flight": k.in_flight,
                    "total_requests": k.total_requests,
                    "total_failures": k.total_failures,
                    "cooldown_remaining": max(0.0, k.cooldown_until - time.monotonic()),
                }
                for k in self._keys
            ],
        }


def _within_rpm(k: _KeyState, now: float) -> bool:
    """Return True if firing one more request right now would stay under RPM."""
    if k.config.rpm <= 0:
        return True
    window_start = now - 60.0
    # Drop expired timestamps.
    while k.request_times and k.request_times[0] < window_start:
        k.request_times.popleft()
    return len(k.request_times) < k.config.rpm


def _rpm_wait_until(k: _KeyState, now: float) -> Optional[float]:
    if k.config.rpm <= 0:
        return None
    window_start = now - 60.0
    while k.request_times and k.request_times[0] < window_start:
        k.request_times.popleft()
    if len(k.request_times) < k.config.rpm:
        return None
    # Earliest time the oldest request falls out of the window.
    return k.request_times[0] + 60.0


# ---------------------------------------------------------------------------
# Scheduler entry point: execute a request with retries on a chosen key.
# ---------------------------------------------------------------------------


class Scheduler:
    """Wraps a Provider with a KeyPool and retry logic."""

    def __init__(
        self,
        *,
        provider: Provider,
        pool: KeyPool,
        retry: RetryConfig,
    ):
        self.provider = provider
        self.pool = pool
        self.retry = retry

    def execute(self, request: ChatRequest) -> tuple:
        """Run the request with retries.

        Returns ``(ChatResponse, attempts, key_id)``.
        """
        attempt = 0
        last_err: Optional[Exception] = None
        while attempt < self.retry.max_attempts:
            attempt += 1
            k = self.pool.acquire()
            success = False
            try:
                resp = self.provider.chat(request, api_key=k.config.key)
                success = True
                tokens = (resp.input_tokens or 0) + (resp.output_tokens or 0)
                self.pool.release(k, tokens=tokens, success=True)
                return resp, attempt, k.id
            except RateLimitError as e:
                # Cool this key down. Honor server hint if present.
                wait = e.retry_after if e.retry_after else self._backoff(attempt)
                self.pool.release(k, success=False)
                self.pool.cooldown(k, wait)
                last_err = e
                log.warning(
                    "[Scheduler] 429 on key=%s attempt=%d wait=%.1fs",
                    k.id, attempt, wait,
                )
                continue
            except RetryableProviderError as e:
                self.pool.release(k, success=False)
                wait = self._backoff(attempt)
                last_err = e
                log.warning(
                    "[Scheduler] retryable error on key=%s attempt=%d wait=%.1fs: %s",
                    k.id, attempt, wait, e,
                )
                time.sleep(wait)
                continue
            except ProviderError as e:
                # Non-retryable.
                if not success:
                    self.pool.release(k, success=False)
                raise
            except Exception as e:  # pragma: no cover
                if not success:
                    self.pool.release(k, success=False)
                raise ProviderError(f"unexpected: {e}") from e

        raise last_err if last_err else ProviderError("exhausted retries")

    def _backoff(self, attempt: int) -> float:
        base = self.retry.initial_backoff * (self.retry.backoff_multiplier ** (attempt - 1))
        base = min(base, self.retry.max_backoff)
        if self.retry.jitter > 0:
            base *= 1.0 + random.uniform(-self.retry.jitter, self.retry.jitter)
        return max(0.0, base)
