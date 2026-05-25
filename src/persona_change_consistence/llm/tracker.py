"""Call tracing, cost accounting, and resumable checkpoints.

Each LLM call writes one JSON line to ``<trace_dir>/<experiment>.jsonl``.
The :class:`Tracker` accumulates token / cost statistics in memory and can
be queried at the end of a run.

For *resumable* experiments, the helper :func:`load_completed_ids` reads
back the trace file and returns the set of ``call_id`` values that already
succeeded. Experiment scripts can skip those before submitting new work.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from .config import LLMConfig, ModelPricing
from .providers import ChatRequest, ChatResponse

log = logging.getLogger(__name__)


@dataclass
class CallRecord:
    """One LLM call's persisted record (one JSONL line)."""

    call_id: str
    timestamp: float
    iso_time: str
    experiment: str
    model_alias: str
    provider: str
    provider_model: str
    key_id: str
    attempts: int
    success: bool
    error: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    input_cost_usd: Optional[float]
    output_cost_usd: Optional[float]
    total_cost_usd: Optional[float]
    latency_sec: float
    request: Dict[str, Any]
    response: Optional[Dict[str, Any]]


class Tracker:
    """Append-only JSONL tracer + in-memory cost accumulator."""

    def __init__(self, *, experiment: str, trace_dir: Path, config: LLMConfig):
        self.experiment = experiment
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.trace_dir / f"{experiment}.jsonl"
        self.config = config

        self._lock = threading.Lock()
        self._totals = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
        }
        self._per_model: Dict[str, Dict[str, float]] = {}

    # ------- recording -------
    def record(
        self,
        *,
        model_alias: str,
        provider: str,
        provider_model: str,
        request: ChatRequest,
        response: Optional[ChatResponse],
        key_id: str,
        attempts: int,
        latency_sec: float,
        error: Optional[Exception] = None,
        call_id: Optional[str] = None,
    ) -> CallRecord:
        ts = time.time()
        pricing = self.config.get_pricing(provider, provider_model)
        input_tokens = response.input_tokens if response else None
        output_tokens = response.output_tokens if response else None
        in_cost, out_cost, total_cost = _compute_cost(pricing, input_tokens, output_tokens)

        record = CallRecord(
            call_id=call_id or str(uuid.uuid4()),
            timestamp=ts,
            iso_time=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
            experiment=self.experiment,
            model_alias=model_alias,
            provider=provider,
            provider_model=provider_model,
            key_id=key_id,
            attempts=attempts,
            success=response is not None,
            error=str(error) if error else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=in_cost,
            output_cost_usd=out_cost,
            total_cost_usd=total_cost,
            latency_sec=latency_sec,
            request=_serialize_request(request),
            response=_serialize_response(response) if response else None,
        )

        line = json.dumps(asdict(record), ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._totals["calls"] += 1
            if record.success:
                self._totals["successes"] += 1
                self._totals["input_tokens"] += input_tokens or 0
                self._totals["output_tokens"] += output_tokens or 0
                self._totals["input_cost_usd"] += in_cost or 0.0
                self._totals["output_cost_usd"] += out_cost or 0.0
                self._totals["total_cost_usd"] += total_cost or 0.0
            else:
                self._totals["failures"] += 1
            bucket = self._per_model.setdefault(
                model_alias,
                {
                    "calls": 0,
                    "successes": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                },
            )
            bucket["calls"] += 1
            if record.success:
                bucket["successes"] += 1
                bucket["input_tokens"] += input_tokens or 0
                bucket["output_tokens"] += output_tokens or 0
                bucket["cost_usd"] += total_cost or 0.0

        return record

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "experiment": self.experiment,
                "trace_path": str(self.path),
                "totals": dict(self._totals),
                "by_model": {k: dict(v) for k, v in self._per_model.items()},
            }

    def print_summary(self) -> None:  # pragma: no cover - human helper
        s = self.summary()
        t = s["totals"]
        print(
            f"[{s['experiment']}] calls={t['calls']} ok={t['successes']} "
            f"fail={t['failures']} tokens_in={t['input_tokens']} "
            f"tokens_out={t['output_tokens']} cost=${t['total_cost_usd']:.4f}"
        )
        for model, b in s["by_model"].items():
            print(
                f"    {model}: calls={b['calls']} ok={b['successes']} "
                f"in={b['input_tokens']} out={b['output_tokens']} "
                f"cost=${b['cost_usd']:.4f}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_request(req: ChatRequest) -> Dict[str, Any]:
    return {
        "model": req.model,
        "messages": req.messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "top_p": req.top_p,
        "stop": req.stop,
        "response_format": req.response_format,
        "thinking": req.thinking,
        "thinking_budget": req.thinking_budget,
        "extra": req.extra,
    }


def _serialize_response(resp: ChatResponse) -> Dict[str, Any]:
    return {
        "content": resp.content,
        "model": resp.model,
        "finish_reason": resp.finish_reason,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
    }


def _compute_cost(
    pricing: ModelPricing,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
):
    if input_tokens is None and output_tokens is None:
        return None, None, None
    in_rate = pricing.input_per_1m
    out_rate = pricing.output_per_1m
    in_cost = (input_tokens or 0) / 1_000_000.0 * in_rate if in_rate is not None else None
    out_cost = (output_tokens or 0) / 1_000_000.0 * out_rate if out_rate is not None else None
    parts = [c for c in (in_cost, out_cost) if c is not None]
    total = sum(parts) if parts else None
    return in_cost, out_cost, total


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def load_completed_ids(trace_path: Path, *, only_success: bool = True) -> Set[str]:
    """Return the set of ``call_id`` values already in the trace file."""
    p = Path(trace_path)
    if not p.exists():
        return set()
    done: Set[str] = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("call_id")
            if not cid:
                continue
            if only_success and not rec.get("success"):
                continue
            done.add(cid)
    return done


def iter_trace(trace_path: Path) -> Iterable[Dict[str, Any]]:
    """Stream the trace file as parsed dicts."""
    p = Path(trace_path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
