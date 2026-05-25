"""Full V2 multi-model experiment runner.

Multi-provider replication of ``scripts/full_experiment_v2.py``. Reuses the V2
persona definitions and life-event prompts unchanged (canonical per the docs in
``docs/source/experiment_v2.rst``) and runs each model in three parallel batch
phases via :class:`persona_change_consistence.llm.LLMClient`:

  Phase 1  - Baselines           (100 jobs / model)
  Phase 2a - Reflections         (100 x 11 = 1100 jobs / model)
  Phase 2b - Post-event BFI-44   (1100 jobs / model, depends on 2a content)

Per model output (matches the single-provider V2 schema for downstream analysis
compatibility):

    results/v2_multi_<timestamp>/<model_alias>/
        baselines.json
        full_results.json

The ``call_id`` for every job is stable and includes
``<experiment>_<model_alias>_<phase>_<persona_id>[_<event>]``, so re-running the
script picks up where it left off via :func:`LLMClient.batch_chat`'s built-in
``resume`` logic backed by the per-experiment JSONL trace.

Usage (pilot):

    python scripts/full_experiment_v2_multi.py --pilot --models sf-qwen3-235b

Usage (curated 12-model preset, full 100 personas x 11 events):

    python scripts/full_experiment_v2_multi.py --preset curated

Usage (custom list):

    python scripts/full_experiment_v2_multi.py \
        --models gpt-5.3-chat claude-sonnet-4.6 gemini-3-flash \
        --output-dir results/v2_multi_smoke

The new runner only exists alongside the canonical
``scripts/full_experiment_v2.py``; it does NOT modify the original, which
remains the reference single-provider implementation for reproducing the
existing Qwen3-235B numbers used in the paper.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Make the local ``scripts`` package importable so we can reuse V2 constants.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.full_experiment_v2 import (  # noqa: E402  (after sys.path tweak)
    BFI_ITEMS,
    GENDERS,
    CONTINENTS,
    LIFE_EVENTS,
    PERSONALITY_DESCRIPTIONS,
    analyze_event_results,
    calculate_trait_scores,
    create_persona_system_prompt,
    create_single_call_prompt,
    extract_single_call_scores,
    generate_all_personas,
)

from persona_change_consistence.llm import LLMClient, LLMResult  # noqa: E402
from persona_change_consistence.llm.tracker import iter_trace  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("full_experiment_v2_multi")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Curated model presets
# ---------------------------------------------------------------------------

# Representative models, 1-3 per family. Confirmed all non-reasoning when
# run with thinking=False. See the project recap for the full audit.
#
# gemini-3-flash was originally in this list but excluded after the first
# full run: the Google API key in configs/api_keys.yaml is on the FREE tier
# (5 RPM cap per model), which is unworkable for ~3300 calls/model. Re-add
# after upgrading the Google billing plan.
CURATED_MODELS: List[str] = [
    # OpenAI (chat-latest only; reasoning models excluded)
    "gpt-5.3-chat",
    "gpt-4.1-mini",
    # Anthropic (Sonnet over Opus for cost: ~$10/run vs ~$80/run)
    "claude-sonnet-4.6",
    "claude-haiku-4.5",
    # DeepSeek native
    "deepseek-v4-pro",
    # Doubao
    "doubao-seed-2.0-pro",
    # MiMo (needs higher max_tokens; see PER_MODEL_MAX_TOKENS)
    "mimo-v2.5-pro",
    # SiliconFlow (one model per upstream family; uses per-provider
    # concurrency cap PROVIDER_MAX_WORKERS to stay under SF's RPM)
    "sf-qwen3-235b",
    "sf-glm-4.6",
    "sf-kimi-k2-0905",
    "sf-minimax-m2.5",
]

# Models that need a larger output budget than the default 1600. Discovered
# empirically: mimo-v2.5-pro burns through 1600 tokens on hidden CoT for
# ~12% of BFI baselines and ~6% of post-event BFI. 3200 leaves headroom.
PER_MODEL_MAX_TOKENS: Dict[str, Dict[str, int]] = {
    "mimo-v2.5-pro": {"baseline": 3200, "post": 3200, "reflection": 1500},
}

# Per-provider concurrent worker cap. SiliconFlow YAML allows concurrency=8
# and rpm=500. The original rescue run held cap=2 out of paranoia about 429s,
# but observed: 2323 SF calls over 6h11m at cap=2 yielded zero 429s — pure
# throughput-limited. Raising to 8 cuts ETA ~4x. Client has retry on 429.
PROVIDER_MAX_WORKERS: Dict[str, int] = {
    "siliconflow": 8,
}

PRESETS: Dict[str, List[str]] = {
    "curated": CURATED_MODELS,
    "openai-only": [m for m in CURATED_MODELS if m.startswith("gpt-")],
    "siliconflow-only": [m for m in CURATED_MODELS if m.startswith("sf-")],
    "smoke": ["sf-qwen3-235b"],  # cheapest known-good single model for smoke tests
}


# ---------------------------------------------------------------------------
# Pilot-mode helpers (mirror full_experiment_v2.py)
# ---------------------------------------------------------------------------

PILOT_EVENTS = ["graduation", "promotion", "chronic_illness"]


def _pilot_personas() -> List[Dict[str, Any]]:
    """10 Asian Male personas (1 per personality type) - matches V2 pilot."""
    return [
        {
            "id": f"M_Asia_{pid}",
            "gender": "Male",
            "continent": "Asia",
            "personality_id": pid,
        }
        for pid in PERSONALITY_DESCRIPTIONS.keys()
    ]


# ---------------------------------------------------------------------------
# call_id helpers
# ---------------------------------------------------------------------------

def _baseline_call_id(experiment: str, model_alias: str, persona_id: str) -> str:
    return f"{experiment}::{model_alias}::baseline::{persona_id}"


def _reflect_call_id(
    experiment: str, model_alias: str, persona_id: str, event: str
) -> str:
    return f"{experiment}::{model_alias}::reflect::{persona_id}::{event}"


def _post_call_id(
    experiment: str, model_alias: str, persona_id: str, event: str
) -> str:
    return f"{experiment}::{model_alias}::post::{persona_id}::{event}"


def _load_trace_results(
    trace_path: Path, wanted_ids: Iterable[str]
) -> Dict[str, LLMResult]:
    """Reconstruct LLMResult dicts for the given call_ids from a trace JSONL.

    Used to recover prior outputs when ``batch_chat`` resumes a partial run
    and therefore returns only newly executed jobs. Reads the file once.
    """
    wanted = set(wanted_ids)
    if not wanted or not trace_path.exists():
        return {}
    out: Dict[str, LLMResult] = {}
    for rec in iter_trace(trace_path):
        cid = rec.get("call_id")
        if cid not in wanted:
            continue
        if not rec.get("success"):
            continue
        resp = rec.get("response") or {}
        out[cid] = LLMResult(
            call_id=cid,
            model_alias=rec.get("model_alias", "?"),
            provider=rec.get("provider", "?"),
            provider_model=rec.get("provider_model", "?"),
            content=resp.get("content") or "",
            finish_reason=resp.get("finish_reason"),
            input_tokens=resp.get("input_tokens"),
            output_tokens=resp.get("output_tokens"),
            cost_usd=rec.get("total_cost_usd"),
            latency_sec=rec.get("latency_sec", 0.0),
            attempts=rec.get("attempts", 1),
            key_id=rec.get("key_id", "-"),
            success=True,
            error=None,
        )
    return out


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------

def _build_baseline_jobs(
    experiment: str,
    model_alias: str,
    personas: List[Dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """One BFI-44 single-call baseline measurement per persona."""
    jobs: List[Dict[str, Any]] = []
    bfi_user_prompt = create_single_call_prompt()
    for persona in personas:
        system_prompt = create_persona_system_prompt(
            persona["gender"], persona["continent"], persona["personality_id"]
        )
        jobs.append(
            {
                "call_id": _baseline_call_id(experiment, model_alias, persona["id"]),
                "model": model_alias,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": bfi_user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "thinking": False,
            }
        )
    return jobs


def _build_reflection_jobs(
    experiment: str,
    model_alias: str,
    personas: List[Dict[str, Any]],
    events: List[str],
    *,
    temperature: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """One free-text reflection per (persona, event)."""
    jobs: List[Dict[str, Any]] = []
    for persona in personas:
        system_prompt = create_persona_system_prompt(
            persona["gender"], persona["continent"], persona["personality_id"]
        )
        for event in events:
            event_info = LIFE_EVENTS[event]
            event_msg = (
                event_info["notification"] + "\n\n" + event_info["reflection_prompt"]
            )
            jobs.append(
                {
                    "call_id": _reflect_call_id(
                        experiment, model_alias, persona["id"], event
                    ),
                    "model": model_alias,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": event_msg},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "thinking": False,
                }
            )
    return jobs


def _build_post_event_jobs(
    experiment: str,
    model_alias: str,
    personas: List[Dict[str, Any]],
    events: List[str],
    reflections: Dict[Tuple[str, str], str],
    *,
    temperature: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """One BFI-44 measurement per (persona, event) conditioned on the model's own reflection."""
    jobs: List[Dict[str, Any]] = []
    bfi_user_prompt = create_single_call_prompt()
    skipped = 0
    for persona in personas:
        system_prompt = create_persona_system_prompt(
            persona["gender"], persona["continent"], persona["personality_id"]
        )
        for event in events:
            reflection_content = reflections.get((persona["id"], event))
            if reflection_content is None:
                skipped += 1
                continue  # Skip cells where the reflection call failed.
            event_info = LIFE_EVENTS[event]
            event_msg = (
                event_info["notification"] + "\n\n" + event_info["reflection_prompt"]
            )
            jobs.append(
                {
                    "call_id": _post_call_id(
                        experiment, model_alias, persona["id"], event
                    ),
                    "model": model_alias,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": event_msg},
                        {"role": "assistant", "content": reflection_content},
                        {"role": "user", "content": bfi_user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "thinking": False,
                }
            )
    if skipped:
        log.warning(
            "[%s] Skipping %d post-event jobs due to missing reflections.",
            model_alias, skipped,
        )
    return jobs


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _parse_bfi(result: LLMResult) -> Dict[str, Any]:
    """Parse a successful BFI result into the V2 schema dict."""
    if not result.success or not result.content:
        return {
            "method": "single",
            "trait_scores": None,
            "item_scores": None,
            "items_extracted": 0,
            "raw_response": result.content or "",
            "error": result.error,
        }
    scores = extract_single_call_scores(result.content)
    trait_scores = calculate_trait_scores(scores) if scores else None
    return {
        "method": "single",
        "trait_scores": trait_scores,
        "item_scores": scores,
        "items_extracted": len(scores),
        "raw_response": result.content,
    }


# ---------------------------------------------------------------------------
# Single-model runner
# ---------------------------------------------------------------------------

def run_one_model(
    *,
    client: LLMClient,
    experiment: str,
    model_alias: str,
    personas: List[Dict[str, Any]],
    events: List[str],
    output_dir: Path,
    max_workers: int,
    baseline_temp: float = 0.0,
    # Note: GPT-5.x-chat-latest and similar non-reasoning-but-still-chain-of-thought
    # models burn some output budget on hidden CoT before emitting visible tokens.
    # At max_tokens=512 (canonical V2 setting tuned for vLLM-served Qwen3-235B),
    # 33% of gpt-5.3-chat baselines were truncated to empty strings. 1600 leaves
    # comfortable headroom: a BFI-44 numbered list is ~250 visible tokens.
    baseline_max_tokens: int = 1600,
    reflection_temp: float = 0.7,
    reflection_max_tokens: int = 800,
    post_temp: float = 0.0,
    post_max_tokens: int = 1600,
) -> Dict[str, Any]:
    """Run all three phases for one model_alias.

    Writes ``baselines.json`` + ``full_results.json`` into
    ``output_dir/<model_alias>/`` in the same schema as ``full_experiment_v2.py``.
    """
    model_dir = output_dir / model_alias.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    log.info("=" * 78)
    log.info(
        "Model %s | personas=%d | events=%d | total_cells=%d",
        model_alias, len(personas), len(events), len(personas) * len(events),
    )
    log.info("=" * 78)

    t_start = time.monotonic()

    # ----------------- Phase 1: Baselines -----------------
    log.info("[%s] Phase 1 — baselines (%d jobs)", model_alias, len(personas))
    baseline_jobs = _build_baseline_jobs(
        experiment, model_alias, personas,
        temperature=baseline_temp, max_tokens=baseline_max_tokens,
    )
    baseline_results = client.batch_chat(
        baseline_jobs, max_workers=max_workers, resume=True, progress=True,
    )

    # Index baselines by persona_id (recover from existing trace too)
    baseline_by_persona: Dict[str, LLMResult] = {}
    for r in baseline_results:
        # call_id format: experiment::model::baseline::persona_id
        try:
            persona_id = r.call_id.split("::")[-1]
            baseline_by_persona[persona_id] = r
        except Exception:
            continue
    # Fill in any baselines that were already completed in a prior run
    missing_baseline_ids = {
        j["call_id"] for j in baseline_jobs if j["call_id"] not in {r.call_id for r in baseline_results}
    }
    if missing_baseline_ids:
        recovered = _load_trace_results(client.tracker.path, missing_baseline_ids)
        log.info(
            "[%s] Recovered %d baselines from prior trace",
            model_alias, len(recovered),
        )
        for cid, r in recovered.items():
            baseline_by_persona[cid.split("::")[-1]] = r

    # Write the per-model baselines.json (V2-compatible)
    baselines: Dict[str, Optional[Dict[str, Any]]] = {}
    for persona in personas:
        pid = persona["id"]
        r = baseline_by_persona.get(pid)
        if r is None or not r.success:
            baselines[pid] = None
            continue
        parsed = _parse_bfi(r)
        baselines[pid] = {
            "method": "single",
            "trait_scores": parsed["trait_scores"],
            "item_scores": parsed["item_scores"],
            "raw_response": parsed["raw_response"],
            "items_extracted": parsed["items_extracted"],
        }
    with open(model_dir / "baselines.json", "w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2, ensure_ascii=False)
    n_ok = sum(1 for v in baselines.values() if v is not None and v["trait_scores"])
    log.info(
        "[%s] Baselines complete: %d/%d ok",
        model_alias, n_ok, len(personas),
    )

    # ----------------- Phase 2a: Reflections -----------------
    log.info(
        "[%s] Phase 2a — reflections (%d jobs)",
        model_alias, len(personas) * len(events),
    )
    reflect_jobs = _build_reflection_jobs(
        experiment, model_alias, personas, events,
        temperature=reflection_temp, max_tokens=reflection_max_tokens,
    )
    reflect_results = client.batch_chat(
        reflect_jobs, max_workers=max_workers, resume=True, progress=True,
    )

    # Index reflections by (persona_id, event)
    reflections: Dict[Tuple[str, str], str] = {}
    fresh_reflect_ids = {r.call_id for r in reflect_results}
    for r in reflect_results:
        if not r.success or not r.content:
            continue
        parts = r.call_id.split("::")
        # experiment::model::reflect::persona_id::event
        if len(parts) >= 5 and parts[2] == "reflect":
            reflections[(parts[3], parts[4])] = r.content
    # Recover reflections completed in a prior run
    missing_reflect_ids = {
        j["call_id"] for j in reflect_jobs if j["call_id"] not in fresh_reflect_ids
    }
    if missing_reflect_ids:
        recovered = _load_trace_results(client.tracker.path, missing_reflect_ids)
        log.info(
            "[%s] Recovered %d reflections from prior trace",
            model_alias, len(recovered),
        )
        for cid, r in recovered.items():
            parts = cid.split("::")
            if len(parts) >= 5 and parts[2] == "reflect" and r.content:
                reflections[(parts[3], parts[4])] = r.content
    log.info("[%s] Reflections complete: %d ok", model_alias, len(reflections))

    # ----------------- Phase 2b: Post-event BFI -----------------
    log.info(
        "[%s] Phase 2b — post-event BFI (%d eligible jobs)",
        model_alias, len(reflections),
    )
    post_jobs = _build_post_event_jobs(
        experiment, model_alias, personas, events, reflections,
        temperature=post_temp, max_tokens=post_max_tokens,
    )
    post_results = client.batch_chat(
        post_jobs, max_workers=max_workers, resume=True, progress=True,
    )

    # Index post-event by (persona_id, event)
    post_by_pe: Dict[Tuple[str, str], LLMResult] = {}
    fresh_post_ids = {r.call_id for r in post_results}
    for r in post_results:
        parts = r.call_id.split("::")
        # experiment::model::post::persona_id::event
        if len(parts) >= 5 and parts[2] == "post":
            post_by_pe[(parts[3], parts[4])] = r
    # Recover post-event results completed in a prior run
    missing_post_ids = {
        j["call_id"] for j in post_jobs if j["call_id"] not in fresh_post_ids
    }
    if missing_post_ids:
        recovered = _load_trace_results(client.tracker.path, missing_post_ids)
        log.info(
            "[%s] Recovered %d post-event BFI results from prior trace",
            model_alias, len(recovered),
        )
        for cid, r in recovered.items():
            parts = cid.split("::")
            if len(parts) >= 5 and parts[2] == "post":
                post_by_pe[(parts[3], parts[4])] = r

    # ----------------- Aggregate full_results.json -----------------
    all_results: List[Dict[str, Any]] = []
    for persona in personas:
        pid = persona["id"]
        baseline_entry = baselines.get(pid)
        if baseline_entry is None or baseline_entry["trait_scores"] is None:
            continue
        baseline_trait_scores = baseline_entry["trait_scores"]
        for event in events:
            reflection = reflections.get((pid, event))
            r = post_by_pe.get((pid, event))
            if reflection is None or r is None or not r.success:
                continue
            parsed_post = _parse_bfi(r)
            if parsed_post["trait_scores"] is None:
                continue
            change = {
                t: parsed_post["trait_scores"][t] - baseline_trait_scores[t]
                for t in baseline_trait_scores
            }
            event_info = LIFE_EVENTS[event]
            user_msg = (
                event_info["notification"] + "\n\n" + event_info["reflection_prompt"]
            )
            all_results.append(
                {
                    "persona_id": pid,
                    "persona": persona,
                    "event": event,
                    "baseline": baseline_entry,
                    "post_event": {
                        "method": "single",
                        "trait_scores": parsed_post["trait_scores"],
                        "item_scores": parsed_post["item_scores"],
                        "raw_response": parsed_post["raw_response"],
                        "items_extracted": parsed_post["items_extracted"],
                    },
                    "change": change,
                    "reflection": reflection,
                    "reflection_messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": reflection},
                    ],
                }
            )

    # Per-event summary (mirror full_experiment_v2 console output)
    by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_results:
        by_event[r["event"]].append(r)
    for event in events:
        event_results = by_event.get(event, [])
        if not event_results:
            continue
        try:
            ea = analyze_event_results(event_results, event)
        except Exception as e:
            log.warning("[%s] analyze_event_results failed for %s: %s", model_alias, event, e)
            continue
        log.info(
            "[%s] %-18s n=%d | %s",
            model_alias, event, len(event_results),
            " ".join(
                f"{t[0]}={(d.get('match_rate') or 0) * 100:4.1f}%"
                f"{'*' if d.get('significant') else ' '}"
                for t, d in ea["traits"].items()
                if d.get("match_rate") is not None
            ),
        )

    # Coerce numpy types before dumping
    def _serialize(obj: Any) -> Any:
        try:
            import numpy as np
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        return obj

    results_path = model_dir / "full_results.json"
    payload = {
        "config": {
            "model": model_alias,
            "method": "single",
            "n_personas": len(personas),
            "n_events": len(events),
            "timestamp": datetime.now().isoformat(),
            "version": "v2_multi",
            "experiment": experiment,
        },
        "results": all_results,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(_serialize(payload), f, indent=2, ensure_ascii=False)

    elapsed = time.monotonic() - t_start
    summary = {
        "model_alias": model_alias,
        "n_baselines": n_ok,
        "n_reflections": len(reflections),
        "n_full_cells": len(all_results),
        "elapsed_sec": round(elapsed, 1),
        "results_path": str(results_path.relative_to(output_dir.parent)),
    }
    log.info(
        "[%s] DONE in %.1fs — %d/%d cells",
        model_alias, elapsed, len(all_results), len(personas) * len(events),
    )
    return summary


# ---------------------------------------------------------------------------
# Multi-model orchestration
# ---------------------------------------------------------------------------

def run_multi_model(
    *,
    models: List[str],
    personas: List[Dict[str, Any]],
    events: List[str],
    output_dir: Path,
    experiment: Optional[str] = None,
    max_workers: int = 16,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run all (model, persona, event) cells. Writes an index.json at the end."""
    experiment = experiment or f"v2_multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Experiment: %s", experiment)
    log.info("Output dir: %s", output_dir)
    log.info("Models   : %d (%s)", len(models), ", ".join(models))
    log.info("Personas : %d", len(personas))
    log.info("Events   : %d (%s)", len(events), ", ".join(events))
    log.info(
        "Total cells planned: %d", len(models) * len(personas) * len(events),
    )

    client = LLMClient(experiment=experiment, config_path=config_path)

    # Sanity: ensure every requested model alias resolves
    unknown = [m for m in models if m not in client.list_models()]
    if unknown:
        raise SystemExit(
            f"Unknown model aliases in api_keys.yaml: {unknown}.\n"
            f"Available aliases: {client.list_models()}"
        )

    summaries: List[Dict[str, Any]] = []
    for model_alias in models:
        # Per-model overrides (from PER_MODEL_MAX_TOKENS / PROVIDER_MAX_WORKERS)
        overrides = PER_MODEL_MAX_TOKENS.get(model_alias, {})
        kwargs: Dict[str, Any] = {}
        if "baseline" in overrides:
            kwargs["baseline_max_tokens"] = overrides["baseline"]
        if "post" in overrides:
            kwargs["post_max_tokens"] = overrides["post"]
        if "reflection" in overrides:
            kwargs["reflection_max_tokens"] = overrides["reflection"]

        # Cap worker count by per-provider limit (avoids hitting RPM caps)
        provider_name = client.config.resolve_model(model_alias)[0]
        effective_workers = min(max_workers, PROVIDER_MAX_WORKERS.get(provider_name, max_workers))
        if effective_workers != max_workers:
            log.info(
                "[%s] provider=%s -> capping workers from %d to %d",
                model_alias, provider_name, max_workers, effective_workers,
            )

        try:
            s = run_one_model(
                client=client,
                experiment=experiment,
                model_alias=model_alias,
                personas=personas,
                events=events,
                output_dir=output_dir,
                max_workers=effective_workers,
                **kwargs,
            )
            summaries.append(s)
        except Exception as e:
            log.exception("Model %s failed: %s", model_alias, e)
            summaries.append(
                {"model_alias": model_alias, "error": str(e)}
            )

        # Persist index incrementally so a long run survives crashes
        index = {
            "experiment": experiment,
            "started_at": datetime.now().isoformat(),
            "personas": [p["id"] for p in personas],
            "events": events,
            "models": models,
            "summaries": summaries,
        }
        with open(output_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    # Final tracker summary
    try:
        client.print_summary()
    except Exception:  # pragma: no cover - cosmetic
        pass

    return {
        "experiment": experiment,
        "output_dir": str(output_dir),
        "summaries": summaries,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run V2 experiment across many models.")
    p.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Explicit list of model aliases (overrides --preset).",
    )
    p.add_argument(
        "--preset", type=str, default=None, choices=sorted(PRESETS.keys()),
        help=f"Use a pre-defined model list. Options: {sorted(PRESETS.keys())}",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/v2_multi_<timestamp>/).",
    )
    p.add_argument(
        "--pilot", action="store_true",
        help="Pilot: 10 Asian-Male personas x 3 events (matches V2 pilot).",
    )
    p.add_argument(
        "--events", type=str, nargs="+", default=None,
        help="Restrict to specific events (default: all 11).",
    )
    p.add_argument(
        "--max-workers", type=int, default=16,
        help="Parallel workers across all providers (default: 16). "
             "Per-provider concurrency is also capped by api_keys.yaml.",
    )
    p.add_argument(
        "--experiment", type=str, default=None,
        help="Override the experiment id (controls trace filename + call_id ns).",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Override api_keys.yaml path (default: configs/api_keys.yaml).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    _setup_logging(verbose=args.verbose)

    if args.models:
        models = list(args.models)
    elif args.preset:
        models = list(PRESETS[args.preset])
    else:
        log.error("Pass --models or --preset to choose which models to run.")
        return 2

    if args.pilot:
        personas = _pilot_personas()
        events = args.events or PILOT_EVENTS
        log.info("Pilot mode: 10 personas x %d events", len(events))
    else:
        personas = generate_all_personas()
        events = args.events or list(LIFE_EVENTS.keys())

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "results" / f"v2_multi_{ts}"

    config_path = Path(args.config) if args.config else None

    run_multi_model(
        models=models,
        personas=personas,
        events=events,
        output_dir=output_dir,
        experiment=args.experiment,
        max_workers=args.max_workers,
        config_path=config_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
