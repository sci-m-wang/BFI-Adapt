"""kappa_dcr_analysis.py — Compute pooled linearly-weighted Cohen's κ and
Directional Consistency Ratio (DCR) for every model in the curated multi-LLM
V2 experiment, then synthesise a composite benchmark for ranking
persona-event adaptation ability.

Definitions follow §5.3 of the GenPT paper (the perturbation-vs-baseline
pair definitions in Eq. 12 and Eq. 13). Mapping to our V2 framework:
    y^{(b)} = baseline BFI-44 item rating  (1..5)
    y^{(p)} = post-event BFI-44 item rating (same persona, same item)

Each (persona, event, item) yields one paired observation.

Two reporting layers:

L1 — pooled (rating reliability + raw rating bias)
  • κ_pool   : linear-weighted κ on ALL 48,400 pairs.
                Measures rating reproducibility across the full corpus,
                with no direction semantics.
  • DCR_pool : max(n↑, n↓) / (n↑ + n↓) on all pairs.
                Measures whether the model exhibits a global rating-up or
                rating-down bias when ratings move at all. It does NOT
                respect the per-(event, trait) expected direction.
  • Adapt_v1 : κ_pool · (2·DCR_pool − 1) · DC%   [legacy, retained for
                Appendix only as historical record].

L2 — event-trait specific (genuine adaptation)
  For each (event, trait) cell with a human prior ∈ {+, -}, compute
    κ_cell   : linear-weighted κ on the cell's 100·9=900 pairs.
                If all pairs are tied (no movement), κ_cell := 1.0
                (rating reliability is trivially perfect).
    DCR_cell : max(n↑, n↓) / (n↑ + n↓); 0.5 if no movement or balanced.
    S_cell   := 2·DCR_cell − 1 ∈ [0, 1].
    1_dir    := 1 if the cell's dominant direction matches the prior,
                else 0 (tie counts as 0).
  Aggregate over the 27 prior-cells:
    κ_avg    := mean κ_cell
    DCR_avg  := mean DCR_cell
    Adapt_v2 := mean (κ_cell · S_cell · 1_dir)
                — this is the primary benchmark.

Why two layers: DCR_pool conflates "correct shift" with "wrong shift"
because different (event, trait) cells have different expected directions.
Adapt_v2 honours those priors per cell; L1 is reported only for rating
reliability and as a raw bias diagnostic.

Outputs (under the chosen results dir):
  - kappa_dcr/per_model.csv         : both L1 and L2 columns
  - kappa_dcr/per_model.tex         : main paper table (L1 + L2)
  - kappa_dcr/per_model_legacy.tex  : appendix table (Adapt_v1 pooled)
  - kappa_dcr/per_cell.csv          : 11 models × 27 prior cells
  - kappa_dcr/kappa_dcr_plane.pdf   : (κ_avg, DCR_avg) scatter
  - kappa_dcr/per_event_dcr.csv     : event × model DCR matrix (legacy)
  - kappa_dcr/composite_ranking.csv : models sorted by Adapt_v2

Usage:
    python3 scripts/kappa_dcr_analysis.py \\
        --results-dirs results/v2_multi_curated_20260512_013721 \\
                       results/v2_multi_20260512_124836 \\
        --output-dir paper/figures/kappa_dcr \\
        --models gpt-5.3-chat gpt-4.1-mini claude-sonnet-4.6 claude-haiku-4.5 \\
                 deepseek-v4-pro doubao-seed-2.0-pro mimo-v2.5-pro \\
                 sf-qwen3-235b sf-glm-4.6 sf-kimi-k2-0905 sf-minimax-m2.5
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
BFI_PATH = REPO_ROOT / "BFI.json"
K_BFI = 5  # ordinal categories per item

TRAIT_ABBR = {
    "Extraversion": "E",
    "Agreeableness": "A",
    "Conscientiousness": "C",
    "Neuroticism": "N",
    "Openness": "O",
}
TRAITS_ORDER = ["O", "C", "E", "A", "N"]

# Same EXPECTED_CHANGES as scripts/full_experiment_v2.py:LIFE_EVENTS.
# Maps event -> {trait_abbr: "+" | "-" | "?"}.
EXPECTED_CHANGES: Dict[str, Dict[str, str]] = {
    "graduation":       {"C": "+", "N": "-", "E": "-"},
    "work_entry":       {"C": "+", "N": "-"},
    "job_change":       {"E": "?", "O": "?"},
    "promotion":        {"C": "+", "N": "-", "O": "+"},
    "unemployment":     {"C": "-", "O": "-", "A": "-"},
    "retirement":       {"C": "-"},
    "new_relationship": {"N": "-", "E": "+", "A": "-"},
    "marriage":         {"N": "-", "E": "-", "O": "-", "A": "-"},
    "child_birth":      {"O": "+", "A": "+"},
    "divorce":          {"E": "-", "C": "-"},
    "chronic_illness":  {"N": "+", "E": "-", "O": "-", "C": "-"},
}

# All 11 curated models (gemini-3-flash dropped for free-tier rate limits).
DEFAULT_MODELS: List[str] = [
    "gpt-5.3-chat", "gpt-4.1-mini",
    "claude-sonnet-4.6", "claude-haiku-4.5",
    "gemini-3-flash",
    "deepseek-v4-pro", "doubao-seed-2.0-pro",
    "mimo-v2.5-pro",
    "sf-qwen3-235b", "sf-glm-4.6", "sf-kimi-k2-0905",
]

# Pretty short labels for the figure.
SHORT_LABEL = {
    "gpt-5.3-chat": "GPT-5.3-chat",
    "gpt-4.1-mini": "GPT-4.1-mini",
    "claude-sonnet-4.6": "Claude-Sonnet-4.6",
    "claude-haiku-4.5": "Claude-Haiku-4.5",
    "gemini-3-flash": "Gemini-3-flash",
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "doubao-seed-2.0-pro": "Doubao-Seed-2.0-Pro",
    "mimo-v2.5-pro": "MiMo-V2.5-Pro",
    "sf-qwen3-235b": "Qwen3-235B",
    "sf-glm-4.6": "GLM-4.6",
    "sf-kimi-k2-0905": "Kimi-K2-0905",
}


# ---------------------------------------------------------------------------
# BFI item -> trait mapping (with reverse-keyed flag)
# ---------------------------------------------------------------------------

def load_bfi_meta(path: Path) -> Tuple[Dict[int, str], set]:
    """Returns (item_id -> trait_abbr, set_of_reverse_item_ids)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    item2trait: Dict[int, str] = {}
    for cat in data["categories"]:
        abbr = TRAIT_ABBR[cat["cat_name"]]
        for q in cat["cat_questions"]:
            item2trait[int(q)] = abbr
    reverse = {int(x) for x in data.get("reverse", [])}
    return item2trait, reverse


# ---------------------------------------------------------------------------
# Result loading: prefers the *latest* full_results.json among multiple dirs
# ---------------------------------------------------------------------------

def find_latest_full_results(model_alias: str, dirs: List[Path]) -> Optional[Path]:
    """Search multiple result dirs for ``<dir>/<model>/full_results.json`` and
    return the one with the most cells, breaking ties by most recent mtime.
    """
    candidates: List[Tuple[Path, int, float]] = []
    for d in dirs:
        p = d / model_alias / "full_results.json"
        if p.exists():
            try:
                with open(p) as f:
                    n_cells = len(json.load(f).get("results", []))
            except Exception:
                n_cells = 0
            candidates.append((p, n_cells, p.stat().st_mtime))
    if not candidates:
        return None
    # Prefer most cells, then most recent.
    candidates.sort(key=lambda t: (-t[1], -t[2]))
    return candidates[0][0]


def load_paired_items(path: Path) -> List[Dict]:
    """Yield one record per (persona, event) cell with paired item lists.

    Returns list of dicts with keys:
      persona_id, event, baseline_items {1..44 -> int}, post_items {1..44 -> int}
    Skips cells where either side has missing or non-integer items.
    """
    with open(path) as f:
        full = json.load(f)
    out: List[Dict] = []
    for cell in full.get("results", []):
        b = (cell.get("baseline") or {}).get("item_scores") or {}
        p = (cell.get("post_event") or {}).get("item_scores") or {}
        if not b or not p:
            continue
        bi: Dict[int, int] = {}
        pi: Dict[int, int] = {}
        for k in range(1, 45):
            kb = b.get(str(k))
            kp = p.get(str(k))
            if kb is None or kp is None:
                bi = pi = {}
                break
            try:
                kb_i, kp_i = int(kb), int(kp)
            except (TypeError, ValueError):
                bi = pi = {}
                break
            if not (1 <= kb_i <= 5 and 1 <= kp_i <= 5):
                bi = pi = {}
                break
            bi[k] = kb_i
            pi[k] = kp_i
        if not bi or not pi:
            continue
        out.append({
            "persona_id": cell.get("persona_id"),
            "event": cell.get("event"),
            "baseline_items": bi,
            "post_items": pi,
        })
    return out


# ---------------------------------------------------------------------------
# Cohen's κ — pooled linearly-weighted, K = 5 ordinal categories
# ---------------------------------------------------------------------------

def linear_weighted_kappa(pairs: List[Tuple[int, int]], K: int = K_BFI) -> float:
    """Pooled linearly-weighted Cohen's κ over a flat list of (b, p) pairs.

    Implements Eq. 12 from §5.3:
        κ = (sum w_ij p^o_ij - sum w_ij p^b_i p^p_j) /
            (1 - sum w_ij p^b_i p^p_j)
    with w_ij = 1 - |i-j|/(K-1). For K=2 this collapses to unweighted κ.
    """
    if not pairs:
        return float("nan")
    n = len(pairs)
    # Joint frequency
    joint = np.zeros((K, K), dtype=np.float64)
    for b, p in pairs:
        joint[b - 1, p - 1] += 1
    p_o = joint / n  # observed joint
    p_b = p_o.sum(axis=1)  # baseline marginal
    p_p = p_o.sum(axis=0)  # perturbed marginal
    # Linear weights
    idx = np.arange(K)
    W = 1 - np.abs(idx[:, None] - idx[None, :]) / (K - 1)
    expected = np.outer(p_b, p_p)
    num = float((W * p_o).sum() - (W * expected).sum())
    den = float(1.0 - (W * expected).sum())
    if den == 0:
        # Degenerate marginal (all pairs map to the same single category):
        # rater is trivially self-consistent → perfect agreement.
        observed = float((W * p_o).sum())
        return 1.0 if abs(observed - 1.0) < 1e-9 else float("nan")
    return num / den


# ---------------------------------------------------------------------------
# DCR — Directional Consistency Ratio (Eq. 13)
# ---------------------------------------------------------------------------

def dcr(pairs: List[Tuple[int, int]]) -> Tuple[float, str, int, int]:
    """Returns (dcr_value, direction, n_up, n_down).

    Direction is "up" if n_up > n_down, "down" otherwise, "tie" if equal.
    DCR = max(n_up, n_down) / (n_up + n_down) ∈ [0.5, 1.0].
    Items that don't change (n_eq) are excluded from the denominator,
    matching the paper's definition.
    """
    n_up = n_down = 0
    for b, p in pairs:
        if p > b:
            n_up += 1
        elif p < b:
            n_down += 1
    total = n_up + n_down
    if total == 0:
        return float("nan"), "tie", 0, 0
    val = max(n_up, n_down) / total
    if n_up > n_down:
        direction = "up"
    elif n_down > n_up:
        direction = "down"
    else:
        direction = "tie"
    return val, direction, n_up, n_down


# ---------------------------------------------------------------------------
# Per-trait paired pooling (handles reverse-keyed items)
# ---------------------------------------------------------------------------

def collect_pairs(
    cells: List[Dict],
    item2trait: Dict[int, str],
    reverse_set: set,
    *,
    by_trait: bool = False,
    by_event: bool = False,
    trait_filter: Optional[str] = None,
    event_filter: Optional[str] = None,
    use_reverse_corrected_for_direction: bool = True,
) -> List[Tuple[int, int]]:
    """Flatten paired (b, p) item ratings according to slicing options.

    When use_reverse_corrected_for_direction=True, *reverse* items are
    transformed via (6 - score) so that an *up* in the returned pair
    consistently means the underlying trait went up. This matters only
    for DCR direction; κ is unaffected (linear-weighted κ on a 1..5
    scale is symmetric under x -> K+1-x as long as the same transform
    is applied to both sides).
    """
    out: List[Tuple[int, int]] = []
    for cell in cells:
        if event_filter and cell["event"] != event_filter:
            continue
        for item_id, b_score in cell["baseline_items"].items():
            p_score = cell["post_items"][item_id]
            t = item2trait.get(item_id)
            if t is None:
                continue
            if trait_filter and t != trait_filter:
                continue
            if use_reverse_corrected_for_direction and item_id in reverse_set:
                b_score = (K_BFI + 1) - b_score
                p_score = (K_BFI + 1) - p_score
            out.append((b_score, p_score))
    return out


# ---------------------------------------------------------------------------
# Directional correctness against EXPECTED_CHANGES
# ---------------------------------------------------------------------------

def directional_correctness(
    cells: List[Dict],
    item2trait: Dict[int, str],
    reverse_set: set,
) -> Tuple[float, int, int]:
    """Compute fraction of (event, trait) cells where the dominant DCR
    direction matches the human prior. Uncertain ("?") and unspecified
    (event_info has no entry for that trait) cells are skipped.

    Returns (correct / total, correct, total).
    """
    correct = 0
    total = 0
    events = sorted({c["event"] for c in cells})
    for event in events:
        prior = EXPECTED_CHANGES.get(event, {})
        for trait, sign in prior.items():
            if sign not in {"+", "-"}:
                continue
            pairs = collect_pairs(
                cells, item2trait, reverse_set,
                trait_filter=trait, event_filter=event,
                use_reverse_corrected_for_direction=True,
            )
            _, direction, n_up, n_down = dcr(pairs)
            if direction == "tie":
                continue
            total += 1
            if (sign == "+" and direction == "up") or (
                sign == "-" and direction == "down"
            ):
                correct += 1
    if total == 0:
        return float("nan"), 0, 0
    return correct / total, correct, total


# ---------------------------------------------------------------------------
# Per-model summary
# ---------------------------------------------------------------------------

def summarise_model(
    cells: List[Dict],
    item2trait: Dict[int, str],
    reverse_set: set,
) -> Dict:
    """Compute both L1 (pooled) and L2 (event-trait cell-level) summaries.

    L1: rating reliability and raw rating bias over all pairs.
    L2: adaptation respecting per-(event, trait) human priors.
    """
    if not cells:
        return {"n_cells": 0, "n_pairs": 0}

    # -----------------------------------------------------------------
    # L1 — pooled (legacy, for rating reliability and Adapt_v1 only)
    # -----------------------------------------------------------------
    all_pairs = collect_pairs(
        cells, item2trait, reverse_set,
        use_reverse_corrected_for_direction=True,
    )
    kappa_pool = linear_weighted_kappa(all_pairs)
    dcr_pool, direction_pool, n_up, n_down = dcr(all_pairs)

    per_trait: Dict[str, Dict] = {}
    for t in TRAITS_ORDER:
        pairs_t = collect_pairs(
            cells, item2trait, reverse_set,
            trait_filter=t,
            use_reverse_corrected_for_direction=True,
        )
        v, dirn, nu, nd = dcr(pairs_t)
        per_trait[t] = {
            "dcr": v, "direction": dirn,
            "n_up": nu, "n_down": nd,
        }

    dc_correct, dc_n, dc_total = directional_correctness(
        cells, item2trait, reverse_set
    )

    # Adapt_v1 (legacy pooled composite) is deferred to after the L2 pass so
    # it can re-use the cell-level DC% (dc_correct_l2). This keeps the legacy
    # vs. main comparison clean: both composites share the same DC% column,
    # so any ranking divergence is attributable purely to the κ / DCR
    # aggregation choice (pooled vs. cell-mean), not to a different DC%
    # denominator. See write_per_model_legacy_tex caption.

    # -----------------------------------------------------------------
    # L2 — per (event, trait) cell, averaged over prior-cells only
    # -----------------------------------------------------------------
    cell_records: List[Dict] = []
    for event in sorted(EXPECTED_CHANGES.keys()):
        prior = EXPECTED_CHANGES[event]
        for trait, sign in prior.items():
            if sign not in {"+", "-"}:
                # Skip uncertain ("?") cells: no ground truth.
                continue
            pairs_c = collect_pairs(
                cells, item2trait, reverse_set,
                trait_filter=trait, event_filter=event,
                use_reverse_corrected_for_direction=True,
            )
            if not pairs_c:
                continue
            # κ_cell: rating reliability within this cell.
            #   All-tied → κ=1.0 (rater self-consistent on a degenerate margin).
            #   Otherwise the standard linear-weighted κ formula.
            kappa_c = linear_weighted_kappa(pairs_c)
            if np.isnan(kappa_c):
                # All pairs map to a single category (b == p for all):
                # treat as perfect agreement.
                if all(b == p for b, p in pairs_c):
                    kappa_c = 1.0
            # DCR_cell: directional asymmetry. 0.5 when no movement.
            dcr_val, direction_c, nu_c, nd_c = dcr(pairs_c)
            if np.isnan(dcr_val):
                dcr_val = 0.5
                direction_c = "tie"
            S = 2 * dcr_val - 1
            # 1_dir: 1 iff dominant direction matches the prior (tie → 0).
            if direction_c == "up" and sign == "+":
                dir_match = 1
            elif direction_c == "down" and sign == "-":
                dir_match = 1
            else:
                dir_match = 0
            adapt_contrib = max(0.0, kappa_c) * S * dir_match
            cell_records.append({
                "event": event,
                "trait": trait,
                "prior": sign,
                "n_pairs": len(pairs_c),
                "n_up": nu_c, "n_down": nd_c,
                "kappa": kappa_c,
                "dcr": dcr_val,
                "direction": direction_c,
                "S": S,
                "dir_match": dir_match,
                "adapt_contrib": adapt_contrib,
            })

    if cell_records:
        kappa_avg = float(np.mean([c["kappa"] for c in cell_records]))
        dcr_avg = float(np.mean([c["dcr"] for c in cell_records]))
        adapt_v2 = float(np.mean([c["adapt_contrib"] for c in cell_records]))
        dc_correct_l2 = float(np.mean([c["dir_match"] for c in cell_records]))
        n_prior_cells = len(cell_records)
    else:
        kappa_avg = dcr_avg = adapt_v2 = dc_correct_l2 = float("nan")
        n_prior_cells = 0

    # Adapt_v1 (legacy pooled composite, Appendix only). Uses pooled κ and
    # pooled DCR but the SAME cell-level DC% (dc_correct_l2) as the main
    # BFI-Adapt, so the legacy↔main comparison isolates the κ / DCR
    # aggregation choice (pooled vs. cell-mean) without confounding it with
    # a different DC% denominator (the legacy DC% from
    # directional_correctness drops tie-direction cells, the main one does
    # not; see write_per_model_legacy_tex caption).
    if (
        not np.isnan(kappa_pool)
        and not np.isnan(dcr_pool)
        and not np.isnan(dc_correct_l2)
    ):
        adapt_v1 = max(0.0, kappa_pool) * (2 * dcr_pool - 1) * dc_correct_l2
    else:
        adapt_v1 = float("nan")

    return {
        "n_cells": len(cells),
        "n_pairs": len(all_pairs),
        # L1 (pooled)
        "kappa": kappa_pool,
        "dcr_pooled": dcr_pool,
        "direction_pooled": direction_pool,
        "n_up_pooled": n_up,
        "n_down_pooled": n_down,
        "per_trait_dcr": per_trait,
        "dc_correct": dc_correct,
        "dc_correct_n": dc_n,
        "dc_correct_total": dc_total,
        "composite": adapt_v1,          # legacy Adapt_v1
        # L2 (event-trait cell)
        "n_prior_cells": n_prior_cells,
        "kappa_avg": kappa_avg,
        "dcr_avg": dcr_avg,
        "dc_correct_l2": dc_correct_l2,
        "adapt_v2": adapt_v2,
        "cell_records": cell_records,
    }


# ---------------------------------------------------------------------------
# Per-event DCR matrix
# ---------------------------------------------------------------------------

def per_event_dcr_matrix(
    cells: List[Dict],
    item2trait: Dict[int, str],
    reverse_set: set,
) -> Dict[str, Dict[str, float]]:
    """For each event, compute DCR per trait. Returns event -> trait -> DCR."""
    out: Dict[str, Dict[str, float]] = {}
    for event in sorted(EXPECTED_CHANGES.keys()):
        out[event] = {}
        for t in TRAITS_ORDER:
            pairs = collect_pairs(
                cells, item2trait, reverse_set,
                trait_filter=t, event_filter=event,
                use_reverse_corrected_for_direction=True,
            )
            v, _, _, _ = dcr(pairs)
            out[event][t] = v
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_per_model_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "n_cells", "n_pairs",
        # L1 pooled
        "kappa", "dcr_pooled", "direction_pooled",
        "n_up_pooled", "n_down_pooled",
        "dcr_O", "dcr_C", "dcr_E", "dcr_A", "dcr_N",
        "dir_O", "dir_C", "dir_E", "dir_A", "dir_N",
        "dc_correct", "dc_correct_n", "dc_correct_total",
        "composite",
        # L2 event-trait
        "n_prior_cells",
        "kappa_avg", "dcr_avg", "dc_correct_l2", "adapt_v2",
        "status",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_per_cell_csv(per_cell_records: Dict[str, List[Dict]], path: Path) -> None:
    """Long-form per-cell table: model × (event, trait) prior cells."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "event", "trait", "prior", "n_pairs",
        "n_up", "n_down", "kappa", "dcr", "direction",
        "S", "dir_match", "adapt_contrib",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for model, records in per_cell_records.items():
            for c in records:
                row = {"model": model}
                row.update({k: c[k] for k in fields if k != "model"})
                w.writerow(row)


def fmt_num(x, prec: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{prec}f}"


def write_per_model_tex(rows: List[Dict], path: Path) -> None:
    """Main paper table: pair-level reporting only.

    Columns (all averaged over the 27 prior cells):
        κ_cell, DCR_cell, DC%_cell

    BFI-Adapt is presented separately in Figure~\\ref{fig:bfi_adapt_ranking}
    to avoid duplicating the same composite in two places.

    Models are sorted by Adapt (BFI-Adapt) descending so that the table row
    order mirrors the figure; pending models go last. Pooled diagnostic
    columns live in the legacy Appendix table.
    """
    valid = [r for r in rows if r.get("status") == "ok"
             and not np.isnan(r.get("adapt_v2", float("nan")))]
    pending = [r for r in rows if r not in valid]
    valid.sort(key=lambda r: r["adapt_v2"], reverse=True)
    ordered = valid + pending

    lines = []
    lines.append(r"% Auto-generated by scripts/kappa_dcr_analysis.py — DO NOT EDIT")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\caption{Pair-level reliability and direction indicators "
                 r"for the 11-model PC-Agent benchmark, computed over the 27 "
                 r"event--trait pairs with a definite "
                 r"expected human direction. $\bar{\kappa}_{\text{pair}}$: mean within-pair "
                 r"rating reliability; $\overline{\mathrm{DCR}}_{\text{pair}}$: "
                 r"mean within-pair directional consistency; "
                 r"$\mathrm{DC}\%_{\text{pair}}$: fraction of pairs whose "
                 r"dominant DCR direction matches the expected human direction. Models are "
                 r"ordered by the pair-level composite \textsc{BFI-Adapt} "
                 r"(Eq.~\ref{eq:bfiadapt}; Figure~\ref{fig:bfi_adapt_ranking}). "
                 r"Rows marked --- could not be evaluated.}")
    lines.append(r"\label{tab:kappa_dcr_per_model}")
    lines.append(r"\begin{tabular*}{\columnwidth}{l@{\extracolsep{\fill}}rrr}")
    lines.append(r"\toprule")
    lines.append(r"Model "
                 r"& $\bar{\kappa}_{\text{pair}}$ "
                 r"& $\overline{\mathrm{DCR}}_{\text{pair}}$ "
                 r"& $\mathrm{DC}\%_{\text{pair}}$ \\")
    lines.append(r"\midrule")
    for r in ordered:
        if r.get("status") != "ok":
            lines.append(
                f"{SHORT_LABEL.get(r['model'], r['model'])} & "
                f"— & — & — \\\\"
            )
            continue
        dc_l2 = (r.get('dc_correct_l2') or 0) * 100 if r.get('dc_correct_l2') is not None else None
        lines.append(
            f"{SHORT_LABEL.get(r['model'], r['model'])} & "
            f"{fmt_num(r['kappa_avg'])} & "
            f"{fmt_num(r['dcr_avg'])} & "
            f"{fmt_num(dc_l2, 1)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_per_model_legacy_tex(rows: List[Dict], path: Path) -> None:
    """Appendix table: legacy L1 Adapt_v1 (= κ_pool·(2·DCR_pool−1)·DC%_pool).

    Retained as a historical record and to expose the pooling artefact:
    Adapt_v1 conflates event-trait priors that point in opposite directions.
    """
    valid = [r for r in rows if r.get("status") == "ok"
             and not np.isnan(r.get("composite", float("nan")))]
    pending = [r for r in rows if r not in valid]
    valid.sort(key=lambda r: r["composite"], reverse=True)
    ordered = valid + pending

    lines = []
    lines.append(r"% Auto-generated by scripts/kappa_dcr_analysis.py — DO NOT EDIT")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\caption{Legacy pooled formulation "
                 r"$\mathrm{Adapt}_{\text{pool}}{=}\max(0,\kappa_{\text{pool}})"
                 r"(2\,\mathrm{DCR}_{\text{pool}}{-}1)\,\mathrm{DC}\%_{\text{pair}}$, "
                 r"retained for comparison with the pair-level "
                 r"\textsc{BFI-Adapt} (main Table~\ref{tab:kappa_dcr_per_model}). "
                 r"The two composites share the same $\mathrm{DC}\%_{\text{pair}}$ column, "
                 r"so cross-model rank divergence is attributable purely to the "
                 r"$\kappa$ / DCR aggregation choice (pooled vs.\ pair-mean), "
                 r"not to a different DC denominator. Pooling conflates "
                 r"event-trait priors that point in opposite directions and "
                 r"reorders 9 of 11 models by at least two ranks (6 of 11 by "
                 r"more than two).}")
    lines.append(r"\label{tab:kappa_dcr_legacy}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & $\kappa_{\text{pool}}$ & $\mathrm{DCR}_{\text{pool}}$ "
                 r"& $\mathrm{DC}\%_{\text{pair}}$ & $\mathrm{Adapt}_{\text{pool}}$ \\")
    lines.append(r"\midrule")
    for r in ordered:
        if r.get("status") != "ok":
            lines.append(
                f"{SHORT_LABEL.get(r['model'], r['model'])} & "
                f"— & — & — & — \\\\"
            )
            continue
        dc = (r.get('dc_correct_l2') or 0) * 100 if r.get('dc_correct_l2') is not None else None
        lines.append(
            f"{SHORT_LABEL.get(r['model'], r['model'])} & "
            f"{fmt_num(r['kappa'])} & "
            f"{fmt_num(r['dcr_pooled'])} & "
            f"{fmt_num(dc, 1)} & "
            f"{fmt_num(r['composite'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_per_event_csv(matrices: Dict[str, Dict[str, Dict[str, float]]], path: Path) -> None:
    """matrices: model -> event -> trait -> DCR"""
    path.parent.mkdir(parents=True, exist_ok=True)
    events = sorted(EXPECTED_CHANGES.keys())
    cols = ["model", "event"] + [f"DCR_{t}" for t in TRAITS_ORDER]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for model, mat in matrices.items():
            for event in events:
                row = {"model": model, "event": event}
                for t in TRAITS_ORDER:
                    v = mat.get(event, {}).get(t)
                    row[f"DCR_{t}"] = "" if (v is None or np.isnan(v)) else f"{v:.4f}"
                w.writerow(row)


# ---------------------------------------------------------------------------
# Plot: (κ, DCR) plane
# ---------------------------------------------------------------------------

def plot_kappa_dcr_plane(rows: List[Dict], path: Path) -> None:
    """(κ_pair, DCR_pair) plane — L2 averages over 27 prior pairs."""
    fig, ax = plt.subplots(figsize=(9.0, 6.5), dpi=140)
    ax.set_xlabel(r"$\bar{\kappa}_{\text{pair}}$  "
                  r"(rating reliability, mean over 27 prior pairs)")
    ax.set_ylabel(r"$\overline{\mathrm{DCR}}_{\text{pair}}$  "
                  r"(systematic drift, mean over 27 prior pairs)")
    valid_rows = [r for r in rows if r.get("status") == "ok"
                  and not np.isnan(r.get("kappa_avg", float("nan")))
                  and not np.isnan(r.get("dcr_avg", float("nan")))]
    if valid_rows:
        xs = [r["kappa_avg"] for r in valid_rows]
        ys = [r["dcr_avg"] for r in valid_rows]
        x_pad = max(0.02, (max(xs) - min(xs)) * 0.15)
        y_pad = max(0.01, (max(ys) - min(ys)) * 0.15)
        ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
        ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
    ax.grid(alpha=0.25, linestyle="--")

    # Reference quadrant lines (loose thresholds)
    ax.axhline(0.55, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0.85, color="gray", lw=0.5, alpha=0.5)

    # Quadrant text in *axes* coords, positioned away from data clusters
    ax.text(0.02, 0.97, "low-$\\kappa$, low-DCR\n(idiosyncratic noise)",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color="gray", style="italic")
    ax.text(0.98, 0.97, "high-$\\kappa$, high-DCR\n(systematic adaptation)",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=8, color="gray", style="italic")
    ax.text(0.02, 0.03, "low-$\\kappa$, DCR$\\!\\approx\\!0.5$\n(unstable, balanced)",
            transform=ax.transAxes, va="bottom", ha="left",
            fontsize=8, color="gray", style="italic")
    ax.text(0.98, 0.03, "high-$\\kappa$, DCR$\\!\\approx\\!0.5$\n(stable, idiosyncratic)",
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=8, color="gray", style="italic")

    cmap = plt.get_cmap("tab10")
    # Use default label offset for the new (κ_avg, DCR_avg) coordinates;
    # per-model nudges can be re-tuned once the rebuilt scatter is inspected.
    LABEL_OFFSETS: Dict[str, Tuple[float, float, str, str]] = {}

    pending_models: List[str] = []
    pending_colors: List = []
    for i, r in enumerate(rows):
        color = cmap(i % 10)
        label = SHORT_LABEL.get(r["model"], r["model"])
        kx = r.get("kappa_avg", float("nan"))
        ky = r.get("dcr_avg", float("nan"))
        if r["status"] == "ok" and not (np.isnan(kx) or np.isnan(ky)):
            ax.scatter(kx, ky, s=130, color=color,
                       edgecolors="black", linewidths=0.8, zorder=3,
                       label=label)
            dx, dy, ha, va = LABEL_OFFSETS.get(
                r["model"], (0.005, 0.006, "left", "bottom")
            )
            ax.annotate(
                label,
                (kx + dx, ky + dy),
                fontsize=8.5, color="black",
                ha=ha, va=va,
            )
        else:
            pending_models.append(label)
            pending_colors.append(color)

    # Pending models in a small box at the top-right of the data area
    if pending_models:
        pend_text = "Pending (run incomplete):\n" + "\n".join(
            f"  • {m}" for m in pending_models
        )
        ax.text(
            0.99, 0.79, pend_text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8.5, color="dimgray",
            bbox=dict(facecolor="white", edgecolor="dimgray", boxstyle="round,pad=0.4", alpha=0.85),
        )

    ax.set_title("Persona-Event Adaptation (L2): Cell-level Stability vs Systematic Drift",
                 fontsize=11.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_composite_ranking(rows: List[Dict], path: Path) -> None:
    """Vertical bar chart of \textsc{BFI-Adapt}$_{\text{v2}}$ (cell-mean) scores."""
    valid = [r for r in rows if r["status"] == "ok"
             and not np.isnan(r.get("adapt_v2", float("nan")))]
    invalid = [r for r in rows if r not in valid]
    valid.sort(key=lambda r: r["adapt_v2"], reverse=True)

    n = len(rows)
    fig, ax = plt.subplots(figsize=(0.32 * n + 0.6, 3.2), dpi=140)
    labels = []
    values = []
    colors = []
    cmap = plt.get_cmap("viridis")
    max_score = max((r["adapt_v2"] for r in valid), default=1.0) or 1.0

    for r in valid:
        labels.append(SHORT_LABEL.get(r["model"], r["model"]))
        values.append(r["adapt_v2"])
        colors.append(cmap(r["adapt_v2"] / max_score * 0.9))
    for r in invalid:
        labels.append(SHORT_LABEL.get(r["model"], r["model"]) + "  (pending)")
        values.append(0.0)
        colors.append("lightgray")

    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=40, ha="right")
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("BFI-Adapt", fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    y_pad = max_score * 0.012
    for xi, v in zip(x, values):
        if v > 0:
            ax.text(xi, v + y_pad, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5)
        else:
            ax.text(xi, y_pad, "—",
                    ha="center", va="bottom", fontsize=7, color="gray")
    ax.set_ylim(0, max_score * 1.12)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dirs", type=Path, nargs="+", required=True,
        help="One or more results dirs (later ones override earlier when "
             "they contain more cells for a given model)."
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS)
    args = ap.parse_args()

    item2trait, reverse_set = load_bfi_meta(BFI_PATH)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    per_event: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_cell_records: Dict[str, List[Dict]] = {}
    for model in args.models:
        path = find_latest_full_results(model, args.results_dirs)
        if path is None:
            print(f"  [pending] {model}: no results found in any --results-dir")
            rows.append({
                "model": model, "status": "pending",
                "n_cells": 0, "n_pairs": 0,
                "kappa": float("nan"), "dcr_pooled": float("nan"),
                "direction_pooled": "—",
                "n_up_pooled": 0, "n_down_pooled": 0,
                "dcr_O": float("nan"), "dcr_C": float("nan"),
                "dcr_E": float("nan"), "dcr_A": float("nan"),
                "dcr_N": float("nan"),
                "dir_O": "—", "dir_C": "—", "dir_E": "—",
                "dir_A": "—", "dir_N": "—",
                "dc_correct": float("nan"),
                "dc_correct_n": 0, "dc_correct_total": 0,
                "composite": float("nan"),
                "n_prior_cells": 0,
                "kappa_avg": float("nan"),
                "dcr_avg": float("nan"),
                "dc_correct_l2": float("nan"),
                "adapt_v2": float("nan"),
            })
            continue

        cells = load_paired_items(path)
        s = summarise_model(cells, item2trait, reverse_set)
        per_event[model] = per_event_dcr_matrix(cells, item2trait, reverse_set)
        per_cell_records[model] = s.get("cell_records", [])
        row = {
            "model": model,
            "status": "ok" if s["n_cells"] > 0 else "pending",
            "n_cells": s["n_cells"],
            "n_pairs": s["n_pairs"],
            # L1 pooled
            "kappa": s.get("kappa", float("nan")),
            "dcr_pooled": s.get("dcr_pooled", float("nan")),
            "direction_pooled": s.get("direction_pooled", "—"),
            "n_up_pooled": s.get("n_up_pooled", 0),
            "n_down_pooled": s.get("n_down_pooled", 0),
            "dc_correct": s.get("dc_correct", float("nan")),
            "dc_correct_n": s.get("dc_correct_n", 0),
            "dc_correct_total": s.get("dc_correct_total", 0),
            "composite": s.get("composite", float("nan")),
            # L2 event-trait
            "n_prior_cells": s.get("n_prior_cells", 0),
            "kappa_avg": s.get("kappa_avg", float("nan")),
            "dcr_avg": s.get("dcr_avg", float("nan")),
            "dc_correct_l2": s.get("dc_correct_l2", float("nan")),
            "adapt_v2": s.get("adapt_v2", float("nan")),
        }
        per_t = s.get("per_trait_dcr", {})
        for t in TRAITS_ORDER:
            tt = per_t.get(t, {})
            row[f"dcr_{t}"] = tt.get("dcr", float("nan"))
            row[f"dir_{t}"] = tt.get("direction", "—")
        rows.append(row)
        print(
            f"  [ok] {model:<22} cells={s['n_cells']:>4}  "
            f"L1: κ={fmt_num(s.get('kappa'))} DCR={fmt_num(s.get('dcr_pooled'))} "
            f"DC%={fmt_num((s.get('dc_correct') or 0)*100, 1)} "
            f"Adapt_v1={fmt_num(s.get('composite'))} | "
            f"L2: κ̄={fmt_num(s.get('kappa_avg'))} "
            f"DCR̄={fmt_num(s.get('dcr_avg'))} "
            f"DC%={fmt_num((s.get('dc_correct_l2') or 0)*100, 1)} "
            f"Adapt_v2={fmt_num(s.get('adapt_v2'))}"
        )

    write_per_model_csv(rows, args.output_dir / "per_model.csv")
    write_per_model_tex(rows, args.output_dir / "per_model.tex")
    write_per_model_legacy_tex(rows, args.output_dir / "per_model_legacy.tex")
    write_per_cell_csv(per_cell_records, args.output_dir / "per_cell.csv")
    write_per_event_csv(per_event, args.output_dir / "per_event_dcr.csv")
    plot_kappa_dcr_plane(rows, args.output_dir / "kappa_dcr_plane.png")
    plot_composite_ranking(rows, args.output_dir / "composite_ranking.png")

    # Adapt_v2 ranking CSV (sorted by L2 cell-mean Adapt).
    valid = [r for r in rows if r["status"] == "ok"
             and not np.isnan(r.get("adapt_v2", float("nan")))]
    invalid = [r for r in rows if r not in valid]
    valid.sort(key=lambda r: r["adapt_v2"], reverse=True)
    ranked = valid + invalid
    with open(args.output_dir / "composite_ranking.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "model",
            "adapt_v2", "kappa_avg", "dcr_avg", "dc_correct_l2",
            "adapt_v1_legacy", "kappa_pool", "dcr_pool", "dc_correct_pool",
            "status",
        ])
        for i, r in enumerate(ranked, 1):
            w.writerow([
                i if r["status"] == "ok" else "—",
                r["model"],
                fmt_num(r.get("adapt_v2", float("nan"))),
                fmt_num(r.get("kappa_avg", float("nan"))),
                fmt_num(r.get("dcr_avg", float("nan"))),
                fmt_num((r.get("dc_correct_l2") or 0) * 100, 1)
                if r.get("dc_correct_l2") is not None else "—",
                fmt_num(r.get("composite", float("nan"))),
                fmt_num(r.get("kappa", float("nan"))),
                fmt_num(r.get("dcr_pooled", float("nan"))),
                fmt_num((r.get("dc_correct") or 0) * 100, 1)
                if r.get("dc_correct") is not None else "—",
                r["status"],
            ])

    print(f"\nOutputs written to {args.output_dir}/")
    print("  per_model.csv, per_model.tex, per_model_legacy.tex")
    print("  per_cell.csv, per_event_dcr.csv")
    print("  kappa_dcr_plane.{png,pdf}")
    print("  composite_ranking.{png,pdf,csv}")


if __name__ == "__main__":
    main()
