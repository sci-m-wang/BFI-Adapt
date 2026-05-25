"""
Cross-model match-rate analysis for V2 multi-model experiments.

Builds a Model x Event match-rate heatmap (one cell per (model, event) using
the per-trait expected directions from full_experiment_v2.LIFE_EVENTS) plus a
companion heatmap aggregated to (model x trait).

Match-rate convention follows paper Sec 3.5:
    For each persona p, event e, trait t with an expected direction in
    {'+', '-'} (cells with '?' or no expectation are excluded), classify
        d_{p,e,t} = +  if change > +eps
                    -  if change < -eps
                    0  otherwise
    P_match(e, t) = |{ p : d_{p,e,t} == expected }| / |{ p : d_{p,e,t} != 0 }|

For the (model, event) cell we average P_match across all expected (e, t)
pairs of that event; for the (model, trait) cell we average across all
expected (e, t) pairs whose trait equals t. We also report a global match-rate
per model and a sample-count per cell so cells with thin data can be flagged.

Outputs:
    paper/figures/cross_model/per_model_event_match.csv
    paper/figures/cross_model/per_model_trait_match.csv
    paper/figures/cross_model/per_model_overall.csv
    paper/figures/cross_model/heatmap_event.{pdf,png}
    paper/figures/cross_model/heatmap_trait.{pdf,png}

Usage:
    python scripts/compare_models_v2.py \
        --results-dirs results/v2_multi_curated_20260512_013721 \
                       results/v2_multi_20260512_124836 \
        --models gpt-5.3-chat gpt-4.1-mini ... \
        --out-dir paper/figures/cross_model
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Re-use canonical helpers from kappa_dcr_analysis to stay in sync with the
# paper's curated 11-model preset, expected-changes table, and label scheme.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kappa_dcr_analysis import (  # type: ignore  # noqa: E402
    DEFAULT_MODELS,
    EXPECTED_CHANGES,
    SHORT_LABEL,
    TRAIT_ABBR,
    find_latest_full_results,
)

# Trait abbreviation -> full BFI trait name as it appears in the change dict.
TRAIT_FULL = {abbr: full for full, abbr in TRAIT_ABBR.items()}
TRAIT_ORDER = ["O", "C", "E", "A", "N"]
TRAIT_LONG = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}

# Event order matches LIFE_EVENTS in full_experiment_v2 grouped by domain.
EVENT_ORDER = [
    "graduation", "work_entry", "job_change", "promotion", "unemployment", "retirement",
    "new_relationship", "marriage", "divorce", "child_birth",
    "chronic_illness",
]
EVENT_DOMAIN = {
    "graduation": "Occupational", "work_entry": "Occupational", "job_change": "Occupational",
    "promotion": "Occupational", "unemployment": "Occupational", "retirement": "Occupational",
    "new_relationship": "Social", "marriage": "Social", "divorce": "Social", "child_birth": "Social",
    "chronic_illness": "Health",
}

EPSILON = 0.1  # noise threshold from paper Sec 3.5


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_change_records(path: Path) -> List[Dict]:
    """Return a list of dicts {persona_id, event, change: {trait_full -> float}}.

    Skips cells with missing or empty change dicts.
    """
    with open(path) as f:
        full = json.load(f)
    out: List[Dict] = []
    for cell in full.get("results", []):
        change = cell.get("change") or {}
        if not change:
            continue
        if not all(t in change for t in TRAIT_LONG.values()):
            continue
        out.append({
            "persona_id": cell.get("persona_id"),
            "event": cell.get("event"),
            "change": {t: float(change[t]) for t in TRAIT_LONG.values()},
        })
    return out


# ---------------------------------------------------------------------------
# Match-rate computation
# ---------------------------------------------------------------------------

def classify_direction(delta: float, eps: float = EPSILON) -> int:
    """+1, -1, or 0 (neutral / excluded)."""
    if delta > eps:
        return +1
    if delta < -eps:
        return -1
    return 0


def cell_match_rate(records: List[Dict], event: str, trait_abbr: str,
                    expected: str, eps: float = EPSILON) -> Tuple[Optional[float], int]:
    """P_match for one (event, trait) cell across all personas of one model.

    Returns (match_rate in [0,1] or None, n_active where n_active is the
    number of personas with non-neutral direction).
    """
    if expected not in {"+", "-"}:
        return None, 0
    trait_full = TRAIT_FULL[trait_abbr]
    expected_dir = +1 if expected == "+" else -1
    n_match = 0
    n_active = 0
    for rec in records:
        if rec["event"] != event:
            continue
        delta = rec["change"].get(trait_full)
        if delta is None:
            continue
        d = classify_direction(delta, eps)
        if d == 0:
            continue
        n_active += 1
        if d == expected_dir:
            n_match += 1
    if n_active == 0:
        return None, 0
    return n_match / n_active, n_active


def model_event_match(records: List[Dict], event: str, eps: float = EPSILON
                      ) -> Tuple[Optional[float], int]:
    """Average P_match across all expected traits of one event for one model.

    Returns (mean P_match in [0,1] or None if no expected traits / no data,
    total n_active aggregated across the included trait cells).
    """
    expected = EXPECTED_CHANGES.get(event, {})
    rates: List[float] = []
    n_active_total = 0
    for trait_abbr, exp_dir in expected.items():
        if exp_dir not in {"+", "-"}:
            continue
        rate, n = cell_match_rate(records, event, trait_abbr, exp_dir, eps)
        if rate is not None:
            rates.append(rate)
            n_active_total += n
    if not rates:
        return None, 0
    return float(np.mean(rates)), n_active_total


def model_trait_match(records: List[Dict], trait_abbr: str, eps: float = EPSILON
                      ) -> Tuple[Optional[float], int]:
    """Average P_match across all events expecting a direction for trait_abbr."""
    rates: List[float] = []
    n_active_total = 0
    for event, expected in EXPECTED_CHANGES.items():
        exp_dir = expected.get(trait_abbr)
        if exp_dir not in {"+", "-"}:
            continue
        rate, n = cell_match_rate(records, event, trait_abbr, exp_dir, eps)
        if rate is not None:
            rates.append(rate)
            n_active_total += n
    if not rates:
        return None, 0
    return float(np.mean(rates)), n_active_total


def model_overall_match(records: List[Dict], eps: float = EPSILON
                        ) -> Tuple[Optional[float], int]:
    """Pooled P_match across all expected (event, trait) cells."""
    n_match_total = 0
    n_active_total = 0
    for event, expected in EXPECTED_CHANGES.items():
        for trait_abbr, exp_dir in expected.items():
            if exp_dir not in {"+", "-"}:
                continue
            trait_full = TRAIT_FULL[trait_abbr]
            expected_dir = +1 if exp_dir == "+" else -1
            for rec in records:
                if rec["event"] != event:
                    continue
                delta = rec["change"].get(trait_full)
                if delta is None:
                    continue
                d = classify_direction(delta, eps)
                if d == 0:
                    continue
                n_active_total += 1
                if d == expected_dir:
                    n_match_total += 1
    if n_active_total == 0:
        return None, 0
    return n_match_total / n_active_total, n_active_total


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_per_model_event_csv(matrix: Dict[str, Dict[str, Tuple[Optional[float], int]]],
                              path: Path) -> None:
    """Rows = models, cols = events."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + EVENT_ORDER + ["overall"])
        for model in matrix:
            row = [model]
            for ev in EVENT_ORDER:
                rate, _ = matrix[model].get(ev, (None, 0))
                row.append(f"{rate*100:.1f}" if rate is not None else "")
            ov, _ = matrix[model].get("__overall__", (None, 0))
            row.append(f"{ov*100:.1f}" if ov is not None else "")
            w.writerow(row)


def write_per_model_trait_csv(matrix: Dict[str, Dict[str, Tuple[Optional[float], int]]],
                              path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + [TRAIT_LONG[t] for t in TRAIT_ORDER] + ["overall"])
        for model in matrix:
            row = [model]
            for t in TRAIT_ORDER:
                rate, _ = matrix[model].get(t, (None, 0))
                row.append(f"{rate*100:.1f}" if rate is not None else "")
            ov, _ = matrix[model].get("__overall__", (None, 0))
            row.append(f"{ov*100:.1f}" if ov is not None else "")
            w.writerow(row)


def write_overall_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "match_rate_pct", "n_active_cells"])
        for r in rows:
            rate = r.get("overall_rate")
            n = r.get("overall_n", 0)
            w.writerow([
                r["model"],
                f"{rate*100:.1f}" if rate is not None else "",
                n,
            ])


# ---------------------------------------------------------------------------
# Heatmap plotting
# ---------------------------------------------------------------------------

def _diverging_cmap() -> LinearSegmentedColormap:
    """Red below 50% (worse than chance), white at 50%, green above."""
    return LinearSegmentedColormap.from_list(
        "match_div", ["#b2182b", "#f7f7f7", "#1a9850"], N=256,
    )


def plot_event_heatmap(matrix: Dict[str, Dict[str, Tuple[Optional[float], int]]],
                       models: List[str], path: Path) -> None:
    """Rows = models (in passed order, with completed models on top),
    cols = events grouped by domain."""
    n_rows = len(models)
    n_cols = len(EVENT_ORDER)
    # Build grid as percentages; missing -> NaN.
    grid = np.full((n_rows, n_cols), np.nan)
    counts = np.zeros((n_rows, n_cols), dtype=int)
    for i, model in enumerate(models):
        for j, ev in enumerate(EVENT_ORDER):
            rate, n = matrix.get(model, {}).get(ev, (None, 0))
            if rate is not None:
                grid[i, j] = rate * 100
                counts[i, j] = n

    fig, ax = plt.subplots(figsize=(11.5, 0.55 * n_rows + 1.6))
    cmap = _diverging_cmap()
    cmap.set_bad("#dddddd")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=100,
                   aspect="auto")

    # Axes
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([ev.replace("_", " ").title() for ev in EVENT_ORDER],
                       rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([SHORT_LABEL.get(m, m) for m in models], fontsize=9)

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = grid[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center",
                        color="#666666", fontsize=8)
                continue
            color = "white" if (val < 30 or val > 70) else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    color=color, fontsize=8.5, fontweight="bold")

    # Domain separators above the heatmap
    domain_breaks: List[Tuple[int, int, str]] = []  # (start, end, name)
    cur_start = 0
    cur_dom = EVENT_DOMAIN[EVENT_ORDER[0]]
    for j in range(1, n_cols):
        d = EVENT_DOMAIN[EVENT_ORDER[j]]
        if d != cur_dom:
            domain_breaks.append((cur_start, j - 1, cur_dom))
            cur_start = j
            cur_dom = d
    domain_breaks.append((cur_start, n_cols - 1, cur_dom))
    # Vertical separators
    for start, end, _ in domain_breaks[:-1]:
        ax.axvline(end + 0.5, color="black", linewidth=1.3)
    # Domain labels above
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    sec = ax.secondary_xaxis("top")
    sec.set_xticks([(s + e) / 2 for s, e, _ in domain_breaks])
    sec.set_xticklabels([name for _, _, name in domain_breaks], fontsize=10,
                        fontweight="bold")
    sec.tick_params(axis="x", length=0)

    ax.set_title("Cross-Model Direction Match Rate by Event (P_match, %)",
                 pad=22, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("P_match (%)  —  50 = chance", fontsize=9)
    cbar.ax.axhline(50, color="black", linewidth=1)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_trait_heatmap(matrix: Dict[str, Dict[str, Tuple[Optional[float], int]]],
                       models: List[str], path: Path) -> None:
    """Rows = models, cols = 5 traits. Compact companion plot."""
    n_rows = len(models)
    n_cols = len(TRAIT_ORDER)
    grid = np.full((n_rows, n_cols), np.nan)
    for i, model in enumerate(models):
        for j, t in enumerate(TRAIT_ORDER):
            rate, _ = matrix.get(model, {}).get(t, (None, 0))
            if rate is not None:
                grid[i, j] = rate * 100

    fig, ax = plt.subplots(figsize=(5.2, 0.55 * n_rows + 1.4))
    cmap = _diverging_cmap()
    cmap.set_bad("#dddddd")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=100,
                   aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([TRAIT_LONG[t] for t in TRAIT_ORDER],
                       rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([SHORT_LABEL.get(m, m) for m in models], fontsize=9)

    for i in range(n_rows):
        for j in range(n_cols):
            val = grid[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center",
                        color="#666666", fontsize=8)
                continue
            color = "white" if (val < 30 or val > 70) else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")

    ax.set_title("Match Rate by Trait (P_match, %)", pad=10, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("P_match (%)", fontsize=9)
    cbar.ax.axhline(50, color="black", linewidth=1)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dirs", nargs="+", required=True,
                    type=Path, help="One or more results/v2_multi_*/ dirs.")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                    help="Model aliases to include (default: 11 curated).")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("paper/figures/cross_model"),
                    help="Output directory.")
    ap.add_argument("--epsilon", type=float, default=EPSILON,
                    help=f"Noise threshold for direction classification (default: {EPSILON}).")
    ap.add_argument("--show-pending", action="store_true",
                    help="Include models with no data as empty rows in plots.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load every model
    event_matrix: Dict[str, Dict[str, Tuple[Optional[float], int]]] = {}
    trait_matrix: Dict[str, Dict[str, Tuple[Optional[float], int]]] = {}
    overall_rows: List[Dict] = []
    completed: List[str] = []
    pending: List[str] = []

    for model in args.models:
        path = find_latest_full_results(model, args.results_dirs)
        if path is None:
            print(f"[skip] no full_results.json for {model}")
            pending.append(model)
            continue
        records = load_change_records(path)
        if not records:
            print(f"[skip] empty change records for {model} ({path})")
            pending.append(model)
            continue
        completed.append(model)
        print(f"[ok]   {model}: {len(records)} cells from {path}")

        event_matrix[model] = {}
        for ev in EVENT_ORDER:
            event_matrix[model][ev] = model_event_match(records, ev, args.epsilon)

        trait_matrix[model] = {}
        for t in TRAIT_ORDER:
            trait_matrix[model][t] = model_trait_match(records, t, args.epsilon)

        overall_rate, overall_n = model_overall_match(records, args.epsilon)
        event_matrix[model]["__overall__"] = (overall_rate, overall_n)
        trait_matrix[model]["__overall__"] = (overall_rate, overall_n)
        overall_rows.append({
            "model": model,
            "overall_rate": overall_rate,
            "overall_n": overall_n,
        })

    # CSVs
    write_per_model_event_csv(event_matrix, args.out_dir / "per_model_event_match.csv")
    write_per_model_trait_csv(trait_matrix, args.out_dir / "per_model_trait_match.csv")
    write_overall_csv(overall_rows, args.out_dir / "per_model_overall.csv")

    # Heatmaps. Order completed by descending overall match rate.
    completed_sorted = sorted(
        completed,
        key=lambda m: -(event_matrix[m]["__overall__"][0] or 0),
    )
    plot_models = completed_sorted + (pending if args.show_pending else [])
    if args.show_pending:
        for m in pending:
            event_matrix.setdefault(m, {})
            trait_matrix.setdefault(m, {})

    plot_event_heatmap(event_matrix, plot_models,
                       args.out_dir / "heatmap_event.pdf")
    plot_trait_heatmap(trait_matrix, plot_models,
                       args.out_dir / "heatmap_trait.pdf")

    # Console summary
    print("\n=== Overall match-rate ranking ===")
    for r in sorted(overall_rows, key=lambda x: -(x["overall_rate"] or 0)):
        rate = r["overall_rate"]
        rate_s = f"{rate*100:5.1f}%" if rate is not None else "  n/a "
        print(f"  {SHORT_LABEL.get(r['model'], r['model']):<22} {rate_s}  (n_active={r['overall_n']})")
    if pending:
        print(f"\n  Pending (no data yet): {', '.join(pending)}")


if __name__ == "__main__":
    main()
