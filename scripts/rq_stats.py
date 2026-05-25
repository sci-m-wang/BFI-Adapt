"""Compute RQ1/RQ3/RQ4 + construct-validity statistics from V2 results.

Reads the same multi-results-dirs the kappa/DCR/cross_model scripts read, and
produces a single CSV + a few plots used in the diagnostic paper.

RQ1 magnitude calibration:
  For each (model, event, trait) with an expected direction in {+,-}, compute
  mean |Δ| across the 100 personas. Compare to human meta-analytic effect-size
  envelope d ≈ 0.05–0.25 (Bühler et al. 2024; Bleidorn et al. 2018). BFI items
  are on a 1–5 Likert so a trait-level Δ ≈ 0.25 corresponds to roughly
  d ≈ 0.25 on a unit-SD scale (typical BFI SD ≈ 0.6–0.9).

RQ4 σ collapse:
  For each (model, event, trait) compute σ_LLM = std(Δ over 100 personas).
  Human longitudinal meta-analyses (Funke et al. 2024 variance-stability,
  Schwaba & Bleidorn 2018) report that within-trait SDs remain roughly
  constant pre/post life events (~0.5–0.8 on BFI Likert scale). σ_LLM far
  below that envelope indicates heterogeneity collapse.

RQ3 demographic invariance:
  For each (model, event, trait) compute the across-stratum variance of mean
  Δ when grouping personas by continent and by gender; small means the model
  applies the same Δ regardless of who the persona is.

Construct validity:
  Pearson(κ_persona, DCR_event) across the 11 models (should be < 0.5 to
  argue κ and DCR measure separable failure modes).

Usage:
  python scripts/rq_stats.py \
    --results-dirs results/v2_multi_curated_20260512_013721 \
                   results/v2_multi_20260512_124836 \
                   results/v2_gemini_20260512_195117 \
    --kappa-csv paper/figures/kappa_dcr/per_model.csv \
    --out-dir paper/figures/rq_stats
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from full_experiment_v2 import LIFE_EVENTS  # noqa: E402
from kappa_dcr_analysis import BFI_PATH, DEFAULT_MODELS, SHORT_LABEL, load_bfi_meta  # noqa: E402

# Trait short-codes used in LIFE_EVENTS expected dict
TRAIT_BY_CODE = {
    "E": "Extraversion",
    "A": "Agreeableness",
    "C": "Conscientiousness",
    "N": "Neuroticism",
    "O": "Openness",
}

# Human anchors (literature-grounded, see file docstring):
# Mean-level meta-analytic effect sizes for life events on Big Five traits.
HUMAN_D_LOW = 0.05   # typical small effects (Bühler 2024 most cells)
HUMAN_D_HIGH = 0.20  # typical large effects (Bühler 2024 strongest cells)
# Within-trait individual SD on BFI Likert scale (1–5):
HUMAN_SIGMA_LOW = 0.5
HUMAN_SIGMA_HIGH = 0.8
# Noise floor for "did the persona move at all" on BFI-44 trait means.
# A 1-Likert-tick change on a single item shifts the trait mean by 1/n_t, where
# n_t is the number of items on trait t (E:8, A:9, C:9, N:8, O:10), giving step
# sizes {0.125, 0.111, 0.10}. We use 0.1 = 1/n_O, matching the smallest one-item
# step across the five trait means (Openness exactly) and sitting one tick below
# the one-item step for the other four traits. Trait-level Δ at or below 0.1 is
# therefore at most one item moving by one Likert tick on at least one trait,
# i.e. the minimum physically discriminable rated response. The V2 pipeline
# (full_experiment_v2.py) treats |Δ|>0.1 as a directional shift and |Δ|≤0.1 as
# neutral; this module preserves that strict-> convention end-to-end.
NOISE_FLOOR = 0.1


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_full(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def collect_cells(results_dirs: List[Path], models: List[str]) -> Dict[str, List[dict]]:
    """Return {model: [cell, ...]} taking the later-dir version when both
    exist with at least as many cells."""
    out: Dict[str, List[dict]] = {}
    out_src: Dict[str, str] = {}
    for rd in results_dirs:
        for m in models:
            fp = rd / m / "full_results.json"
            data = load_full(fp)
            if data is None:
                continue
            cells = data.get("results", [])
            if not cells:
                continue
            prev = out.get(m)
            if prev is None or len(cells) >= len(prev):
                out[m] = cells
                out_src[m] = str(fp.relative_to(rd.parent.parent)) if rd.parent.parent in fp.parents else str(fp)
    for m in models:
        if m in out:
            print(f"  [ok] {m:24s} n_cells={len(out[m])}  src={out_src[m]}")
        else:
            print(f"  [missing] {m}")
    return out


# ---------------------------------------------------------------------------
# RQ1 magnitude calibration
# ---------------------------------------------------------------------------

def rq1_magnitude(model_cells: Dict[str, List[dict]]) -> List[dict]:
    """Per (model, event, trait_with_expected_direction) compute distribution
    statistics over the 100 personas (NOT a mean: the persona-axis carries the
    individual-heterogeneity signal RQ4 measures, so we keep the distribution
    rather than collapse it).

    Reported per cell:
      median_delta     : persona-level median signed Δ (centre of distribution)
      pct_in_dir       : fraction of personas whose Δ matches the human prior
                         direction (>0 for '+', <0 for '-'), excluding exact 0s
      q25, q75         : persona-level inter-quartile envelope of Δ
      cohens_d_approx  : median_delta / 0.7 (BFI Likert σ anchor)
    """
    from statistics import median
    rows = []
    for m, cells in model_cells.items():
        bucket: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
        for c in cells:
            ev = c["event"]
            for code, sign in LIFE_EVENTS[ev]["expected_changes"].items():
                trait = TRAIT_BY_CODE[code]
                delta = c["change"][trait]
                bucket[(ev, trait, sign)].append(float(delta))
        for (ev, trait, sign), deltas in bucket.items():
            if sign == "?":
                continue
            n = len(deltas)
            med = median(deltas)
            sorted_d = sorted(deltas)
            q25 = sorted_d[max(0, int(0.25 * (n - 1)))]
            q75 = sorted_d[min(n - 1, int(0.75 * (n - 1)))]
            in_dir = sum(1 for x in deltas if (x > 0 and sign == "+") or
                                              (x < 0 and sign == "-"))
            pct_in_dir = in_dir / n if n else 0.0
            d_est = med / 0.7
            rows.append({
                "model": m,
                "event": ev,
                "trait": trait,
                "expected_sign": sign,
                "n_personas": n,
                "median_delta": round(med, 4),
                "pct_in_dir": round(pct_in_dir, 4),
                "q25": round(q25, 4),
                "q75": round(q75, 4),
                "cohens_d_approx": round(d_est, 4),
            })
    return rows


# ---------------------------------------------------------------------------
# RQ1/RQ2 full grid: all 11 events x 5 traits signed Δ (incl. uncertain cells)
# ---------------------------------------------------------------------------

def rq1_full_grid(model_cells: Dict[str, List[dict]],
                  eps: float = NOISE_FLOOR) -> List[dict]:
    """Per (model, event, trait) compute persona-level distribution statistics
    across ALL 11 events x 5 traits = 55 cells (regardless of prior).

    Used by the 5-trait Δ grid that shows BOTH target-direction fidelity AND
    no-definite-direction stability. We deliberately avoid persona-axis mean: 100 personas
    are samples FROM the distribution we want to characterise (heterogeneity in
    individual-level reaction to the same event), not entities whose value
    should be collapsed away. We therefore report:

      median_delta       : centre of the persona-level Δ distribution
      pct_in_dir         : for directional priors, fraction of personas whose
                           Δ matches the human prior direction
      pct_offtarget      : for no-definite-direction pairs, fraction of
                           personas with |Δ|>eps (i.e. above the noise floor)
      pct_pos / pct_neg  : raw fraction of personas with Δ>0 / Δ<0
    """
    from statistics import median
    rows = []
    for m, cells in model_cells.items():
        bucket: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for c in cells:
            ev = c["event"]
            for trait in TRAIT_BY_CODE.values():
                delta = c["change"][trait]
                bucket[(ev, trait)].append(float(delta))
        expected = LIFE_EVENTS  # alias
        for (ev, trait), deltas in bucket.items():
            sign_dict = expected[ev]["expected_changes"]
            code = next((k for k, v in TRAIT_BY_CODE.items() if v == trait), None)
            prior = sign_dict.get(code, "0")  # 0 = no expectation
            n = len(deltas)
            med = median(deltas)
            n_pos = sum(1 for d in deltas if d > 0)
            n_neg = sum(1 for d in deltas if d < 0)
            n_off = sum(1 for d in deltas if abs(d) > eps)
            if prior == "+":
                pct_in_dir = n_pos / n
            elif prior == "-":
                pct_in_dir = n_neg / n
            else:
                pct_in_dir = None
            rows.append({
                "model": m,
                "event": ev,
                "trait": trait,
                "expected_sign": prior,
                "n_personas": n,
                "median_delta": round(med, 4),
                "pct_in_dir": round(pct_in_dir, 4) if pct_in_dir is not None else "",
                "pct_offtarget": round(n_off / n, 4),
                "pct_pos": round(n_pos / n, 4),
                "pct_neg": round(n_neg / n, 4),
            })
    return rows


def rq1_full_grid_summary(rows: List[dict], eps: float = NOISE_FLOOR,
                          net_thr: float = 0.10) -> List[dict]:
    """Collapse rq1_full_grid across the 11 models. Cross-model axis IS reduced
    by median (models are independent entities, not samples from a distribution
    we are characterising). Persona axis is already preserved upstream as
    pct_pos / pct_neg / pct_offtarget.

    Per (event, trait) cell we report (all medians taken across the 11 models):
      pct_pos_med    : median across models of fraction of personas with Δ>0
      pct_neg_med    : median across models of fraction of personas with Δ<0
      pct_offtarget_med : median across models of fraction with |Δ|>eps
      median_delta   : median across models of the persona-level median Δ
                       (informational only --- BFI's 9-item discrete scale
                       drives most medians to 0, so this column is auxiliary)
      n_models_match : count of models whose net directional intensity
                       (pct_in_dir - pct_against) exceeds net_thr
      class          : match / reverse / drift / stable / uncertain (based on
                       cross-model median pct_pos vs pct_neg with threshold
                       net_thr; replaces any centre-statistic dependence)
    """
    from statistics import median
    by_cell: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    prior_by_cell: Dict[Tuple[str, str], str] = {}
    for r in rows:
        key = (r["event"], r["trait"])
        by_cell[key].append(r)
        prior_by_cell[key] = r["expected_sign"]
    n_models = len(set(r["model"] for r in rows))
    out = []
    for (ev, trait), recs in by_cell.items():
        prior = prior_by_cell[(ev, trait)]
        pos_list = [r["pct_pos"] for r in recs]
        neg_list = [r["pct_neg"] for r in recs]
        med_list = [r["median_delta"] for r in recs]
        off_list = [r["pct_offtarget"] for r in recs]
        pct_pos_med = median(pos_list)
        pct_neg_med = median(neg_list)
        med_of_med = median(med_list)
        pct_off_med = median(off_list)
        if prior == "+":
            n_match = sum(1 for r in recs
                          if (r["pct_pos"] - r["pct_neg"]) > net_thr)
            cls = "match" if (pct_pos_med - pct_neg_med) > net_thr else "reverse"
        elif prior == "-":
            n_match = sum(1 for r in recs
                          if (r["pct_neg"] - r["pct_pos"]) > net_thr)
            cls = "match" if (pct_neg_med - pct_pos_med) > net_thr else "reverse"
        elif prior == "?":
            n_match = -1
            cls = "uncertain"
        else:  # "0"
            n_match = sum(1 for r in recs if r["pct_offtarget"] > 0.5)
            cls = "drift" if pct_off_med > 0.5 else "stable"
        out.append({
            "event": ev,
            "trait": trait,
            "expected_sign": prior,
            "n_models": n_models,
            "pct_pos_med": round(pct_pos_med, 4),
            "pct_neg_med": round(pct_neg_med, 4),
            "pct_offtarget_med": round(pct_off_med, 4),
            "median_delta": round(med_of_med, 4),
            "n_models_match": n_match,
            "class": cls,
        })
    return out


# ---------------------------------------------------------------------------
# RQ4 sigma collapse
# ---------------------------------------------------------------------------

def rq4_sigma(model_cells: Dict[str, List[dict]]) -> List[dict]:
    rows = []
    for m, cells in model_cells.items():
        bucket: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for c in cells:
            ev = c["event"]
            for trait in TRAIT_BY_CODE.values():
                delta = c["change"][trait]
                bucket[(ev, trait)].append(float(delta))
        for (ev, trait), deltas in bucket.items():
            if len(deltas) < 2:
                continue
            sigma = pstdev(deltas)
            sorted_d = sorted(deltas)
            n = len(sorted_d)
            q25 = sorted_d[max(0, int(0.25 * (n - 1)))]
            q75 = sorted_d[min(n - 1, int(0.75 * (n - 1)))]
            iqr = q75 - q25
            pct_nonzero = sum(1 for x in deltas if abs(x) > NOISE_FLOOR) / n
            rows.append({
                "model": m,
                "event": ev,
                "trait": trait,
                "n_personas": len(deltas),
                "sigma_llm": round(sigma, 4),
                "iqr_llm": round(iqr, 4),
                "pct_personas_moved": round(pct_nonzero, 4),
                "human_sigma_low": HUMAN_SIGMA_LOW,
                "human_sigma_high": HUMAN_SIGMA_HIGH,
                "collapse_ratio_to_low": round(sigma / HUMAN_SIGMA_LOW, 4),
            })
    return rows


# ---------------------------------------------------------------------------
# RQ1 existence: persona-level noise-floor crossing rate
# ---------------------------------------------------------------------------

def rq_existence(model_cells: Dict[str, List[dict]]) -> List[dict]:
    """Per (model, event, trait) cell, compute the fraction of the 100 personas
    whose absolute Δ exceeds the noise floor ε = 0.1 = 1/n_O (the smallest
    one-item step in BFI-44 trait means; one tick below the step for the other
    four traits). This is the primary indicator for new RQ1 'do PC-Agents move
    at all under life events?'. The strict-> convention matches the V2 pipeline
    in full_experiment_v2.py and paper Eq.~(\\ref{eq:direction}).

    Each row is a (model, event, trait) cell; the row's prior bucket
    ('prior' = directional pair with sign in {+,−}; 'off_target' = no definite direction or
    uncertain) is recorded so callers can split the 27 directional cells from
    the 28 non-directional cells without re-deriving it."""
    rows = []
    for m, cells in model_cells.items():
        bucket: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for c in cells:
            ev = c["event"]
            for trait in TRAIT_BY_CODE.values():
                delta = c["change"][trait]
                bucket[(ev, trait)].append(float(delta))
        for (ev, trait), deltas in bucket.items():
            if not deltas:
                continue
            sign_dict = LIFE_EVENTS[ev]["expected_changes"]
            code = next((k for k, v in TRAIT_BY_CODE.items() if v == trait), None)
            prior = sign_dict.get(code, "0")
            if prior in ("+", "-"):
                bucket_label = "prior"
            else:
                bucket_label = "off_target"
            n = len(deltas)
            n_moved = sum(1 for x in deltas if abs(x) > NOISE_FLOOR)
            rows.append({
                "model": m,
                "event": ev,
                "trait": trait,
                "expected_sign": prior,
                "bucket": bucket_label,
                "n_personas": n,
                "pct_personas_moved": round(n_moved / n, 4),
            })
    return rows


# ---------------------------------------------------------------------------
# RQ2 signed-magnitude calibration with mutually exclusive bins
# ---------------------------------------------------------------------------

def rq2_signed_magnitude(model_cells: Dict[str, List[dict]]) -> List[dict]:
    """Per (model, event, trait, persona) instance, compute the signed
    projected Δ = sign(prior) · Δ. Cells with a definite prior project Δ onto
    that prior; cells without a definite prior project onto the 'should not
    move' anchor by negating |Δ| (any movement counts as wrong-direction).
    Context-dependent (±) cells are excluded.

    Each output row preserves a single persona-level signed Δ instance — we
    deliberately do NOT collapse over the persona axis here; RQ4 needs that
    distribution intact, and aggregate statistics (median, in-direction rate,
    under/in/overshoot fractions) are computed downstream from this long table.

    Mutually exclusive bins for definite-direction pairs:
      reversed    : signed Δ < 0
      under_shift : 0 ≤ signed Δ < HUMAN_D_LOW (0.05)
      in_range    : HUMAN_D_LOW ≤ signed Δ ≤ HUMAN_D_HIGH (0.05–0.20)
      overshoot   : signed Δ > HUMAN_D_HIGH (0.20)
    For no-definite-direction pairs we retain the same signed convention used by
    older exploratory files, but main-paper RQ2 reports only definite-direction
    pairs.
    """
    rows = []
    for m, cells in model_cells.items():
        for c in cells:
            ev = c["event"]
            persona_id = c.get("persona_id", c.get("persona", {}).get("id", ""))
            for trait in TRAIT_BY_CODE.values():
                delta = float(c["change"][trait])
                sign_dict = LIFE_EVENTS[ev]["expected_changes"]
                code = next((k for k, v in TRAIT_BY_CODE.items() if v == trait), None)
                prior = sign_dict.get(code, "0")
                if prior == "?":
                    continue  # context-dependent; excluded from magnitude
                if prior == "+":
                    signed = delta
                    bucket_label = "prior"
                elif prior == "-":
                    signed = -delta
                    bucket_label = "prior"
                else:  # no-definite-direction pair: any movement is treated as drift
                    signed = -abs(delta)
                    bucket_label = "off_target"
                if bucket_label == "prior" and signed < 0:
                    bin_ = "reversed"
                elif signed < HUMAN_D_LOW:
                    bin_ = "under_shift"
                elif signed > HUMAN_D_HIGH:
                    bin_ = "overshoot"
                else:
                    bin_ = "in_range"
                rows.append({
                    "model": m,
                    "event": ev,
                    "trait": trait,
                    "expected_sign": prior,
                    "bucket": bucket_label,
                    "persona_id": persona_id,
                    "delta": round(delta, 4),
                    "signed_delta": round(signed, 4),
                    "envelope_bin": bin_,
                })
    return rows


# ---------------------------------------------------------------------------
# RQ3 demographic invariance (continent × gender)
# ---------------------------------------------------------------------------

def rq3_invariance(model_cells: Dict[str, List[dict]]) -> List[dict]:
    """Demographic invariance: for each (model, event, trait) compute the SD
    across the 10 demographic strata (2 genders x 5 continents) of the
    stratum-level Δ. We use the persona-level median within each stratum to
    avoid collapsing the persona distribution within strata."""
    from statistics import median
    rows = []
    for m, cells in model_cells.items():
        bucket: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
        for c in cells:
            ev = c["event"]
            cont = c["persona"].get("continent", "?")
            gen = c["persona"].get("gender", "?")
            for trait in TRAIT_BY_CODE.values():
                delta = c["change"][trait]
                bucket[(ev, trait, cont, gen)].append(float(delta))
        # collapse to stratum medians then take SD across strata per (event,trait)
        per_cell: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for (ev, trait, cont, gen), deltas in bucket.items():
            if not deltas:
                continue
            per_cell[(ev, trait)].append(median(deltas))
        for (ev, trait), stratum_meds in per_cell.items():
            if len(stratum_meds) < 2:
                continue
            sd_across = pstdev(stratum_meds)
            rows.append({
                "model": m,
                "event": ev,
                "trait": trait,
                "n_strata": len(stratum_meds),
                "sd_across_strata": round(sd_across, 4),
            })
    return rows


def rq3_per_model_summary(rq3_rows: List[dict]) -> List[dict]:
    """Per-model table for RQ3 demographic-invariance results."""
    by_model: Dict[str, List[float]] = defaultdict(list)
    for r in rq3_rows:
        by_model[r["model"]].append(r["sd_across_strata"])
    rows = []
    ordered = [m for m in DEFAULT_MODELS if m in by_model]
    ordered += [m for m in by_model if m not in ordered]
    for m in ordered:
        vals = by_model[m]
        if not vals:
            continue
        rows.append({
            "model": m,
            "median_sd_across_strata": round(float(np.median(vals)), 4),
            "max_sd_across_strata": round(float(np.max(vals)), 4),
            "pct_pairs_sd_below_0_10": round(sum(v < NOISE_FLOOR for v in vals) / len(vals), 4),
            "n_pairs": len(vals),
        })
    return rows


def baseline_diversity(model_cells: Dict[str, List[dict]],
                       item2trait: Dict[int, str],
                       reverse_set: set) -> List[dict]:
    """Across-persona baseline BFI diversity for the RQ4 sanity check."""
    rows = []
    for m, cells in model_cells.items():
        by_persona: Dict[str, Dict[str, float]] = {}
        for c in cells:
            persona_id = c.get("persona_id", c.get("persona", {}).get("id", ""))
            if not persona_id or persona_id in by_persona:
                continue
            baseline = (c.get("baseline") or {}).get("item_scores") or {}
            trait_vals: Dict[str, List[float]] = defaultdict(list)
            for raw_id, raw_score in baseline.items():
                try:
                    item_id = int(raw_id)
                    score = float(raw_score)
                except Exception:
                    continue
                trait = item2trait.get(item_id)
                if trait is None:
                    continue
                if item_id in reverse_set:
                    score = 6.0 - score
                trait_vals[trait].append(score)
            scores = {
                trait: float(np.mean(vals))
                for trait, vals in trait_vals.items()
                if vals
            }
            if scores:
                by_persona[persona_id] = scores
        for trait in sorted({t for scores in by_persona.values() for t in scores}):
            vals = [scores[trait] for scores in by_persona.values() if trait in scores]
            if len(vals) < 2:
                continue
            rows.append({
                "model": m,
                "trait": trait,
                "n_personas": len(vals),
                "baseline_sd": round(float(np.std(vals)), 4),
            })
    return rows


# ---------------------------------------------------------------------------
# Construct validity: Pearson(κ, DCR)
# ---------------------------------------------------------------------------

def construct_validity(kappa_csv: Path) -> Tuple[float, int, List[Tuple[str, float, float]]]:
    """Pearson r between pair-level (κ̄_pair, DCR̄_pair) over the 27 prior pairs.

    Falls back to pooled (κ_pool, DCR_pool) when L2 columns absent, but the
    refactored pipeline always emits both."""
    rows = []
    with open(kappa_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                k = float(r.get("kappa_avg", r.get("kappa", "nan")))
                dcr = float(r.get("dcr_avg", r.get("dcr_pooled", r.get("DCR", r.get("dcr", "nan")))))
            except Exception:
                continue
            if not (np.isnan(k) or np.isnan(dcr)):
                rows.append((r["model"], k, dcr))
    if len(rows) < 3:
        return float("nan"), len(rows), rows
    ks = np.array([r[1] for r in rows])
    ds = np.array([r[2] for r in rows])
    r = float(np.corrcoef(ks, ds)[0, 1])
    return r, len(rows), rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_sigma_distribution(rq4_rows: List[dict], out_path: Path):
    """One violin per model showing distribution of σ_LLM across all
    (event, trait) cells, with the human σ envelope shaded."""
    by_model: Dict[str, List[float]] = defaultdict(list)
    for r in rq4_rows:
        by_model[r["model"]].append(r["sigma_llm"])
    # keep canonical order; only models present
    models = [m for m in DEFAULT_MODELS if m in by_model]
    labels = [SHORT_LABEL.get(m, m) for m in models]
    data = [by_model[m] for m in models]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    parts = ax.violinplot(data, showmeans=True, showextrema=True, widths=0.85)
    for body in parts['bodies']:
        body.set_alpha(0.55)
        body.set_facecolor('#7aa8c9')
        body.set_edgecolor('#33647f')
    ax.axhspan(HUMAN_SIGMA_LOW, HUMAN_SIGMA_HIGH, color='#e8a07a', alpha=0.30,
               label=f'Human within-trait σ envelope ({HUMAN_SIGMA_LOW}–{HUMAN_SIGMA_HIGH})')
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel(r'$\sigma_{\mathrm{LLM}}$ across 100 personas per event--trait pair')
    ax.set_title('Heterogeneity collapse: LLM trait-change SD vs human longitudinal SD anchor')
    ax.set_ylim(0, max(HUMAN_SIGMA_HIGH * 1.1, max(max(d) for d in data) * 1.05))
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix('.png'), dpi=200)
    fig.savefig(out_path.with_suffix('.pdf'))
    plt.close(fig)


def plot_magnitude_distribution(rq1_rows: List[dict], out_path: Path):
    """For each model, distribution of persona-level median Δ projected onto
    the expected direction (per cell, one value)."""
    by_model: Dict[str, List[float]] = defaultdict(list)
    for r in rq1_rows:
        # Use signed median for in-direction cells (multiply by expected sign)
        sign = 1.0 if r["expected_sign"] == "+" else -1.0
        s = r["median_delta"] * sign
        by_model[r["model"]].append(s)
    models = [m for m in DEFAULT_MODELS if m in by_model]
    labels = [SHORT_LABEL.get(m, m) for m in models]
    data = [by_model[m] for m in models]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    parts = ax.violinplot(data, showmeans=True, showextrema=True, widths=0.85)
    for body in parts['bodies']:
        body.set_alpha(0.55)
        body.set_facecolor('#7ec48f')
        body.set_edgecolor('#2f6a3d')
    # human d envelope scaled to BFI Likert: Δ ≈ d × σ_human (use σ=0.7 anchor)
    d_low_delta = HUMAN_D_LOW * 0.7
    d_high_delta = HUMAN_D_HIGH * 0.7
    ax.axhspan(d_low_delta, d_high_delta, color='#e8a07a', alpha=0.30,
               label=f'Human d=({HUMAN_D_LOW}–{HUMAN_D_HIGH}) × σ≈0.7 → Δ≈({d_low_delta:.2f}–{d_high_delta:.2f})')
    ax.axhline(0, color='gray', linewidth=0.7)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Signed Δ in expected direction (BFI Likert units)')
    ax.set_title('Magnitude calibration: LLM trait shifts vs human meta-analytic effect-size envelope')
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix('.png'), dpi=200)
    fig.savefig(out_path.with_suffix('.pdf'))
    plt.close(fig)


def plot_construct_validity(rows: List[Tuple[str, float, float]], r: float, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    xs = [x[1] for x in rows]
    ys = [x[2] for x in rows]
    ax.scatter(xs, ys, c='#33647f', s=40)
    for name, x, y in rows:
        ax.annotate(SHORT_LABEL.get(name, name), (x, y), fontsize=7,
                    xytext=(4, 3), textcoords='offset points')
    ax.set_xlabel(r'$\bar{\kappa}_{\mathrm{pair}}$ (mean over 27 prior pairs)')
    ax.set_ylabel(r'$\overline{\mathrm{DCR}}_{\mathrm{pair}}$ (mean over 27 prior pairs)')
    ax.set_title(rf'Construct validity:' + '\n' +
                 rf'Pearson $r$={r:+.3f} between $\bar{{\kappa}}_{{\mathrm{{pair}}}}$ and $\overline{{\mathrm{{DCR}}}}_{{\mathrm{{pair}}}}$ (N={len(rows)})', fontsize=11)
    ax.grid(linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix('.png'), dpi=200)
    fig.savefig(out_path.with_suffix('.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def aggregate_summary(rq1: List[dict], rq4: List[dict], rq3: List[dict],
                      r_kd: float, n_kd: int) -> dict:
    """Per-model summary statistics. We deliberately summarise persona-level
    median Δ (NOT mean), and pct_in_dir, to avoid collapsing the distribution."""
    from statistics import median
    by_model_med_abs: Dict[str, List[float]] = defaultdict(list)
    by_model_pct: Dict[str, List[float]] = defaultdict(list)
    by_model_sigma: Dict[str, List[float]] = defaultdict(list)
    by_model_sd_strata: Dict[str, List[float]] = defaultdict(list)
    for r in rq1:
        by_model_med_abs[r["model"]].append(abs(r["median_delta"]))
        by_model_pct[r["model"]].append(r["pct_in_dir"])
    for r in rq4:
        by_model_sigma[r["model"]].append(r["sigma_llm"])
    for r in rq3:
        by_model_sd_strata[r["model"]].append(r["sd_across_strata"])
    out = {
        "construct_validity": {
            "pearson_kappa_dcr": round(r_kd, 4),
            "n_models": n_kd,
        },
        "per_model": {},
    }
    for m in by_model_med_abs:
        out["per_model"][m] = {
            "median_abs_median_delta_expected_cells":
                round(median(by_model_med_abs[m]), 4),
            "median_pct_in_dir_expected_cells":
                round(median(by_model_pct[m]), 4),
            "median_sigma_all_cells":
                round(median(by_model_sigma[m]), 4),
            "median_sd_across_strata_all_cells":
                round(median(by_model_sd_strata[m]), 4),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_csv(rows: List[dict], path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_rq3_per_model_tex(rows: List[dict], path: Path):
    lines = []
    lines.append(r"% Auto-generated by scripts/rq_stats.py -- DO NOT EDIT")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Median SD & Max SD & $\%<0.10$ \\")
    lines.append(r"\midrule")
    ordered = [m for m in DEFAULT_MODELS if any(r["model"] == m for r in rows)]
    ordered += [r["model"] for r in rows if r["model"] not in ordered]
    by_model = {r["model"]: r for r in rows}
    for m in ordered:
        r = by_model[m]
        lines.append(
            f"{SHORT_LABEL.get(m, m)} & "
            f"{r['median_sd_across_strata']:.3f} & "
            f"{r['max_sd_across_strata']:.3f} & "
            f"{100*r['pct_pairs_sd_below_0_10']:.1f} \\\\" 
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Per-model demographic-shape statistics for RQ3. "
        r"For each model--event--trait combination we compute the standard "
        r"deviation of the ten demographic-stratum median $\Delta$ values "
        r"(2 genders $\times$ 5 continents). The table reports the median and "
        r"maximum across the 55 event--trait pairs per model, plus the fraction "
        r"of pairs whose across-strata SD is below 0.10 BFI Likert units.}"
    )
    lines.append(r"\label{tab:rq3_demographic_shape}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Order used in the 5-trait grid table (matches Table 1 in paper)
GRID_TRAIT_ORDER = ["Extraversion", "Agreeableness", "Conscientiousness",
                    "Neuroticism", "Openness"]
GRID_TRAIT_CODE = {t: c for c, t in TRAIT_BY_CODE.items()}

# Order used in the 5-trait grid table (matches Table 1 in paper)
GRID_EVENT_ORDER = [
    ("graduation",      "Graduation"),
    ("work_entry",      "Work Entry"),
    ("job_change",      "Job Change"),
    ("promotion",       "Promotion"),
    ("unemployment",    "Unemployment"),
    ("retirement",      "Retirement"),
    ("new_relationship", "New Rel."),
    ("marriage",        "Marriage"),
    ("divorce",         "Divorce"),
    ("child_birth",     "Child Birth"),
    ("chronic_illness", "Chronic Ill."),
]


def write_cell_grid_tex(rq1_summary: List[dict], path: Path, eps: float = NOISE_FLOOR,
                        net_thr: float = 0.10):
    """Render the (event × trait) Δ grid with traffic-light colouring.

    BFI uses a discrete 1--5 Likert with 9 items per trait, so persona-level
    medians cluster on multiples of 1/9 and frequently land on 0 even when
    the persona distribution is clearly skewed. We therefore avoid any centre
    statistic and report the persona axis directly as

        arrow + pct_pos% / pct_neg%

    where pct_pos (pct_neg) is the cross-model median of the fraction of the
    100 personas with $\\Delta > 0$ ($\\Delta < 0$). This is the cell-level
    analogue of DCR: a non-parametric, distribution-faithful directional
    intensity that does not depend on the magnitude of individual shifts.

    Background:
        match     -> light green   (prior matched: net = pct_in_dir - pct_against > net_thr)
        reverse   -> light red     (prior reversed)
        drift     -> light yellow  (no definite direction: pct_offtarget > 0.5 in cross-model median)
        stable    -> white         (no definite direction, mostly within noise floor)
        uncertain -> light gray    (prior = ?)
    """
    by_cell: Dict[Tuple[str, str], dict] = {(r["event"], r["trait"]): r
                                             for r in rq1_summary}
    arrow = {"+": r"$\uparrow$", "-": r"$\downarrow$",
             "?": r"$\pm$", "0": r"--"}
    colour = {
        "match":     r"\cellcolor{cgrn}",
        "reverse":   r"\cellcolor{cred}",
        "drift":     r"\cellcolor{cyel}",
        "stable":    "",
        "uncertain": r"\cellcolor{cgry}",
    }
    lines = []
    lines.append(r"% auto-generated by scripts/rq_stats.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering\scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{1.6pt}")
    lines.append(r"  \renewcommand{\arraystretch}{1.10}")
    lines.append(r"  \begin{tabular}{l ccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Event} & \textbf{E} & \textbf{A} & \textbf{C} & \textbf{N} & \textbf{O} \\")
    lines.append(r"    \midrule")
    for ev_key, ev_label in GRID_EVENT_ORDER:
        cells = []
        for trait in GRID_TRAIT_ORDER:
            rec = by_cell.get((ev_key, trait))
            if not rec:
                cells.append(r"--")
                continue
            prior = rec["expected_sign"]
            cls = rec["class"]
            pp = int(round(rec.get("pct_pos_med", 0.0) * 100))
            pn = int(round(rec.get("pct_neg_med", 0.0) * 100))
            txt = f"{arrow[prior]}\\,{pp}/{pn}"
            cells.append(f"{colour[cls]} {txt}")
        lines.append("    " + ev_label + " & " + " & ".join(cells) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(
        r"  \caption{Per event--trait pair, against the "
        r"expected human direction from \citet{SPECHT2017341}. Each entry shows the arrow "
        r"of the prior ($\uparrow$ increase, $\downarrow$ decrease, "
        r"$\pm$ context-dependent, -- no expectation) followed by "
        r"$\mathrm{pct}_{+}\,/\,\mathrm{pct}_{-}$: the percentage of "
        r"the 100 personas with post-event $\Delta>0$ and $\Delta<0$ "
        r"respectively, reported as the cross-model median over 11 "
        r"models. \colorbox{cgrn}{green}: expected direction matched "
        r"(net $\mathrm{pct}_{\text{in-dir}}-\mathrm{pct}_{\text{against}}>"
        + f"{net_thr:.2f}" + r"$). \colorbox{cred}{red}: prior "
        r"reversed. \colorbox{cyel}{yellow}: no definite direction, but more than "
        r"half the personas left the noise floor ($|\Delta|>"
        + f"{eps:g}$). "
        r"White: no definite direction, stable. \colorbox{cgry}{gray}: "
        r"context-dependent direction.}")
    lines.append(r"  \label{tab:cell_grid}")
    lines.append(r"\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dirs", nargs="+", required=True)
    ap.add_argument("--kappa-csv", required=True,
                    help="paper/figures/kappa_dcr/per_model.csv from kappa_dcr_analysis.py")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dirs = [Path(p) for p in args.results_dirs]
    print("== loading cells ==")
    model_cells = collect_cells(results_dirs, args.models)
    item2trait, reverse_set = load_bfi_meta(BFI_PATH)

    print("\n== RQ1 magnitude ==")
    rq1 = rq1_magnitude(model_cells)
    write_csv(rq1, out_dir / "rq1_magnitude.csv")

    print("== RQ1/RQ2 full grid (all 55 cells) ==")
    rq1_full = rq1_full_grid(model_cells)
    write_csv(rq1_full, out_dir / "rq1_full_grid.csv")
    rq1_summary = rq1_full_grid_summary(rq1_full)
    write_csv(rq1_summary, out_dir / "rq1_full_grid_summary.csv")
    write_cell_grid_tex(rq1_summary, out_dir / "cell_grid.tex")

    print("== RQ1 existence (persona |Δ|>0.1 crossings) ==")
    rq_ex = rq_existence(model_cells)
    write_csv(rq_ex, out_dir / "rq_existence.csv")

    print("== RQ2 signed-projected magnitude (persona-level long form) ==")
    rq2_sm = rq2_signed_magnitude(model_cells)
    write_csv(rq2_sm, out_dir / "rq2_signed_magnitude.csv")

    print("== RQ4 sigma collapse ==")
    rq4 = rq4_sigma(model_cells)
    write_csv(rq4, out_dir / "rq4_sigma.csv")

    print("== RQ3 demographic invariance ==")
    rq3 = rq3_invariance(model_cells)
    write_csv(rq3, out_dir / "rq3_invariance.csv")
    rq3_pm = rq3_per_model_summary(rq3)
    write_csv(rq3_pm, out_dir / "rq3_per_model.csv")
    write_rq3_per_model_tex(rq3_pm, out_dir / "rq3_per_model.tex")

    print("== baseline persona diversity ==")
    baseline = baseline_diversity(model_cells, item2trait, reverse_set)
    write_csv(baseline, out_dir / "baseline_diversity.csv")

    print("== construct validity ==")
    r_kd, n_kd, kd_rows = construct_validity(Path(args.kappa_csv))
    print(f"  Pearson(κ, DCR) over {n_kd} models = {r_kd:+.4f}")

    print("\n== plots ==")
    plot_sigma_distribution(rq4, out_dir / "rq4_sigma_violin")
    plot_magnitude_distribution(rq1, out_dir / "rq1_magnitude_violin")
    plot_construct_validity(kd_rows, r_kd, out_dir / "construct_validity_scatter")

    print("\n== aggregate summary ==")
    summary = aggregate_summary(rq1, rq4, rq3, r_kd, n_kd)
    print(json.dumps(summary, indent=2))
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOutputs under {out_dir}/:")
    print("  rq1_magnitude.csv  rq4_sigma.csv  rq3_invariance.csv  rq3_per_model.csv  baseline_diversity.csv  summary.json")
    print("  rq1_magnitude_violin.{png,pdf}  rq4_sigma_violin.{png,pdf}  construct_validity_scatter.{png,pdf}")


if __name__ == "__main__":
    main()
