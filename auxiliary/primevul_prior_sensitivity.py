"""Sensitivity of Naive Bayes fusion to the PrimeVul safe-sample downsampling prior.

Motivation: a reviewer argued that downsampling PrimeVul's safe class from 193,047 to
10,000 rows (auxiliary/sample_primevul.py) inflates the class prior P(y=1) that Naive
Bayes fusion (analysis/fusion/bayes.py, Eq. 4 in the paper) draws from the calibration
confusion matrix, and asked for a sensitivity analysis re-evaluating NB fusion under a
corrected prior -- either the original PrimeVul ratio (193,047:3,277 in this repo) or a
realistic industry prevalence of ~0.1%.

IMPORTANT METHODOLOGICAL NOTE (this went through a revision -- see git history of this
file): an earlier version of this script only swapped the prior term *inside* the NB
scoring formula and re-evaluated on the same (still-downsampled) validation fold. That
is close to tautological: on a fixed validation set, replacing a family's prior log-odds
with a different constant is a strictly monotonic per-family rescoring, which trivially
cannot change the ranking (ROC/PR-AUC) or the best F1 reachable by an unconstrained
threshold search on THAT SAME set -- regardless of whether the corrected prior bears any
relationship to reality. It does not test what the reviewer actually asked: whether
results hold up if the *real* class balance were different. Precision (and hence F1) is
NOT invariant to a change in the test population's actual prevalence, even though ROC-AUC
is -- more realistic negatives means more absolute false positives at any fixed
per-sample false-positive rate, which straightforwardly hurts precision and is NOT
undone by moving the threshold, because it reshapes the precision-recall curve itself.

This version fixes that by reweighting the VALIDATION fold too, not just the calibration
prior: since the downsampling is a uniform random subsample (fixed seed, no
stratification -- see sample_primevul.py), every PrimeVul-safe row still present in a
validation fold is an unbiased draw from the larger population, so we can importance-
reweight it (via ``sample_weight`` in sklearn's precision_recall_curve) to simulate
evaluating against the reconstructed population, without needing new tool runs on
samples that were never scored. Both scenarios below apply the SAME reweighting
mechanism to calibration (for the prior) and to validation (for the metrics), using
consistent per-split scale factors, so the "hypothetical world" being simulated is the
same on both sides:

  original_ratio   -- PrimeVul-safe rows (calibration and validation, independently)
                       are reweighted by the real physical scale factor
                       193,047 / 10,000 = 19.30, reconstructing the pre-downsampling
                       population exactly.
  industry_0_1pct  -- PrimeVul-safe rows are reweighted by a factor solved, per
                       (split, family), to hit a target prevalence of 0.1%.

Per-tool log-likelihood-ratio terms (fnr/fpr, from calibration) are identical to
analysis/fusion/bayes.py's computation in every scenario -- only the prior and the
evaluation weights change, so any difference in results is attributable to the
class-balance correction alone, not to a different tool model.

Requires data/enriched/c_cpp.parquet and data/cwec_latest.xml (neither ships in the git
repo) and, for the exact original-PrimeVul-safe count, data/merged/c_cpp_merged_unsampled
.parquet (falls back to the documented count of 193,047 if that file is absent).

    uv run python auxiliary/primevul_prior_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.calibration import compute_reliability
from analysis.dataset import as_list
from analysis.experiment import prepare_canonical
from analysis.folds import split_train_validation
from analysis.fusion.common import build_fire_index, evidence_row, is_supported, metric_lookup, metric_value, precompute_labels, sample_ids_of

LANGUAGE = "c_cpp"
LEVEL = "pillar_child"
TIER = "full"
N_SPLITS = 5
SEED = 42
MIN_CWE_COUNT = 100  # matches TIER_MIN_COUNT["full"] in analysis/fusion/__init__.py

SAMPLED_PRIMEVUL_SAFE = 10_000
DEFAULT_ORIGINAL_PRIMEVUL_SAFE = 193_047
UNSAMPLED_PARQUET = ROOT / "data" / "merged" / "c_cpp_merged_unsampled.parquet"

INDUSTRY_PREVALENCE = 0.001
GRID_TAUS = tuple(round(0.1 * i, 1) for i in range(1, 10))  # the paper's own tau grid
EPS = 1e-3


def _clamp(x: float) -> float:
    return min(max(x, EPS), 1.0 - EPS)


def original_primevul_safe_count() -> int:
    if not UNSAMPLED_PARQUET.exists():
        print(f"[warn] {UNSAMPLED_PARQUET} not found; using documented count "
              f"{DEFAULT_ORIGINAL_PRIMEVUL_SAFE}")
        return DEFAULT_ORIGINAL_PRIMEVUL_SAFE
    unsampled = pd.read_parquet(UNSAMPLED_PARQUET, columns=["source", "label"])
    count = int(((unsampled["source"] == "PrimeVul") & (unsampled["label"] == 0)).sum())
    print(f"Verified original PrimeVul safe count from {UNSAMPLED_PARQUET.name}: {count:,}")
    return count


def primevul_safe_mask(df: pd.DataFrame) -> np.ndarray:
    return ((df["source"] == "PrimeVul") & (df["label"] == 0)).to_numpy()


def family_neg_pos_counts(df: pd.DataFrame, families: list[str]) -> dict[str, tuple[int, int, int]]:
    """Per family: (positive count, total negative count, PrimeVul-safe negative count).

    Every PrimeVul-safe row has label=0 and an empty ``cwes`` list (verified: 0/33,347
    label=0 rows have any CWE), so it is a negative for every family -- the boolean
    "is this row reweightable" does not depend on which family we are scoring.
    """
    cwe_sets = [set(as_list(v)) for v in df["cwes"]]
    is_pv_safe = primevul_safe_mask(df)
    counts: dict[str, tuple[int, int, int]] = {}
    for family in families:
        pos = sum(family in s for s in cwe_sets)
        neg_total = len(cwe_sets) - pos
        neg_pv_safe = int(is_pv_safe.sum())  # PrimeVul-safe rows are negative for every family
        counts[family] = (pos, neg_total, neg_pv_safe)
    return counts


def solve_scale_for_prevalence(pos: int, neg_total: int, neg_pv_safe: int, target: float) -> float:
    """Scale factor on PrimeVul-safe negatives that makes the family's weighted
    prevalence equal ``target``. No-op (scale=1) if there is no PrimeVul-safe row to
    reweight for this family -- the target then cannot be reached through this
    mechanism, and we leave the family's count untouched rather than distort other
    negatives that downsampling never touched."""
    if neg_pv_safe <= 0:
        return 1.0
    neg_other = neg_total - neg_pv_safe
    scale = (pos / target - pos - neg_other) / neg_pv_safe
    return max(scale, 0.0)


def reweighted_prior_logodds(counts: dict[str, tuple[int, int, int]], scale_by_family: dict[str, float]) -> dict[str, float]:
    priors = {}
    for family, (pos, neg_total, neg_pv_safe) in counts.items():
        neg_other = neg_total - neg_pv_safe
        neg_corrected = neg_other + neg_pv_safe * scale_by_family[family]
        total = pos + neg_corrected
        if total <= 0:
            priors[family] = 0.0
            continue
        priors[family] = float(np.log(_clamp(pos / total)) - np.log(_clamp(neg_corrected / total)))
    return priors


def naive_bayes_predictions_with_prior(
    df: pd.DataFrame,
    lookup: dict[tuple[str, str], dict],
    tools: list[str],
    families: list[str],
    labels: dict[tuple[int, str], bool],
    fire_index: tuple[dict[tuple[int, str], set[str]], list[set[str]]],
    prior_override: dict[str, float],
) -> pd.DataFrame:
    """Same score as analysis.fusion.bayes.naive_bayes_predictions, but with the
    per-family prior log-odds replaced by ``prior_override``."""
    fire_sets, _ = fire_index
    sample_ids = sample_ids_of(df)
    n = len(df)

    rows: list[dict] = []
    for family in families:
        sup, llr_fire, llr_silence = [], [], []
        for tool in tools:
            row = lookup.get((tool, family))
            if not is_supported(row):
                continue
            fnr = metric_value(row, "fnr") or 0.0
            fpr = metric_value(row, "fpr") or 0.0
            p_fire_v, p_fire_s = _clamp(1.0 - fnr), _clamp(fpr)
            p_sil_v, p_sil_s = _clamp(fnr), _clamp(1.0 - fpr)
            sup.append(tool)
            llr_fire.append(np.log(p_fire_v / p_fire_s))
            llr_silence.append(np.log(p_sil_v / p_sil_s))
        k = len(sup)
        abstained = len(tools) - k
        if k == 0:
            for ri in range(n):
                rows.append(evidence_row(sample_ids[ri], ri, family, "naive_bayes", False, 0.0, 0.0, 0.0, 0, 0, abstained, labels[(ri, family)]))
            continue

        prior_logodds = prior_override[family]
        fire = np.array([[family in fire_sets[(ri, tool)] for tool in sup] for ri in range(n)], dtype=bool)
        contrib = np.where(fire, np.asarray(llr_fire), np.asarray(llr_silence)).sum(axis=1)
        score = 1.0 / (1.0 + np.exp(-(prior_logodds + contrib)))
        fired_counts = fire.sum(axis=1)
        for ri in range(n):
            rows.append(evidence_row(
                sample_ids[ri], ri, family, "naive_bayes",
                prediction=bool(score[ri] >= 0.5), score=float(score[ri]),
                vuln=float(score[ri]), safe=float(1.0 - score[ri]),
                k_supported=k, k_fired=int(fired_counts[ri]), k_abstained=abstained,
                label=labels[(ri, family)],
            ))
    return pd.DataFrame(rows)


def weighted_f1(labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray) -> float:
    tp = weights[labels & predictions].sum()
    fp = weights[(~labels) & predictions].sum()
    fn = weights[labels & (~predictions)].sum()
    if tp + fp == 0 or tp + fn == 0:
        return float("nan")
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def weighted_best_f1_exhaustive(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.0
    precision, recall, _ = precision_recall_curve(labels, scores, sample_weight=weights)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1.0), 0.0)
    return float(f1.max())


def evaluate_weighted(
    predictions_by_fold: list[pd.DataFrame],
    weights_by_fold: list[dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per (family, tau) weighted F1, plus the exhaustive-threshold best F1, aggregated
    across folds via the paper's own two-stage protocol: mean over folds per family,
    then support-weighted (by true positive_support, unaffected by reweighting since
    only negatives are reweighted) mean over families.

    ``weights_by_fold[fold]`` maps family -> per-row weight array for that fold's
    validation split (one weight per row, indexed like the fold's original val_df --
    families can have different weights on the same row, e.g. in the industry_0_1pct
    scenario where the target-prevalence scale is solved per family)."""
    rows = []
    for fold, (predictions, weights_by_family) in enumerate(zip(predictions_by_fold, weights_by_fold, strict=True)):
        for family, group in predictions.groupby("family", sort=False):
            idx = group["row_index"].to_numpy()
            labels = group["label"].to_numpy(dtype=bool)
            scores = group["score"].to_numpy(dtype=float)
            w = weights_by_family[family][idx]
            record = {"fold": fold, "family": family, "positive_support": int(labels.sum()),
                      "f1_exhaustive": weighted_best_f1_exhaustive(labels, scores, w)}
            for tau in GRID_TAUS:
                record[f"f1_tau_{tau:.1f}"] = weighted_f1(labels, scores >= tau, w)
            rows.append(record)
    per_fold = pd.DataFrame(rows)
    metric_cols = ["f1_exhaustive"] + [f"f1_tau_{tau:.1f}" for tau in GRID_TAUS]
    per_family = per_fold.groupby("family", as_index=False).agg(
        positive_support=("positive_support", "sum"),
        **{c: (c, "mean") for c in metric_cols},
    )
    weights_arr = per_family["positive_support"].to_numpy(dtype=float)
    overall = {}
    for c in metric_cols:
        values = per_family[c].to_numpy(dtype=float)
        mask = ~np.isnan(values) & (weights_arr > 0)
        overall[c] = float(np.average(values[mask], weights=weights_arr[mask])) if mask.sum() > 0 else float("nan")
    return per_family, overall


def main() -> None:
    enriched_path = ROOT / "data" / "enriched" / f"{LANGUAGE}.parquet"
    if not enriched_path.exists():
        print(f"missing {enriched_path} -- run `main.py enrich` first.")
        return

    original_safe = original_primevul_safe_count()
    scale = original_safe / SAMPLED_PRIMEVUL_SAFE
    print(f"Downsampling scale factor (original / sampled PrimeVul safe): {scale:.4f}")

    prep = prepare_canonical(LANGUAGE, LEVEL, n_splits=N_SPLITS, min_cwe_count=MIN_CWE_COUNT, seed=SEED)
    cdf, tools, families, folds = prep.cdf, prep.tools, prep.families, prep.folds
    print(f"{LANGUAGE}/{LEVEL}/{TIER}: {len(families)} families, tools={tools}")

    scenario_preds: dict[str, list[pd.DataFrame]] = {"baseline": [], "original_ratio": [], "industry_0_1pct": []}
    scenario_weights: dict[str, list[dict[str, np.ndarray]]] = {"baseline": [], "original_ratio": [], "industry_0_1pct": []}

    for fold in sorted(folds["fold"].unique()):
        cal_df, val_df = split_train_validation(cdf, folds, validation_fold=fold)
        reliability = compute_reliability(cal_df, tools, families)
        lookup = metric_lookup(reliability)
        labels = precompute_labels(val_df, families)
        fire_index = build_fire_index(val_df, tools)

        cal_counts = family_neg_pos_counts(cal_df, families)
        val_counts = family_neg_pos_counts(val_df, families)
        val_pv_safe = primevul_safe_mask(val_df)

        # baseline: scale=1 everywhere (no reweighting) -- reproduces the released prior
        baseline_prior = reweighted_prior_logodds(cal_counts, {f: 1.0 for f in families})
        scenario_preds["baseline"].append(
            naive_bayes_predictions_with_prior(val_df, lookup, tools, families, labels, fire_index, baseline_prior))
        scenario_weights["baseline"].append({f: np.ones(len(val_df)) for f in families})

        # original_ratio: real physical scale factor, applied identically to both splits
        ratio_prior = reweighted_prior_logodds(cal_counts, {f: scale for f in families})
        scenario_preds["original_ratio"].append(
            naive_bayes_predictions_with_prior(val_df, lookup, tools, families, labels, fire_index, ratio_prior))
        ratio_weight = np.where(val_pv_safe, scale, 1.0)
        scenario_weights["original_ratio"].append({f: ratio_weight for f in families})

        # industry_0.1%: per-family scale solved independently on each split, both
        # targeting the same 0.1% prevalence
        cal_scale_by_family = {f: solve_scale_for_prevalence(*cal_counts[f], INDUSTRY_PREVALENCE) for f in families}
        val_scale_by_family = {f: solve_scale_for_prevalence(*val_counts[f], INDUSTRY_PREVALENCE) for f in families}
        industry_prior = reweighted_prior_logodds(cal_counts, cal_scale_by_family)
        scenario_preds["industry_0_1pct"].append(
            naive_bayes_predictions_with_prior(val_df, lookup, tools, families, labels, fire_index, industry_prior))
        scenario_weights["industry_0_1pct"].append(
            {f: np.where(val_pv_safe, val_scale_by_family[f], 1.0) for f in families})

    print("\n=== Weighted best F1 (support-weighted over families) ===")
    print("f1_exhaustive: unconstrained per-family threshold search under reweighted evaluation.")
    print(f"f1_tau_X: the paper's own fixed grid {GRID_TAUS}, also under reweighted evaluation.\n")

    summary_rows = []
    per_family_frames = []
    for name in ("baseline", "original_ratio", "industry_0_1pct"):
        per_family, overall = evaluate_weighted(scenario_preds[name], scenario_weights[name])
        per_family.insert(0, "scenario", name)
        per_family_frames.append(per_family)
        best_tau_col = max((c for c in overall if c.startswith("f1_tau_")), key=lambda c: overall[c])
        summary_rows.append({
            "scenario": name,
            "f1_exhaustive": overall["f1_exhaustive"],
            "best_grid_f1": overall[best_tau_col],
            "best_grid_tau": float(best_tau_col.rsplit("_", 1)[-1]),
        })

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    baseline_row = summary[summary.scenario == "baseline"].iloc[0]
    for _, r in summary.iterrows():
        if r["scenario"] == "baseline":
            continue
        print(f"\n{r['scenario']}: exhaustive-threshold F1 delta vs baseline = "
              f"{r['f1_exhaustive'] - baseline_row['f1_exhaustive']:+.4f}; "
              f"grid-best F1 delta = {r['best_grid_f1'] - baseline_row['best_grid_f1']:+.4f} "
              f"(tau {r['best_grid_tau']} vs {baseline_row['best_grid_tau']})")

    out_dir = ROOT / "data" / "results" / "_cross_language"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "primevul_prior_sensitivity_summary.csv", index=False)
    pd.concat(per_family_frames, ignore_index=True).to_csv(
        out_dir / "primevul_prior_sensitivity_per_family.csv", index=False)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
