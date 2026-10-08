"""Two-stage aggregation of per-family fusion metrics.

The experiment produces, for every fold, a table of metrics per ``(strategy, family)``.
The headline numbers are built in two explicit stages, as required:

* **Stage 1 — mean over folds**: for each ``(strategy, family)`` average every metric
  across the K folds. Family support (``positive_support``) is *summed* across folds, so
  it equals the family's total ground-truth occurrences in the dataset.
* **Stage 2 — support-weighted mean over families**: for each ``(strategy, metric)`` take
  the mean across families weighted by that family support. This is the headline value:
  it reflects how the ensemble does on the bulk of real positives. The unweighted macro
  mean and the median are kept alongside as context (a rare family weighs the same as a
  common one in the macro mean; the median is robust to the long tail).

Missing-value convention
------------------------
``binary_metrics`` returns None (-> NaN) for a metric whose defining ratio is 0/0. Two
different situations produce that NaN, and they must not be handled the same way:

* **Strategy failure** — the NaN is caused by the strategy's own degenerate prediction
  vector: zero positive predictions (precision, and hence F1/F2), or a constant
  prediction vector (MCC), or an all-positive vector (NPV). "No detection" is a failure,
  not a missing observation, so the standard zero-division convention scores it 0.0.
  These are ``ZERO_FILL_METRICS``, and ``zero_fill`` is applied to them at *every*
  aggregation site (both stages here, and the tau selection and paired differences in
  ``analysis.mean_difference_ci``) so that a strategy abandoning a family is penalised
  consistently rather than silently dropped from its own average. Without this, families
  a strategy fails to cover fall out of its mean instead of lowering it -- and since
  different strategies (notably weak baselines) fail on different families, the
  comparison is biased inconsistently, which can flip the sign of a weighted difference
  that is positive for every family pairwise (see git history for a worked example on
  the 2ooN baseline, which abstains entirely on several CWE families).

* **Degenerate group** — the NaN is a property of the (family, fold) slice itself, not of
  the strategy: recall/FNR are 0/0 when the slice holds no positives, specificity/FPR
  when it holds no negatives, and ROC-AUC/PR-AUC need both. Every strategy is NaN there
  simultaneously, so there is nothing to penalise; filling with 0.0 would invent a score
  that no strategy earned (and for the lower-is-better FPR/FNR it would invent a
  *perfect* one). These metrics stay NaN and are dropped from the reduction.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

# Classification metrics aggregated for the report (per-family columns produced by
# analysis.fusion.predictions.binary_metrics).
REPORT_METRICS = (
    "precision", "recall", "specificity", "npv", "fpr", "fnr",
    "f1", "f2", "mcc", "accuracy", "balanced_accuracy", "roc_auc", "pr_auc",
)
# Metrics whose NaN means "the strategy made no (or a constant) positive prediction".
# See the module docstring: these are zero-filled everywhere, the others are not.
ZERO_FILL_METRICS = frozenset({"precision", "npv", "f1", "f2", "mcc"})
SUPPORT_COLUMN = "positive_support"


def zero_fill(values: "pd.Series", metric: str) -> "pd.Series":
    """Apply the shared missing-value convention to one metric column.

    ``ZERO_FILL_METRICS`` are filled with 0.0 (strategy failure); every other metric is
    returned unchanged, NaN included (degenerate group). Always use this instead of a
    bare ``fillna(0.0)`` so the convention stays defined in exactly one place.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.fillna(0.0) if metric in ZERO_FILL_METRICS else numeric


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def mean_over_folds(
    per_family_per_fold: pd.DataFrame,
    metric_columns: Iterable[str] = REPORT_METRICS,
) -> pd.DataFrame:
    """Stage 1: average metrics across folds per (strategy, family); sum the support.

    ``ZERO_FILL_METRICS`` are zero-filled *before* the mean, so a fold in which the
    strategy is silent on a family counts as 0.0 rather than dropping out of that
    family's mean (pandas' groupby mean is skipna). The remaining metrics keep the skipna
    behaviour: a fold whose slice is degenerate for them carries no information about any
    strategy, so averaging over the folds where it is defined is the right reduction.
    """
    metrics = [m for m in metric_columns if m in per_family_per_fold.columns]
    frame = per_family_per_fold.copy()
    for metric in metrics:
        frame[metric] = zero_fill(frame[metric], metric)
    agg = {m: "mean" for m in metrics}
    if SUPPORT_COLUMN in frame.columns:
        agg[SUPPORT_COLUMN] = "sum"  # total GT occurrences across the dataset
    out = (
        frame.groupby(["strategy", "family"], as_index=False, sort=True)
        .agg(agg)
    )
    return out


def support_weighted_over_families(
    per_family: pd.DataFrame,
    metric_columns: Iterable[str] = REPORT_METRICS,
    support_column: str = SUPPORT_COLUMN,
) -> pd.DataFrame:
    """Stage 2: per strategy, reduce families to weighted/macro/median per metric."""
    metrics = [m for m in metric_columns if m in per_family.columns]
    has_weight = support_column in per_family.columns

    rows: list[dict] = []
    for strategy, group in per_family.groupby("strategy", dropna=False, sort=True):
        record: dict = {"strategy": strategy, "n_families": int(len(group))}
        weights = pd.to_numeric(group[support_column], errors="coerce") if has_weight else None
        for metric in metrics:
            values = zero_fill(group[metric], metric)
            record[f"{metric}_macro"] = _safe_float(values.mean())
            record[f"{metric}_median"] = _safe_float(values.median())
            if weights is None:
                record[f"{metric}_weighted"] = None
                continue
            mask = values.notna() & weights.notna() & (weights > 0)
            total = float(weights[mask].sum())
            record[f"{metric}_weighted"] = (
                _safe_float((values[mask] * weights[mask]).sum() / total) if total > 0 else None
            )
        rows.append(record)
    return pd.DataFrame(rows)


def aggregate_fusion_metrics(
    per_family_per_fold: pd.DataFrame,
    metric_columns: Iterable[str] = REPORT_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both stages; return (per_family_mean_over_folds, per_strategy_overall)."""
    per_family = mean_over_folds(per_family_per_fold, metric_columns)
    overall = support_weighted_over_families(per_family, metric_columns)
    return per_family, overall
