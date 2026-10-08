"""Scoring of fusion predictions.

Two views:
* ``evaluate_predictions`` — per-(strategy, family) (or per-strategy) classification metrics
  from the prediction/label columns.
* ``detection_from_predictions`` — collapses the per-family grid to the coarser per-row
  question "did the ensemble flag this row as vulnerable at all?", plus the conditional
  CWE/family-attribution accuracy (when a correct detection also names a right family).
"""

from __future__ import annotations

from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from analysis.dataset import as_list
from analysis.fusion.common import DEFAULT_TAUS, split_tau_strategy, tau_suffix

DETECTION_STRATEGY_COL = "strategy"


def _safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def _roc_auc(labels: list[bool], scores: list[float]) -> float | None:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg_rank
        i = j
    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=False) if label)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision(labels: list[bool], scores: list[float]) -> float | None:
    """Average precision (PR-AUC), delegated to sklearn for correct tie handling.

    A hand-rolled version of this (rank-by-score, average precision at each positive)
    silently mis-handles score ties: with a plain stable sort, tied samples are broken
    by incidental row order rather than treated as a single block, making the result
    order-dependent. Ties are the norm here, not an edge case -- every fusion score is a
    deterministic function of an at-most-2^N-valued binary tool-fire vector (N <= 6), so
    many same-family samples share an identical score. sklearn's implementation groups
    tied scores correctly.
    """
    n_pos = sum(labels)
    if n_pos == 0:
        return None
    return float(average_precision_score(labels, scores))


def binary_metrics(labels: list[bool], predictions: list[bool], scores: list[float]) -> dict:
    """Full classification metric set for one group of (label, prediction, score)."""
    tp = sum(l and p for l, p in zip(labels, predictions, strict=False))
    fp = sum((not l) and p for l, p in zip(labels, predictions, strict=False))
    tn = sum((not l) and (not p) for l, p in zip(labels, predictions, strict=False))
    fn = sum(l and (not p) for l, p in zip(labels, predictions, strict=False))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    npv = _safe_div(tn, tn + fn)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    accuracy = _safe_div(tp + tn, tp + fp + tn + fn)
    balanced_accuracy = None if recall is None or specificity is None else (recall + specificity) / 2

    def f_beta(beta: float) -> float | None:
        if precision is None or recall is None:
            return None
        b2 = beta * beta
        den = b2 * precision + recall
        return None if den == 0 else (1 + b2) * precision * recall / den

    mcc_den = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = None if mcc_den == 0 else ((tp * tn) - (fp * fn)) / mcc_den

    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "support": int(len(labels)),
        "positive_support": int(sum(labels)),
        "negative_support": int(len(labels) - sum(labels)),
        "precision": precision, "recall": recall, "specificity": specificity, "npv": npv,
        "fpr": fpr, "fnr": fnr, "accuracy": accuracy, "balanced_accuracy": balanced_accuracy,
        "f1": f_beta(1.0), "f2": f_beta(2.0), "mcc": mcc,
        "roc_auc": _roc_auc(labels, scores),
        "pr_auc": _average_precision(labels, scores),
    }


def evaluate_predictions(predictions: pd.DataFrame, group_cols: Iterable[str] = ("strategy", "family")) -> pd.DataFrame:
    """Compute classification metrics for each group of predictions."""
    group_cols = list(group_cols)
    rows: list[dict] = []
    for keys, group in predictions.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(group_cols, keys))
        record.update(binary_metrics(
            group["label"].astype(bool).tolist(),
            group["prediction"].astype(bool).tolist(),
            group["score"].astype(float).tolist(),
        ))
        rows.append(record)
    return pd.DataFrame(rows)


def score_histogram(predictions: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-row scores to the sufficient statistic for any threshold metric.

    The pipeline does not persist per-row scores: for C/C++ that would be ~14M rows per
    language, and every downstream question about thresholds -- the PR curves, a tau
    sweep, whether any operating point dominates the baseline -- therefore costs a full
    re-run of calibration and fusion just to recover numbers the run already had.

    It does not need to. Every threshold metric is a function of (TP, FP, FN) at a cut,
    and those follow from counting, per distinct score, how many rows carry it and how
    many of those are positive. Since a fusion score is a deterministic function of an
    at-most-2^N-valued fire vector, the distinct scores per (strategy, family) number in
    the tens, so this table is four orders of magnitude smaller than the rows it
    summarises and loses nothing that a threshold can ask.

    Returns one row per (strategy, family, score) with ``n`` rows carrying that score and
    ``n_positive`` of them labelled vulnerable. Tau-suffixed variants are collapsed to
    their base strategy: expanding tau changes the decision, never the score.
    """
    frame = predictions.copy()
    frame["strategy"] = frame["strategy"].map(lambda name: split_tau_strategy(str(name))[0])
    frame["label"] = frame["label"].astype(bool)
    out = (
        frame.groupby(["strategy", "family", "score"], as_index=False, sort=True)
        .agg(n=("label", "size"), n_positive=("label", "sum"))
    )
    out["n_positive"] = out["n_positive"].astype(int)
    return out


def metrics_from_histogram(histogram: pd.DataFrame, taus: Iterable[float]) -> pd.DataFrame:
    """Per-(strategy, family, tau) metrics recovered from ``score_histogram`` alone.

    Exact, not approximate: at threshold tau the positives are every score >= tau, so the
    counts come straight from the histogram. This is what makes the persisted table a
    replacement for the per-row scores rather than a summary of them.
    """
    rows: list[dict] = []
    for (strategy, family), group in histogram.groupby(["strategy", "family"], sort=False):
        scores = group["score"].to_numpy(dtype=float)
        n = group["n"].to_numpy(dtype=float)
        pos = group["n_positive"].to_numpy(dtype=float)
        total_positive = float(pos.sum())
        for tau in taus:
            fired = scores >= tau
            tp = float(pos[fired].sum())
            predicted = float(n[fired].sum())
            precision = tp / predicted if predicted > 0 else 0.0
            recall = tp / total_positive if total_positive > 0 else float("nan")
            den = precision + recall
            rows.append({
                "strategy": strategy, "family": family, "tau": float(tau),
                "precision": precision, "recall": recall,
                "f1": (2 * precision * recall / den) if den > 0 else 0.0,
                "positive_support": int(total_positive),
            })
    return pd.DataFrame(rows)


def detection_from_predictions(predictions: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-(row, family) predictions to a per-(strategy, row) detection task.

    A row is predicted vulnerable when the strategy fires *any* tracked family on it; the
    label is the dataset's own row label (``label == 1``). ``cwe_correct`` records, for
    detected rows, whether any fired family is actually in the GT family set (exact) —
    enabling the conditional attribution accuracy.
    """
    df = df.reset_index(drop=True)
    row_labels = df["label"].astype(int).tolist() if "label" in df.columns else None
    gt = [set(as_list(df.at[i, "cwes"])) if "cwes" in df.columns else set() for i in range(len(df))]

    rows: list[dict] = []
    for (strategy, row_index), group in predictions.groupby(["strategy", "row_index"], sort=False):
        ri = int(row_index)
        fired = group.loc[group["prediction"].astype(bool), "family"].tolist()
        predicted = len(fired) > 0
        score = float(group["score"].max()) if "score" in group.columns else (1.0 if predicted else 0.0)
        det_label = bool(row_labels[ri] == 1) if row_labels is not None else bool(group["label"].any())
        cwe_correct = None if not predicted else any(f in gt[ri] for f in fired)
        rows.append({
            "strategy": strategy, "row_index": ri,
            "prediction": predicted, "label": det_label, "score": score,
            "cwe_correct": cwe_correct,
        })
    return pd.DataFrame(rows)


def expand_tau_variants(predictions: pd.DataFrame, taus=DEFAULT_TAUS) -> pd.DataFrame:
    """Materialise each scored strategy as one explicit strategy per threshold.

    ``foo`` becomes ``foo_tau_0_1`` ... ``foo_tau_0_9``. The continuous score is preserved
    for ROC/PR metrics; only the boolean decision and strategy name change.
    """
    taus = DEFAULT_TAUS if taus is None else tuple(taus)
    frames = []
    for tau in taus:
        frame = predictions.copy()
        frame["strategy"] = frame["strategy"].map(lambda name: f"{name}_{tau_suffix(tau)}")
        frame["prediction"] = frame["score"].astype(float) >= float(tau)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else predictions.iloc[0:0].copy()


def tau_sweep_detection(predictions: pd.DataFrame, df: pd.DataFrame, taus=DEFAULT_TAUS) -> pd.DataFrame:
    """Re-threshold each strategy's continuous score at a discrete grid of τ and score the
    per-row vuln/safe detection task. Returns one row per (strategy, τ). A row is predicted
    vulnerable when the max family score on it reaches τ. The trivial ``always_vulnerable``
    predictor is added once as a degeneracy reference (high F2 on a positive-majority set).
    """
    df = df.reset_index(drop=True)
    n = len(df)
    label = (df["label"].astype(int) == 1).to_numpy() if "label" in df.columns else np.zeros(n, bool)

    rows: list[dict] = []
    keep = ("precision", "recall", "f1", "f2", "mcc", "roc_auc")
    for strategy, group in predictions.groupby("strategy", sort=False):
        row_score = (group.groupby("row_index")["score"].max()
                     .reindex(range(n)).fillna(0.0).to_numpy())
        for tau in taus:
            m = binary_metrics(label.tolist(), (row_score >= tau).tolist(), row_score.tolist())
            rows.append({"strategy": strategy, "tau": tau, **{k: m[k] for k in keep}})

    m = binary_metrics(label.tolist(), [True] * n, [1.0] * n)
    rows.append({"strategy": "always_vulnerable", "tau": 0.0, **{k: m[k] for k in keep}})
    return pd.DataFrame(rows)


def tau_variant_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add ``base_strategy`` and ``tau`` columns for explicit ``*_tau_*`` variants."""
    if metrics.empty or "strategy" not in metrics.columns:
        return metrics.copy()
    out = metrics.copy()
    parts = out["strategy"].map(split_tau_strategy)
    out.insert(1, "base_strategy", parts.map(lambda item: item[0]))
    out.insert(2, "tau", parts.map(lambda item: item[1]))
    return out[out["tau"].notna()].reset_index(drop=True)


def detection_metrics(detection: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy detection metrics + conditional family-attribution accuracy."""
    metrics = evaluate_predictions(detection, group_cols=("strategy",))
    true_pos = detection[detection["prediction"].astype(bool) & detection["label"].astype(bool)].copy()
    true_pos["cwe_correct"] = true_pos["cwe_correct"].astype(float)
    attribution = (
        true_pos.groupby("strategy")["cwe_correct"].mean()
        .rename("cwe_attribution_acc").reset_index()
    )
    return metrics.merge(attribution, on="strategy", how="left")
