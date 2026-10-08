"""Per-(tool, family) reliability calibration with EXACT family matching.

This replaces the old permissive (vertical-closure) matcher. By the time we get here
both the ground-truth ``cwes`` column and every tool column hold *canonical families*
(see ``analysis.canonical``), so a tool "fires" a family iff the family is literally
in its set, and the ground truth is positive for a family iff the family is literally
in the GT set. Matching is therefore plain set membership — exact, symmetric, and
identical between calibration and fusion.

For each ``(tool, family)`` we build the binary confusion matrix one-vs-rest over the
calibration rows and derive the reliability metrics the fusion strategies consume:
``ppv`` (fire credibility), ``npv`` (silence credibility), ``fpr``/``fnr`` and their
complements, plus ``supported`` (the tool fired the family at least once, ``tp+fp>0``).
The computation is vectorised over families via boolean incidence matrices.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from analysis.dataset import as_list

RELIABILITY_COLUMNS = [
    "tool", "family",
    "tp", "fp", "tn", "fn",
    "ppv", "npv", "fpr", "fnr", "sensitivity", "specificity",
    "positive_support", "supported",
]


def _family_sets(series: pd.Series) -> list[set[str]]:
    """Each cell (a list of family IDs) as a set, for O(1) membership tests."""
    return [set(as_list(values)) for values in series]


def _incidence(family_sets: Sequence[set[str]], family_index: dict[str, int], n_families: int) -> np.ndarray:
    """Boolean [n_rows, n_families] matrix: entry True iff family present in the row's set."""
    matrix = np.zeros((len(family_sets), n_families), dtype=bool)
    for row_id, families in enumerate(family_sets):
        for family in families:
            col = family_index.get(family)
            if col is not None:
                matrix[row_id, col] = True
    return matrix


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise division returning NaN where the denominator is zero."""
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out


def gt_family_support(df: pd.DataFrame, families: Iterable[str]) -> dict[str, int]:
    """Ground-truth positive support per family (rows where the family is a GT label).

    Used both for the rare-family restriction and as the weight in the support-weighted
    aggregation. Counted on whatever rows ``df`` contains (full dataset when called once).
    """
    wanted = set(families)
    counts: Counter = Counter()
    for values in df["cwes"]:
        for family in set(as_list(values)):
            if family in wanted:
                counts[family] += 1
    return {family: counts.get(family, 0) for family in wanted}


def compute_reliability(
    df: pd.DataFrame,
    tools: Sequence[str],
    families: Sequence[str],
) -> pd.DataFrame:
    """Confusion matrix + reliability metrics for every (tool, family) pair.

    ``df`` must already be canonicalised (GT + tool columns hold family IDs). Returns a
    long DataFrame with one row per (tool, family), schema ``RELIABILITY_COLUMNS``.
    """
    n = len(df)
    family_index = {family: j for j, family in enumerate(families)}
    n_families = len(families)

    gt_mat = _incidence(_family_sets(df["cwes"]), family_index, n_families)

    records: list[dict] = []
    for tool in tools:
        source = df[tool] if tool in df.columns else pd.Series([[]] * n)
        tool_mat = _incidence(_family_sets(source), family_index, n_families)

        tp = (tool_mat & gt_mat).sum(axis=0)
        fp = (tool_mat & ~gt_mat).sum(axis=0)
        fn = (~tool_mat & gt_mat).sum(axis=0)
        tn = n - tp - fp - fn

        ppv = _safe_div(tp, tp + fp)            # fire credibility  (precision)
        npv = _safe_div(tn, tn + fn)            # silence credibility
        fpr = _safe_div(fp, fp + tn)
        fnr = _safe_div(fn, fn + tp)
        sensitivity = _safe_div(tp, tp + fn)    # recall = 1 - fnr
        specificity = _safe_div(tn, tn + fp)    # 1 - fpr

        for j, family in enumerate(families):
            records.append({
                "tool": tool, "family": family,
                "tp": int(tp[j]), "fp": int(fp[j]), "tn": int(tn[j]), "fn": int(fn[j]),
                "ppv": ppv[j], "npv": npv[j], "fpr": fpr[j], "fnr": fnr[j],
                "sensitivity": sensitivity[j], "specificity": specificity[j],
                "positive_support": int(tp[j] + fn[j]),
                "supported": bool(tp[j] + fp[j] > 0),
            })
    return pd.DataFrame(records, columns=RELIABILITY_COLUMNS)
