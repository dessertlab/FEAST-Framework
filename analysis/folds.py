"""Multilabel-stratified k-fold splitting with rare-label pruning.

Operates on the row label column, which by the time folding runs already holds the
**canonical CWE families** (see ``analysis.canonical``), not raw leaf CWEs. The
stratifier therefore balances family occurrences across folds, and the rare-label
pruning drops families with fewer than K occurrences (the necessary condition for a
per-fold-stable confusion matrix).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from analysis.dataset import as_list, normalise_cwes

# Synthetic label attached to safe (non-vulnerable) rows. The multilabel
# stratifier only "sees" rows through their label set; without this marker every
# negative row would carry an *empty* set, become invisible to stratification,
# and could clump into a few folds — unbalancing the safe/vuln ratio per fold.
SAFE_LABEL = "SAFE"


def row_multilabels(value, safe_label: str = SAFE_LABEL) -> set[str]:
    """Return the CWE label set of one row, or {SAFE} for a safe row."""
    labels = normalise_cwes(as_list(value))
    return labels or {safe_label}


# ── rare-CWE pruning ──────────────────────────────────────────────────────────
# Why prune before folding: with K folds, a CWE that occurs in fewer than K rows
# *cannot* appear in every validation fold. In the folds where it is absent its
# confusion matrix has zero positives, so PPV/NPV are undefined and the per-fold
# reliability estimate degenerates. Requiring count >= K (the configured
# minimum, default = number of folds) is the NECESSARY condition for a per-fold
# stable estimate. It is not *sufficient* — the stratifier may still fail to put
# a borderline CWE in every fold — but it removes the guaranteed-degenerate
# cases up front. Counting is EXACT: a CWE counts only when it literally appears
# in a row's ground-truth set (no vertical-closure expansion), which is the same
# notion the stratifier balances on.


def cwe_support_counts(
    labels_by_row: list[set[str]],
    safe_label: str = SAFE_LABEL,
) -> Counter:
    """Exact per-CWE occurrence count across rows (the SAFE marker excluded)."""
    counts: Counter = Counter()
    for labels in labels_by_row:
        for label in labels:
            if label != safe_label:
                counts[label] += 1
    return counts


def kept_cwes_by_min_count(
    labels_by_row: list[set[str]],
    min_count: int,
    safe_label: str = SAFE_LABEL,
) -> set[str]:
    """CWEs whose exact occurrence count is at least ``min_count``.

    This is the single source of truth for "which CWEs survive the rare-CWE
    filter": it is consumed both by the fold assignment (stratification labels)
    and by the evaluation-grid builder, so the two never disagree.
    """
    counts = cwe_support_counts(labels_by_row, safe_label=safe_label)
    return {cwe for cwe, n in counts.items() if n >= min_count}


def kept_cwes_for_dataframe(
    df: pd.DataFrame,
    min_count: int,
    label_column: str = "cwes",
    safe_label: str = SAFE_LABEL,
) -> set[str]:
    """Convenience wrapper: surviving CWEs computed directly from a DataFrame.

    Used by the fusion grid builder so it prunes exactly the same CWEs the
    folding step prunes, without re-deriving the counting logic.
    """
    labels_by_row = [row_multilabels(v, safe_label=safe_label) for v in df[label_column]]
    return kept_cwes_by_min_count(labels_by_row, min_count, safe_label=safe_label)


def prune_rare_cwes(
    labels_by_row: list[set[str]],
    min_count: int,
    safe_label: str = SAFE_LABEL,
) -> tuple[list[set[str]], set[str]]:
    """Drop rare CWEs from each row's *stratification* label set.

    Returns ``(pruned_labels_by_row, kept_cwes)``. A row whose labels are all
    pruned collapses to {SAFE} — but only for the purpose of balancing folds.
    The row's real ground truth (used everywhere else) is never modified here.
    """
    kept = kept_cwes_by_min_count(labels_by_row, min_count, safe_label=safe_label)
    pruned: list[set[str]] = []
    for labels in labels_by_row:
        surviving = {l for l in labels if l == safe_label or l in kept}
        pruned.append(surviving or {safe_label})
    return pruned, kept


# ── fold assignment ───────────────────────────────────────────────────────────


def _label_matrix(labels_by_row: list[set[str]]) -> tuple[list[str], np.ndarray]:
    """One-hot binary indicator matrix (rows × distinct labels) for iterstrat."""
    labels = sorted({label for row_labels in labels_by_row for label in row_labels})
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels_by_row), len(labels)), dtype=int)
    for row_id, row_labels in enumerate(labels_by_row):
        for label in row_labels:
            matrix[row_id, index[label]] = 1
    return labels, matrix


def _iterstrat_folds(
    labels_by_row: list[set[str]],
    n_splits: int,
    random_state: int,
) -> list[int]:
    """Assign each row to a fold via MultilabelStratifiedKFold.

    Exact multilabel stratification is NP-hard; iterstrat implements the
    well-known iterative heuristic (Sechidis et al., 2011), which spreads each
    label as evenly as possible across folds. It is a hard dependency on
    purpose: keeping a single implementation makes the fold assignment fully
    reproducible from (data, n_splits, random_state) alone.
    """
    _labels, y = _label_matrix(labels_by_row)
    # iterstrat ignores feature values; a single dummy column is enough.
    x = np.zeros((len(labels_by_row), 1))
    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    folds = [0] * len(labels_by_row)
    for fold, (_train_idx, test_idx) in enumerate(splitter.split(x, y)):
        for row_id in test_idx:
            folds[int(row_id)] = fold
    return folds


def multilabel_stratified_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    label_column: str = "cwes",
    id_column: str = "sample_id",
    safe_label: str = SAFE_LABEL,
    random_state: int = 0,
    min_cwe_count: int = 1,
) -> pd.DataFrame:
    """Assign rows to multilabel-stratified folds using their CWE labels.

    Parameters
    ----------
    n_splits:
        Number of cross-validation folds (>= 2).
    min_cwe_count:
        Rare-CWE pruning threshold. CWEs with fewer than this many exact
        occurrences are removed from the stratification labels (see
        ``prune_rare_cwes``). ``1`` disables pruning; callers typically pass
        ``n_splits`` so every stratified CWE can in principle reach all folds.

    Returns a DataFrame with columns ``[sample_id?, row_index, fold]``.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if len(df) < n_splits:
        raise ValueError("n_splits cannot exceed the number of rows")
    if label_column not in df.columns:
        raise ValueError(f"missing label column: {label_column}")

    labels_by_row = [row_multilabels(value, safe_label=safe_label) for value in df[label_column]]
    if min_cwe_count > 1:
        labels_by_row, _kept = prune_rare_cwes(labels_by_row, min_cwe_count, safe_label=safe_label)

    folds = _iterstrat_folds(labels_by_row, n_splits, random_state)

    out = pd.DataFrame({"row_index": list(range(len(df))), "fold": folds})
    if id_column in df.columns:
        out.insert(0, id_column, df[id_column].astype(str).tolist())
    return out


def split_train_validation(
    df: pd.DataFrame,
    folds: pd.DataFrame,
    validation_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (calibration, validation) for one held-out fold.

    Calibration = all rows NOT in ``validation_fold`` (used to estimate per-tool
    reliability); validation = the held-out fold (used to score fusion). The
    join is positional via ``row_index`` against ``df`` reset to a clean index.
    """
    if "row_index" not in folds.columns or "fold" not in folds.columns:
        raise ValueError("folds must contain row_index and fold columns")
    fold_map = folds.set_index("row_index")["fold"]
    aligned = df.copy().reset_index(drop=True)
    fold_values = aligned.index.to_series().map(fold_map)
    if fold_values.isna().any():
        raise ValueError("fold assignment is missing rows from the dataframe")
    validation_mask = fold_values.astype(int) == int(validation_fold)
    return (
        aligned[~validation_mask].reset_index(drop=True),
        aligned[validation_mask].reset_index(drop=True),
    )
