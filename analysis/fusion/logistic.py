"""Per-family logistic regression over the tools' fire indicators.

Unlike the metrics-only strategies this one *learns* from the calibration split: features
are the per-tool fire flags, the target is the calibration family label.
``class_weight='balanced'`` counters the heavy class imbalance. A family whose calibration
labels are single-class can't be fit, so it falls back to a constant equal to that rate.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from analysis.fusion.common import (
    build_fire_index,
    evidence_row,
    precompute_labels,
    sample_ids_of,
)


def _add_interactions(x: np.ndarray) -> np.ndarray:
    """Append pairwise products of the fire indicators (the A∧B synergy terms)."""
    k = x.shape[1]
    extra = [x[:, [a]] * x[:, [b]] for a in range(k) for b in range(a + 1, k)]
    return np.hstack([x, *extra]) if extra else x


def logistic_regression_predictions(
    calibration_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    tools: Sequence[str],
    families: Sequence[str],
    labels: dict[tuple[int, str], bool],
    fire_index: tuple[dict[tuple[int, str], set[str]], list[set[str]]],
    tau: float = 0.5,
    seed: int = 0,
    interactions: bool = False,
) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression

    cal_fire, _ = build_fire_index(calibration_df, tools)
    val_fire, _ = fire_index
    cal_labels = precompute_labels(calibration_df, families)
    sample_ids = sample_ids_of(validation_df)
    n_cal, n_val = len(calibration_df), len(validation_df)
    k = len(tools)
    name = "logistic_interactions" if interactions else "logistic_regression"

    rows: list[dict] = []
    for family in families:
        x_cal = np.array([[family in cal_fire[(i, t)] for t in tools] for i in range(n_cal)], dtype=float)
        y_cal = np.array([cal_labels[(i, family)] for i in range(n_cal)], dtype=int)
        x_val = np.array([[family in val_fire[(i, t)] for t in tools] for i in range(n_val)], dtype=float)
        fired_counts = x_val.sum(axis=1).astype(int)
        if interactions:
            x_cal, x_val = _add_interactions(x_cal), _add_interactions(x_val)

        if len(np.unique(y_cal)) < 2:
            score = np.full(n_val, float(y_cal.mean()) if n_cal else 0.0)
        else:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
            clf.fit(x_cal, y_cal)
            score = clf.predict_proba(x_val)[:, 1]

        for ri in range(n_val):
            rows.append(evidence_row(
                sample_ids[ri], ri, family, name,
                prediction=bool(score[ri] >= tau), score=float(score[ri]),
                vuln=float(score[ri]), safe=float(1.0 - score[ri]),
                k_supported=k, k_fired=int(fired_counts[ri]), k_abstained=0,
                label=labels[(ri, family)],
            ))
    return pd.DataFrame(rows)
