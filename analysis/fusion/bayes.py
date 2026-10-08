"""Naive Bayes fusion.

Score = prior log-odds + sum of per-tool log-likelihood ratios, passed through a sigmoid.
Likelihoods come from the calibration confusion matrix: ``P(fire|V)=1-fnr``,
``P(fire|S)=fpr`` (and complements for silence). The prior odds use the family's own
positive rate in calibration. Probabilities are clamped to keep the logs finite.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from analysis.fusion.common import evidence_row, is_supported, metric_value, sample_ids_of


def naive_bayes_predictions(
    df: pd.DataFrame,
    lookup: dict[tuple[str, str], dict],
    tools: Sequence[str],
    families: Sequence[str],
    labels: dict[tuple[int, str], bool],
    fire_index: tuple[dict[tuple[int, str], set[str]], list[set[str]]],
    tau: float = 0.5,
    eps: float = 1e-3,
) -> pd.DataFrame:
    fire_sets, _ = fire_index
    sample_ids = sample_ids_of(df)
    n = len(df)

    def clamp(x: float) -> float:
        return min(max(x, eps), 1.0 - eps)

    rows: list[dict] = []
    for family in families:
        sup, llr_fire, llr_silence = [], [], []
        prior_logodds = 0.0
        for tool in tools:
            row = lookup.get((tool, family))
            if not is_supported(row):
                continue
            fnr = metric_value(row, "fnr") or 0.0
            fpr = metric_value(row, "fpr") or 0.0
            p_fire_v, p_fire_s = clamp(1.0 - fnr), clamp(fpr)
            p_sil_v, p_sil_s = clamp(fnr), clamp(1.0 - fpr)
            sup.append(tool)
            llr_fire.append(np.log(p_fire_v / p_fire_s))
            llr_silence.append(np.log(p_sil_v / p_sil_s))
            pos = float(row["tp"]) + float(row["fn"])
            neg = float(row["tn"]) + float(row["fp"])
            total = pos + neg
            if total > 0:
                prior_logodds = np.log(clamp(pos / total) / clamp(neg / total))
        k = len(sup)
        abstained = len(tools) - k
        if k == 0:
            for ri in range(n):
                rows.append(evidence_row(sample_ids[ri], ri, family, "naive_bayes", False, 0.0, 0.0, 0.0, 0, 0, abstained, labels[(ri, family)]))
            continue

        fire = np.array([[family in fire_sets[(ri, tool)] for tool in sup] for ri in range(n)], dtype=bool)
        contrib = np.where(fire, np.asarray(llr_fire), np.asarray(llr_silence)).sum(axis=1)
        score = 1.0 / (1.0 + np.exp(-(prior_logodds + contrib)))
        fired_counts = fire.sum(axis=1)
        for ri in range(n):
            rows.append(evidence_row(
                sample_ids[ri], ri, family, "naive_bayes",
                prediction=bool(score[ri] >= tau), score=float(score[ri]),
                vuln=float(score[ri]), safe=float(1.0 - score[ri]),
                k_supported=k, k_fired=int(fired_counts[ri]), k_abstained=abstained,
                label=labels[(ri, family)],
            ))
    return pd.DataFrame(rows)
