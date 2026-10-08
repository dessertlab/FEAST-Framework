"""Dempster-Shafer fusion on the binary frame Theta = {V, S}.

Each supported tool's basic belief assignment comes from a chosen pair of calibration
metrics: on a fire ``m({V}) = fire_metric`` (rest is ignorance ``m(Theta)``); on silence
``m({S}) = silence_metric``. The experiment expands the same metric pairs used by
weighted voting. Masses are combined with Dempster's rule, PCR6, or Yager's rule, and the
decision uses the pignistic probability ``BetP(V) = m({V}) + m(Theta)/2`` as the score.
"""

from __future__ import annotations

from itertools import product as _iproduct
from typing import Sequence

import numpy as np
import pandas as pd

from analysis.fusion.common import evidence_row, is_supported, metric_value, sample_ids_of


def _combine_dempster(v: np.ndarray, s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dempster's rule on the binary frame, vectorised over rows."""
    prod_vt = np.prod(v + t, axis=1)
    prod_st = np.prod(s + t, axis=1)
    prod_t = np.prod(t, axis=1)
    mV, mS, mT = prod_vt - prod_t, prod_st - prod_t, prod_t
    norm = mV + mS + mT  # = 1 - conflict
    with np.errstate(invalid="ignore", divide="ignore"):
        mV = np.where(norm > 0, mV / norm, 0.0)
        mS = np.where(norm > 0, mS / norm, 0.0)
    return mV, mS


def _combine_pcr6(v: np.ndarray, s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCR6 on the binary frame, fusing all sources simultaneously.

    The conjunctive core is kept; each partial conflict (a tuple mixing at least one {V}
    and one {S}) is redistributed to {V}/{S} proportionally to the summed masses each side
    committed. Non-associative, so enumerate the 3^k focal combinations (k = supported
    tools); cheap on the binary frame.
    """
    n, k = v.shape
    masses = (v, s, t)  # 0 -> {V}, 1 -> {S}, 2 -> Theta
    mV = np.zeros(n)
    mS = np.zeros(n)
    for combo in _iproduct((0, 1, 2), repeat=k):
        prod = np.ones(n)
        idx_v: list[int] = []
        idx_s: list[int] = []
        for j, c in enumerate(combo):
            prod = prod * masses[c][:, j]
            if c == 0:
                idx_v.append(j)
            elif c == 1:
                idx_s.append(j)
        if idx_v and idx_s:  # conflict -> PCR6 redistribution
            sum_v = v[:, idx_v].sum(axis=1)
            sum_s = s[:, idx_s].sum(axis=1)
            denom = sum_v + sum_s
            with np.errstate(invalid="ignore", divide="ignore"):
                w_v = np.where(denom > 0, sum_v / denom, 0.5)
            mV += prod * w_v
            mS += prod * (1.0 - w_v)
        elif idx_v:
            mV += prod
        elif idx_s:
            mS += prod
    return mV, mS


def _combine_yager(v: np.ndarray, s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Yager's rule: conjunctive masses kept *unnormalised*; the conflict is reassigned to
    ignorance (Theta) rather than redistributed. The singleton masses are therefore the raw
    conjunctive ``mV``/``mS``; ``mTheta = 1 - mV - mS`` absorbs the conflict. More cautious
    than Dempster (no normalisation amplification under high conflict)."""
    prod_t = np.prod(t, axis=1)
    mV = np.prod(v + t, axis=1) - prod_t
    mS = np.prod(s + t, axis=1) - prod_t
    return mV, mS


def dst_vote_predictions(
    df: pd.DataFrame,
    lookup: dict[tuple[str, str], dict],
    rule: str,
    tools: Sequence[str],
    families: Sequence[str],
    labels: dict[tuple[int, str], bool],
    fire_index: tuple[dict[tuple[int, str], set[str]], list[set[str]]],
    fire_metric: str = "ppv",
    silence_metric: str = "npv",
    tau: float = 0.5,
) -> pd.DataFrame:
    combiners = {"pcr6": _combine_pcr6, "dempster": _combine_dempster, "yager": _combine_yager}
    if rule not in combiners:
        raise ValueError(f"unknown DST rule: {rule!r}")
    combine = combiners[rule]
    strategy = f"dst_{rule}_fire_{fire_metric}_silence_{silence_metric}"
    fire_sets, _ = fire_index
    sample_ids = sample_ids_of(df)
    n = len(df)

    rows: list[dict] = []
    for family in families:
        sup, fire_weights, silence_weights = [], [], []
        for tool in tools:
            row = lookup.get((tool, family))
            if not is_supported(row):
                continue
            p, q = metric_value(row, fire_metric), metric_value(row, silence_metric)
            sup.append(tool)
            fire_weights.append(0.0 if p is None else p)
            silence_weights.append(0.0 if q is None else q)
        k = len(sup)
        abstained = len(tools) - k
        if k == 0:
            for ri in range(n):
                rows.append(evidence_row(sample_ids[ri], ri, family, strategy, False, 0.0, 0.0, 0.0, 0, 0, abstained, labels[(ri, family)]))
            continue

        fire = np.array([[family in fire_sets[(ri, tool)] for tool in sup] for ri in range(n)], dtype=bool)
        v = np.where(fire, np.asarray(fire_weights), 0.0)
        s = np.where(fire, 0.0, np.asarray(silence_weights))
        t = 1.0 - v - s
        mV, mS = combine(v, s, t)
        betp = mV + (1.0 - mV - mS) / 2.0
        fired_counts = fire.sum(axis=1)
        for ri in range(n):
            rows.append(evidence_row(
                sample_ids[ri], ri, family, strategy,
                prediction=bool(betp[ri] >= tau), score=float(betp[ri]),
                vuln=float(mV[ri]), safe=float(mS[ri]),
                k_supported=k, k_fired=int(fired_counts[ri]), k_abstained=abstained,
                label=labels[(ri, family)],
            ))
    return pd.DataFrame(rows)
