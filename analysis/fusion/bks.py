"""Behavior-Knowledge Space (BKS) fusion (Huang & Suen, 1995).

With ``k`` binary tools there are only ``2^k`` possible fire patterns per family. BKS is
the *saturated* combiner: estimate ``P(vulnerable | pattern)`` directly from the
calibration split and look it up on validation. It is the empirical ceiling of what these
tools can yield — no fusion rule can beat the per-pattern positive rate.

**The m-estimate is part of the method, not a variant of it.** The textbook estimator is
the raw per-pattern rate ``#{f, y=1} / #{f}``, but with ``2^k`` cells and a calibration
split of finite size, rare patterns are estimated from one or two observations and return
exactly 0.0 or 1.0, while patterns never seen in calibration have no estimate at all. The
lookup table is therefore always smoothed toward the family prior,
``(positives + m·prior) / (total + m)`` with ``m = 2``, which leaves well-populated cells
essentially untouched and pulls the sparse ones back to the prior (an unseen pattern
scores exactly the prior). There is no raw-BKS code path: ``m`` is a parameter of the
estimator, not a switch between two methods.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from analysis.fusion.common import build_fire_index, evidence_row, precompute_labels, sample_ids_of


def bks_predictions(
    calibration_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    tools: Sequence[str],
    families: Sequence[str],
    labels: dict[tuple[int, str], bool],
    fire_index: tuple[dict[tuple[int, str], set[str]], list[set[str]]],
    tau: float = 0.5,
    m: float = 2.0,
) -> pd.DataFrame:
    cal_fire, _ = build_fire_index(calibration_df, tools)
    val_fire, _ = fire_index
    cal_labels = precompute_labels(calibration_df, families)
    sample_ids = sample_ids_of(validation_df)
    n_cal, n_val = len(calibration_df), len(validation_df)

    def pattern(fire, i, family):
        return tuple(family in fire[(i, t)] for t in tools)

    rows: list[dict] = []
    for family in families:
        # calibration: aggregate label rate per output pattern
        pos: dict[tuple, int] = {}
        tot: dict[tuple, int] = {}
        n_pos = 0
        for i in range(n_cal):
            p = pattern(cal_fire, i, family)
            tot[p] = tot.get(p, 0) + 1
            if cal_labels[(i, family)]:
                pos[p] = pos.get(p, 0) + 1
                n_pos += 1
        prior = n_pos / n_cal if n_cal else 0.0

        for i in range(n_val):
            p = pattern(val_fire, i, family)
            # m-estimate: (positives + m·prior) / (total + m); unseen pattern -> prior
            score = (pos.get(p, 0) + m * prior) / (tot.get(p, 0) + m)
            fired = sum(p)
            rows.append(evidence_row(
                sample_ids[i], i, family, "bks",
                prediction=bool(score >= tau), score=float(score),
                vuln=float(score), safe=float(1.0 - score),
                k_supported=len(tools), k_fired=fired, k_abstained=0,
                label=labels[(i, family)],
            ))
    return pd.DataFrame(rows)
