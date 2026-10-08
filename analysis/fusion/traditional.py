"""Traditional K-of-N voting baseline.

The unweighted reference the paper argues against: a family is predicted vulnerable when
at least ``threshold`` of all configured tools fire it.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from analysis.fusion.common import evidence_row, sample_ids_of


def traditional_vote_predictions(
    df: pd.DataFrame,
    tools: Sequence[str],
    families: Sequence[str],
    threshold: int,
    labels: dict[tuple[int, str], bool],
    fire_index: tuple[dict[tuple[int, str], set[str]], list[set[str]]],
) -> pd.DataFrame:
    fire_sets, fired_union = fire_index
    sample_ids = sample_ids_of(df)
    strategy = f"traditional_{threshold}_of_{len(tools)}"

    rows: list[dict] = []
    for row_index in range(len(df)):
        fired_here = fired_union[row_index]
        for family in families:
            if family not in fired_here:
                fire_count = 0
            else:
                fire_count = sum(1 for tool in tools if family in fire_sets[(row_index, tool)])
            n = len(tools)
            rows.append(evidence_row(
                sample_ids[row_index], row_index, family, strategy,
                prediction=(fire_count >= threshold),
                score=(0.0 if n == 0 else fire_count / n),
                vuln=float(fire_count), safe=float(n - fire_count),
                k_supported=n, k_fired=fire_count, k_abstained=0,
                label=labels[(row_index, family)],
            ))
    return pd.DataFrame(rows)
