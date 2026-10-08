"""Non-fusion reference strategies, shown at the top of every comparison.

* ``tool:<name>`` — a single tool acting alone (fires a family ⇒ predicts it).
* ``or_1_of_<N>`` — the union (OR / 1-out-of-N) of the tools: any tool firing the family.

They emit the shared prediction schema, so they flow through the same evaluation, τ-sweep
and aggregation as the real fusers and anchor the comparison from below.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from analysis.fusion.common import evidence_row, sample_ids_of


def single_tool_predictions(
    df: pd.DataFrame,
    tool: str,
    families: Sequence[str],
    labels: dict[tuple[int, str], bool],
    fire_index: tuple[dict[tuple[int, str], set[str]], list[set[str]]],
) -> pd.DataFrame:
    fire_sets, _ = fire_index
    sample_ids = sample_ids_of(df)
    rows: list[dict] = []
    for ri in range(len(df)):
        for family in families:
            fired = family in fire_sets[(ri, tool)]
            rows.append(evidence_row(
                sample_ids[ri], ri, family, f"tool:{tool}",
                prediction=fired, score=1.0 if fired else 0.0,
                vuln=1.0 if fired else 0.0, safe=0.0 if fired else 1.0,
                k_supported=1, k_fired=int(fired), k_abstained=0,
                label=labels[(ri, family)],
            ))
    return pd.DataFrame(rows)


def or_predictions(
    df: pd.DataFrame,
    tools: Sequence[str],
    families: Sequence[str],
    labels: dict[tuple[int, str], bool],
    fire_index: tuple[dict[tuple[int, str], set[str]], list[set[str]]],
) -> pd.DataFrame:
    fire_sets, _ = fire_index
    sample_ids = sample_ids_of(df)
    name = f"or_1_of_{len(tools)}"
    rows: list[dict] = []
    for ri in range(len(df)):
        for family in families:
            count = sum(1 for t in tools if family in fire_sets[(ri, t)])
            fired = count > 0
            rows.append(evidence_row(
                sample_ids[ri], ri, family, name,
                prediction=fired, score=1.0 if fired else 0.0,
                vuln=float(count), safe=float(len(tools) - count),
                k_supported=len(tools), k_fired=count, k_abstained=0,
                label=labels[(ri, family)],
            ))
    return pd.DataFrame(rows)
