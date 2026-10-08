"""Reliability-weighted voting.

For each (row, family) every *supported* tool casts a weighted vote: if it fired the
family it adds its ``fire_metric`` to the vuln score, otherwise it adds its
``silence_metric`` to the safe score. The family is predicted vulnerable when the vuln
score wins. Different (fire, silence) metric pairs give different strategies — e.g.
trusting precision on a fire and NPV on a silence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from analysis.fusion.common import DEFAULT_METRIC_PAIRS, evidence_row, is_supported, metric_value, sample_ids_of


@dataclass(frozen=True)
class WeightedVotingStrategy:
    fire_metric: str
    silence_metric: str

    @property
    def name(self) -> str:
        return f"weighted_fire_{self.fire_metric}_silence_{self.silence_metric}"


DEFAULT_WEIGHTED_STRATEGIES = (
    *(WeightedVotingStrategy(fire, silence) for fire, silence in DEFAULT_METRIC_PAIRS),
)


def weighted_vote_predictions(
    df: pd.DataFrame,
    lookup: dict[tuple[str, str], dict],
    strategy: WeightedVotingStrategy,
    tools: Sequence[str],
    families: Sequence[str],
    labels: dict[tuple[int, str], bool],
    fire_index: tuple[dict[tuple[int, str], set[str]], list[set[str]]],
) -> pd.DataFrame:
    fire_sets, fired_union = fire_index
    sample_ids = sample_ids_of(df)

    # Per-family constants (independent of the row): which tools are supported, their fire
    # and silence weights, and the total silence score assumed when no tool fires.
    supported_tools: dict[str, list[str]] = {}
    fire_weight: dict[tuple[str, str], float] = {}
    silence_weight: dict[tuple[str, str], float] = {}
    silence_total: dict[str, float] = {}
    n_abstained: dict[str, int] = {}
    for family in families:
        sup: list[str] = []
        total_silence = 0.0
        for tool in tools:
            row = lookup.get((tool, family))
            if not is_supported(row):
                continue
            sup.append(tool)
            fw = metric_value(row, strategy.fire_metric)
            sw = metric_value(row, strategy.silence_metric)
            fire_weight[(family, tool)] = 0.0 if fw is None else fw
            sw = 0.0 if sw is None else sw
            silence_weight[(family, tool)] = sw
            total_silence += sw
        supported_tools[family] = sup
        silence_total[family] = total_silence
        n_abstained[family] = len(tools) - len(sup)

    rows: list[dict] = []
    for row_index in range(len(df)):
        fired_here = fired_union[row_index]
        for family in families:
            sup = supported_tools[family]
            if family not in fired_here:
                # No supported tool fired -> everyone silent.
                vuln_score, safe_score, tools_fired = 0.0, silence_total[family], 0
            else:
                vuln_score = safe_score = 0.0
                tools_fired = 0
                for tool in sup:
                    if family in fire_sets[(row_index, tool)]:
                        tools_fired += 1
                        vuln_score += fire_weight[(family, tool)]
                    else:
                        safe_score += silence_weight[(family, tool)]

            total = vuln_score + safe_score
            score = 0.0 if total == 0 else vuln_score / total
            rows.append(evidence_row(
                sample_ids[row_index], row_index, family, strategy.name,
                prediction=(vuln_score >= safe_score and total > 0), score=score,
                vuln=vuln_score, safe=safe_score,
                k_supported=len(sup), k_fired=tools_fired, k_abstained=n_abstained[family],
                label=labels[(row_index, family)],
            ))
    return pd.DataFrame(rows)
