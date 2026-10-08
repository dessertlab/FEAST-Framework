"""Shared plumbing for the fusion strategies.

All strategies operate on **canonical families** with exact matching:
* a tool fires family ``f`` on a row iff ``f`` is in that tool's family set for the row;
* the ground-truth label for ``(row, f)`` is True iff ``f`` is in the row's GT family set.

Every strategy emits the same per-(row, family) prediction schema (see ``evidence_row``)
so they all feed straight into ``analysis.fusion.predictions``. Reliability metrics come
from ``analysis.calibration`` and are looked up by ``(tool, family)``.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import pandas as pd

from analysis.dataset import as_list

# Prediction frame columns shared by every strategy.
PREDICTION_COLUMNS = [
    "sample_id", "row_index", "family", "strategy",
    "prediction", "score", "vuln_score", "safe_score",
    "tools_supported", "tools_fired", "tools_abstained", "label",
]

DEFAULT_TAUS = tuple(round(0.1 * i, 1) for i in range(1, 10))

DEFAULT_METRIC_PAIRS = (
    ("ppv", "npv"),
    ("specificity", "sensitivity"),
    ("fpr", "fnr"),
)
CALIBRATION_METRICS = tuple(dict.fromkeys(metric for pair in DEFAULT_METRIC_PAIRS for metric in pair))


def metric_pairs_for_calibration(metrics: Iterable[str] | None = None) -> tuple[tuple[str, str], ...]:
    """Supported fire/silence calibration pairs allowed by a metric list.

    The CLI accepts a comma-separated metric set. We keep only the predefined
    fire/silence pairs whose two metrics are both present, so excluded calibration
    combinations are never materialised as strategies.
    """
    if metrics is None:
        return DEFAULT_METRIC_PAIRS
    allowed = {str(metric).strip().lower() for metric in metrics if str(metric).strip()}
    unknown = allowed - set(CALIBRATION_METRICS)
    if unknown:
        raise ValueError(f"unknown calibration metric(s): {sorted(unknown)}")
    pairs = tuple(pair for pair in DEFAULT_METRIC_PAIRS if set(pair) <= allowed)
    if not pairs:
        raise ValueError(
            "calibration metrics do not contain any supported pair; "
            f"supported pairs are {DEFAULT_METRIC_PAIRS}"
        )
    return pairs


def taus_in_range(tau_min: float = 0.1, tau_max: float = 0.9) -> tuple[float, ...]:
    """Default tau grid clipped to the inclusive [tau_min, tau_max] range."""
    lo, hi = float(tau_min), float(tau_max)
    if lo > hi:
        raise ValueError(f"taumin must be <= taumax, got {lo} > {hi}")
    if lo < 0.0 or hi > 1.0:
        raise ValueError("tau bounds must be within [0.0, 1.0]")
    taus = tuple(tau for tau in DEFAULT_TAUS if lo - 1e-9 <= tau <= hi + 1e-9)
    if not taus:
        raise ValueError(f"tau range [{lo}, {hi}] does not include any default tau in {DEFAULT_TAUS}")
    return taus


def tau_suffix(tau: float) -> str:
    return f"tau_{tau:.1f}".replace(".", "_")


_TAU_RE = re.compile(r"_tau_(\d+)_(\d+)$")


def split_tau_strategy(strategy: str) -> tuple[str, float | None]:
    match = _TAU_RE.search(strategy)
    if not match:
        return strategy, None
    base = strategy[:match.start()]
    return base, float(f"{match.group(1)}.{match.group(2)}")


# Canonical display order of the real fusers (after the single-tool / OR / traditional
# baselines), used so every table and CSV lists strategies in the same sequence.
FUSER_ORDER = (
    *(
        f"weighted_fire_{fire}_silence_{silence}"
        for fire, silence in DEFAULT_METRIC_PAIRS
    ),
    *(
        f"dst_{rule}_fire_{fire}_silence_{silence}"
        for rule in ("dempster", "pcr6", "yager")
        for fire, silence in DEFAULT_METRIC_PAIRS
    ),
    "naive_bayes", "bks",
    "logistic_regression", "logistic_interactions",
    "decision_tree", "random_forest", "gradient_boosting", "xgboost",
)


def canonical_strategy_order(tools: Sequence[str], present: Iterable[str]) -> list[str]:
    """Fixed strategy order: single tools → OR (1-of-N) → traditional voting → fusers.

    Anything unrecognised (e.g. ``always_vulnerable``) is appended last so nothing is lost.
    Only strategies actually present are returned.
    """
    present = list(dict.fromkeys(present))
    order: list[str] = [f"tool:{t}" for t in tools]
    order += [s for s in present if s.startswith("or_1_of_")]
    order += sorted(s for s in present if s.startswith("traditional"))
    fuser_rank = {name: i for i, name in enumerate(FUSER_ORDER)}
    fuser_present = []
    for strategy in present:
        base, tau = split_tau_strategy(strategy)
        if base in fuser_rank:
            fuser_present.append((fuser_rank[base], 1.0 if tau is None else tau, strategy))
    order += [strategy for _rank, _tau, strategy in sorted(fuser_present)]
    seen: set[str] = set()
    out: list[str] = []
    for s in order:
        if s in present and s not in seen:
            out.append(s)
            seen.add(s)
    out += [s for s in present if s not in seen]   # trailing unknowns (e.g. baselines)
    return out


def order_by_strategy(df: pd.DataFrame, tools: Sequence[str], extra_sort: Sequence[str] = ()) -> pd.DataFrame:
    """Return ``df`` row-sorted by the canonical strategy order (then ``extra_sort``)."""
    if "strategy" not in df.columns or df.empty:
        return df
    order = canonical_strategy_order(tools, df["strategy"].unique())
    cat = pd.Categorical(df["strategy"], categories=order, ordered=True)
    ordered = df.assign(_strategy_rank=cat.codes).sort_values(["_strategy_rank", *extra_sort])
    return ordered.drop(columns="_strategy_rank").reset_index(drop=True)


def family_sets(series: pd.Series) -> list[set[str]]:
    return [set(as_list(values)) for values in series]


def build_fire_index(
    df: pd.DataFrame,
    tools: Sequence[str],
) -> tuple[dict[tuple[int, str], set[str]], list[set[str]]]:
    """Parse each tool's family output once per row.

    Returns ``(fire_sets, fired_union)`` where ``fire_sets[(row, tool)]`` is the set of
    families the tool fired on that row and ``fired_union[row]`` is the union across tools
    (lets strategies skip families no tool fired on a row).
    """
    fire_sets: dict[tuple[int, str], set[str]] = {}
    fired_union: list[set[str]] = []
    for row_index, row in enumerate(df.to_dict("records")):
        union: set[str] = set()
        for tool in tools:
            fires = set(as_list(row[tool])) if tool in row else set()
            fire_sets[(row_index, tool)] = fires
            union |= fires
        fired_union.append(union)
    return fire_sets, fired_union


def precompute_labels(df: pd.DataFrame, families: Sequence[str]) -> dict[tuple[int, str], bool]:
    """Exact GT positivity per (row, family); shared by all strategies."""
    gt_by_row = family_sets(df["cwes"])
    return {
        (row_index, family): family in gt_families
        for family in families
        for row_index, gt_families in enumerate(gt_by_row)
    }


def metric_lookup(reliability: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """Index calibration metrics by (tool, family)."""
    lookup: dict[tuple[str, str], dict] = {}
    for row in reliability.to_dict("records"):
        lookup[(str(row["tool"]), str(row["family"]))] = row
    return lookup


def metric_value(row: dict, metric: str) -> float | None:
    """Read a reliability metric, coercing NaN/None to None.

    All metrics (ppv, npv, fpr, fnr, sensitivity, specificity) are materialised as
    columns by ``analysis.calibration``, so this is a direct, name-based read.
    """
    value = row.get(metric)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def is_supported(metric_row: dict | None) -> bool:
    """A (tool, family) pair is usable iff the tool fired the family in calibration."""
    if not metric_row:
        return False
    value = metric_row.get("supported")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def sample_ids_of(df: pd.DataFrame) -> list:
    return df["sample_id"].tolist() if "sample_id" in df.columns else list(range(len(df)))


def evidence_row(sample_id, row_index, family, strategy, prediction, score, vuln, safe,
                 k_supported, k_fired, k_abstained, label) -> dict:
    """Build one prediction record in the shared schema. Each strategy supplies its own
    ``prediction`` (boolean decision) and ``score`` (continuous, for ROC/PR)."""
    return {
        "sample_id": sample_id,
        "row_index": int(row_index),
        "family": family,
        "strategy": strategy,
        "prediction": bool(prediction),
        "score": float(score),
        "vuln_score": float(vuln),
        "safe_score": float(safe),
        "tools_supported": int(k_supported),
        "tools_fired": int(k_fired),
        "tools_abstained": int(k_abstained),
        "label": bool(label),
    }


def resolve_families(families: Iterable[str] | None, df: pd.DataFrame, tools: Sequence[str]) -> list[str]:
    """Family grid: explicit list if given, else every family present in df (GT ∪ tools)."""
    if families is not None:
        return list(families)
    present: set[str] = set()
    for column in ["cwes", *tools]:
        if column in df.columns:
            for values in df[column]:
                present |= set(as_list(values))
    return sorted(present, key=lambda c: int(str(c).removeprefix("CWE-")))
