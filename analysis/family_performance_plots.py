"""Top-CWE family heatmaps for tools and representative fusion strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from analysis.aggregation import REPORT_METRICS, SUPPORT_COLUMN
from analysis.fusion.common import canonical_strategy_order
from analysis.reporting import LOWER_IS_BETTER, best_variants_for_metric

MAXIMIZE_METRICS = tuple(metric for metric in REPORT_METRICS if metric not in LOWER_IS_BETTER)


def top_families(per_family: pd.DataFrame, n: int = 10) -> list[str]:
    """Most frequent CWE-1000 families by ground-truth support."""
    required = {"family", SUPPORT_COLUMN}
    missing = required - set(per_family.columns)
    if missing:
        raise ValueError(f"per_family is missing required columns: {sorted(missing)}")
    support = (
        per_family[["family", SUPPORT_COLUMN]]
        .assign(**{SUPPORT_COLUMN: lambda df: pd.to_numeric(df[SUPPORT_COLUMN], errors="coerce").fillna(0)})
        .groupby("family", as_index=False)[SUPPORT_COLUMN]
        .max()
        .sort_values([SUPPORT_COLUMN, "family"], ascending=[False, True])
    )
    return support.head(n)["family"].astype(str).tolist()


def selected_strategy_rows(
    per_family: pd.DataFrame,
    overall: pd.DataFrame,
    metric: str,
    tools: Sequence[str] = (),
) -> pd.DataFrame:
    """Select tool/baseline rows and one best variant for each fusion group."""
    selected = best_variants_for_metric(overall, metric)
    if selected.empty:
        return selected
    present = set(per_family["strategy"].astype(str))
    selected = selected[selected["strategy"].astype(str).isin(present)].copy()
    order = canonical_strategy_order(tools, selected["strategy"].astype(str).tolist())
    rank = {strategy: i for i, strategy in enumerate(order)}
    selected["_rank"] = selected["strategy"].map(rank).fillna(len(rank)).astype(int)
    return selected.sort_values(["_rank", "strategy"]).drop(columns="_rank").reset_index(drop=True)


def family_metric_matrix(
    per_family: pd.DataFrame,
    overall: pd.DataFrame,
    metric: str,
    tools: Sequence[str] = (),
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.Series]:
    """Matrix of strategy rows by top CWE-1000 families for one metric."""
    required = {"strategy", "family", metric, SUPPORT_COLUMN}
    missing = required - set(per_family.columns)
    if missing:
        raise ValueError(f"per_family is missing required columns: {sorted(missing)}")
    if metric not in MAXIMIZE_METRICS:
        raise ValueError(f"metric {metric!r} is not a maximize metric")

    families = top_families(per_family, top_n)
    selected = selected_strategy_rows(per_family, overall, metric, tools=tools)
    if selected.empty or not families:
        return pd.DataFrame(), pd.Series(dtype=float)

    label_map = dict(zip(selected["strategy"], selected["selected_variant"], strict=False))
    ordered_strategies = selected["strategy"].astype(str).tolist()
    frame = per_family[
        per_family["strategy"].astype(str).isin(ordered_strategies)
        & per_family["family"].astype(str).isin(families)
    ].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame["display_strategy"] = frame["strategy"].map(label_map).fillna(frame["strategy"])

    matrix = frame.pivot_table(
        index="display_strategy",
        columns="family",
        values=metric,
        aggfunc="mean",
        dropna=False,
    )
    ordered_labels = [str(label_map.get(strategy, strategy)) for strategy in ordered_strategies]
    matrix = matrix.reindex(index=ordered_labels, columns=families)

    support = (
        per_family[per_family["family"].astype(str).isin(families)]
        .assign(**{SUPPORT_COLUMN: lambda df: pd.to_numeric(df[SUPPORT_COLUMN], errors="coerce").fillna(0)})
        .groupby("family")[SUPPORT_COLUMN]
        .max()
        .reindex(families)
    )
    return matrix, support


def _metric_limits(values: pd.DataFrame) -> tuple[float, float]:
    finite = pd.to_numeric(values.stack(), errors="coerce").dropna()
    if finite.empty:
        return 0.0, 1.0
    lo = min(0.0, float(finite.min()))
    hi = max(1.0, float(finite.max()))
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def _family_labels(support: pd.Series) -> list[str]:
    return [f"{family}\nn={int(value)}" for family, value in support.items()]


def save_family_heatmap(matrix: pd.DataFrame, support: pd.Series, metric: str, stem: Path) -> list[Path]:
    """Save a conventional heatmap in SVG and PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if matrix.empty:
        return []
    stem.parent.mkdir(parents=True, exist_ok=True)
    values = matrix.astype(float)
    vmin, vmax = _metric_limits(values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f0f0f0")

    fig_w = max(10, 0.85 * len(matrix.columns) + 4.5)
    fig_h = max(5, 0.34 * len(matrix.index) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(np.ma.masked_invalid(values.to_numpy()), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)), _family_labels(support), rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index.tolist())
    ax.set_title(f"Top 10 famiglie CWE-1000 - {metric}")
    ax.set_xlabel("Famiglia CWE-1000 e supporto GT")
    ax.set_ylabel("Tool / strategia")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(metric)
    ax.set_xticks([x - 0.5 for x in range(1, len(matrix.columns))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(matrix.index))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()

    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    return [svg_path, png_path]


def save_family_dot_heatmap(matrix: pd.DataFrame, support: pd.Series, metric: str, stem: Path) -> list[Path]:
    """Save a dot heatmap in SVG and PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if matrix.empty:
        return []
    stem.parent.mkdir(parents=True, exist_ok=True)
    values = matrix.astype(float)
    vmin, vmax = _metric_limits(values)
    span = (vmax - vmin) or 1.0

    xs: list[int] = []
    ys: list[int] = []
    colors: list[float] = []
    sizes: list[float] = []
    for y, _strategy in enumerate(values.index):
        for x, _family in enumerate(values.columns):
            value = values.iat[y, x]
            if pd.isna(value):
                continue
            scaled = max(0.0, min(1.0, (float(value) - vmin) / span))
            xs.append(x)
            ys.append(y)
            colors.append(float(value))
            sizes.append(45 + 420 * scaled)

    fig_w = max(10, 0.85 * len(matrix.columns) + 4.5)
    fig_h = max(5, 0.34 * len(matrix.index) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    scatter = ax.scatter(xs, ys, c=colors, s=sizes, cmap="viridis", vmin=vmin, vmax=vmax, alpha=0.95)
    ax.set_xlim(-0.5, len(matrix.columns) - 0.5)
    ax.set_ylim(len(matrix.index) - 0.5, -0.5)
    ax.set_xticks(range(len(matrix.columns)), _family_labels(support), rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index.tolist())
    ax.set_title(f"Top 10 famiglie CWE-1000 - {metric} (dot heatmap)")
    ax.set_xlabel("Famiglia CWE-1000 e supporto GT")
    ax.set_ylabel("Tool / strategia")
    ax.grid(color="#dddddd", linewidth=0.8)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(metric)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [
        ax.scatter([], [], s=size, color="#666666", alpha=0.8, label=label)
        for size, label in [(90, "basso"), (250, "medio"), (450, "alto")]
    ]
    ax.legend(handles=handles, title="Valore", loc="upper left", bbox_to_anchor=(1.04, 1.0), frameon=False)
    fig.tight_layout()

    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    return [svg_path, png_path]


def save_family_performance_plots(
    per_family: pd.DataFrame,
    overall: pd.DataFrame,
    out_dir: Path,
    *,
    metrics: Iterable[str] = MAXIMIZE_METRICS,
    tools: Sequence[str] = (),
    top_n: int = 10,
) -> list[Path]:
    """Write heatmap and dot-heatmap pairs for every maximize metric."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric in metrics:
        if metric in LOWER_IS_BETTER or metric not in per_family.columns:
            continue
        matrix, support = family_metric_matrix(per_family, overall, metric, tools=tools, top_n=top_n)
        if matrix.empty:
            continue
        written.extend(save_family_heatmap(matrix, support, metric, out_dir / f"heatmap_{metric}"))
        written.extend(save_family_dot_heatmap(matrix, support, metric, out_dir / f"dot_heatmap_{metric}"))
    return written
