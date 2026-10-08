"""Output organisation for the fusion experiment: CSVs, SVGs and PNGs.

Layout (per language at the single canonical level), rooted at ``results_dir``:

    <results_dir>/
      config.json                    run config + restriction statistics
      canonical_map.csv              raw CWE -> family audit table
      folds.csv                      sample_id -> fold
      calibration_reliability.csv    per (fold, tool, family) confusion + metrics
      fusion_predictions.csv         per (fold, strategy, family, row)
      fusion_metrics_per_family.csv  per (strategy, family) mean over folds + support
      fusion_metrics_overall.csv     per strategy: support-weighted / macro / median
      fusion_detection_overall.csv   per strategy: vuln/safe detection + attribution acc
      plots/<metric>.svg, .png       strategy comparison bar charts, all variants
      plots/best_variants/<metric>.svg, .png
                                     best tau/calibration-metric variant per strategy
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd

from analysis.aggregation import REPORT_METRICS
from analysis.fusion.common import split_tau_strategy

# Metrics charted by default (the support-weighted reductions from aggregation).
PLOT_METRICS = REPORT_METRICS
LOWER_IS_BETTER = {"fpr", "fnr"}


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_config(config: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return path


# ── plots ─────────────────────────────────────────────────────────────────────

def _bar_svg(values: list[tuple[str, float | None]], title: str, width: int = 960) -> str:
    row_h = 28
    max_label = max((len(str(name)) for name, _value in values), default=0)
    label_w = min(560, max(300, max_label * 7 + 10))
    width = max(width, label_w + 520)
    plot_w = width - label_w - 90
    height = 54 + max(len(values), 1) * row_h
    finite = [v for _n, v in values if v is not None]
    lo, hi = min([0.0, *finite]) if finite else 0.0, max([1.0, *finite]) if finite else 1.0
    span = (hi - lo) or 1.0
    zero_x = label_w + int((0 - lo) / span * plot_w)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px}.t{font-size:16px;font-weight:700}'
        '.ax{stroke:#888;stroke-width:1}.bar{fill:#3568a8}.v{fill:#222}</style>',
        f'<text class="t" x="0" y="20">{escape(title)}</text>',
        f'<line class="ax" x1="{zero_x}" y1="34" x2="{zero_x}" y2="{height - 10}"/>',
    ]
    for i, (name, value) in enumerate(values):
        y = 42 + i * row_h
        parts.append(f'<text x="0" y="{y + 15}">{escape(str(name))}</text>')
        if value is None:
            parts.append(f'<text class="v" x="{label_w}" y="{y + 15}">n/a</text>')
            continue
        x = label_w + int((min(value, 0) - lo) / span * plot_w)
        end = label_w + int((max(value, 0) - lo) / span * plot_w)
        parts.append(f'<rect class="bar" x="{min(x, end)}" y="{y}" width="{max(abs(end - x), 2)}" height="19" rx="2"/>')
        parts.append(f'<text class="v" x="{label_w + plot_w + 10}" y="{y + 15}">{value:.3f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _save_png(values: list[tuple[str, float | None]], title: str, path: Path) -> None:
    """Horizontal bar chart via matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n, _v in values]
    vals = [0.0 if v is None else v for _n, v in values]
    max_label = max((len(str(name)) for name in names), default=0)
    fig_width = max(9, min(14, 6 + max_label * 0.11))
    fig, ax = plt.subplots(figsize=(fig_width, 0.45 * len(values) + 1.5))
    ax.barh(names, vals, color="#3568a8")
    ax.set_xlabel(title)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _ascending_for_metric(metric: str) -> bool:
    return metric in LOWER_IS_BETTER


def _metric_value(row: pd.Series, column: str) -> float | None:
    value = row.get(column)
    if pd.isna(value):
        return None
    return float(value)


def _strategy_variant(strategy: str) -> tuple[str, str]:
    base, tau = split_tau_strategy(strategy)
    group = base
    details: list[str] = []

    if base.startswith("weighted_fire_") and "_silence_" in base:
        fire, silence = base.removeprefix("weighted_fire_").split("_silence_", 1)
        group = "weighted"
        details.append(f"cal={fire}/{silence}")
    elif base.startswith("dst_") and "_fire_" in base and "_silence_" in base:
        rule, rest = base.removeprefix("dst_").split("_fire_", 1)
        fire, silence = rest.split("_silence_", 1)
        group = f"dst_{rule}"
        details.append(f"cal={fire}/{silence}")

    if tau is not None:
        details.insert(0, f"tau={tau:.1f}")
    label = group if not details else f"{group} ({', '.join(details)})"
    return group, label


def _sort_for_plot(df: pd.DataFrame, column: str, metric: str) -> pd.DataFrame:
    return df.sort_values(column, ascending=_ascending_for_metric(metric), na_position="last")


def best_variants_for_metric(overall: pd.DataFrame, metric: str, suffix: str = "_weighted") -> pd.DataFrame:
    """Select one best tau/calibration-metric variant per strategy for one validation metric."""
    column = f"{metric}{suffix}"
    if overall.empty or column not in overall.columns or "strategy" not in overall.columns:
        return overall.iloc[0:0].copy()

    frame = overall.copy()
    variants = frame["strategy"].map(_strategy_variant)
    frame.insert(1, "strategy_group", variants.map(lambda item: item[0]))
    frame.insert(2, "selected_variant", variants.map(lambda item: item[1]))
    frame[column] = pd.to_numeric(frame[column], errors="coerce")

    selected = []
    ascending = _ascending_for_metric(metric)
    for _group, group in frame.groupby("strategy_group", sort=False):
        valid = group[group[column].notna()]
        candidates = valid if not valid.empty else group
        selected.append(
            candidates.sort_values([column, "strategy"], ascending=[ascending, True], na_position="last").iloc[0]
        )
    return pd.DataFrame(selected).reset_index(drop=True)


def _write_bar_pair(values: list[tuple[str, float | None]], title: str, stem: Path) -> list[Path]:
    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    svg_path.write_text(_bar_svg(values, title), encoding="utf-8")
    _save_png(values, title, png_path)
    return [svg_path, png_path]


def save_strategy_plots(
    overall: pd.DataFrame,
    out_dir: Path,
    metrics: Iterable[str] = PLOT_METRICS,
    suffix: str = "_weighted",
) -> list[Path]:
    """One bar chart (SVG + PNG) per metric, comparing all strategy variants."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric in metrics:
        column = f"{metric}{suffix}"
        if column not in overall.columns:
            continue
        ordered = _sort_for_plot(overall, column, metric)
        values = [(str(r["strategy"]), _metric_value(r, column)) for _i, r in ordered.iterrows()]
        title = f"{metric} (support-weighted)"
        written.extend(_write_bar_pair(values, title, out_dir / metric))
    return written


def save_best_variant_plots(
    overall: pd.DataFrame,
    out_dir: Path,
    metrics: Iterable[str] = PLOT_METRICS,
    suffix: str = "_weighted",
) -> list[Path]:
    """One bar chart per metric after choosing the best variant within each strategy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric in metrics:
        column = f"{metric}{suffix}"
        selected = best_variants_for_metric(overall, metric, suffix=suffix)
        if selected.empty or column not in selected.columns:
            continue
        ordered = _sort_for_plot(selected, column, metric)
        values = [(str(r["selected_variant"]), _metric_value(r, column)) for _i, r in ordered.iterrows()]
        title = f"{metric} best variant per strategy (support-weighted)"
        written.extend(_write_bar_pair(values, title, out_dir / metric))
    return written
