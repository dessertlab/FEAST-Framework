"""Calibration profile plot: Line plot showing metrics per (tool, CWE-family).

X-axis = CWE families, Y-axis = Metric (sensitivity or specificity).
Each tool is represented by a line with distinct color and marker.
A 2x3 grid: rows = metrics (sensitivity, specificity), cols = languages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from analysis.plot_style import LANG_TITLES, load_calibration, plot_cli_args, rc_style, tool_palette

def make_calibration_profile(
    langs: Sequence[str] = ("python", "java", "c_cpp"),
    results_root: str | Path = "data/results",
    out_stem: str | Path | None = None,
    min_tools: int = 2,
    use_sans: bool = True,
) -> list[Path]:
    results_root = Path(results_root)
    if out_stem is None:
        out_stem = results_root / "calibration_profile"
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    all_tools: list[str] = []
    data: dict[str, pd.DataFrame] = {}
    for lang in langs:
        df = load_calibration(lang, results_root, min_tools=min_tools)
        data[lang] = df
        for t in df["tool"].unique():
            if t not in all_tools:
                all_tools.append(t)

    tool_colors, tool_markers = tool_palette(all_tools)
    rc = rc_style(use_sans, {"font.size": 10})

    n_langs = len(langs)
    metrics = ["sensitivity", "specificity"]
    metric_labels = {"sensitivity": "Sensitivity (recall)", "specificity": "Specificity"}

    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, n_langs, figsize=(6.5 * n_langs, 8.5), sharex="col", sharey="row")
        
        for col_idx, lang in enumerate(langs):
            df = data[lang]
            if df.empty:
                continue
            
            # Get unique CWE families, sorted by their average support or support descending
            family_order = (
                df.groupby("family")["positive_support"]
                .max()
                .sort_values(ascending=False)
                .index.tolist()
            )
            
            x_ticks = np.arange(len(family_order))
            
            for row_idx, metric in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                
                # Draw lines for each tool
                for tool in sorted(df["tool"].unique()):
                    tool_df = df[df["tool"] == tool].set_index("family").reindex(family_order)
                    
                    y_values = tool_df[metric].values
                    # Filter out tools that have no values at all for this language
                    if np.all(np.isnan(y_values)):
                        continue
                        
                    ax.plot(
                        x_ticks,
                        y_values,
                        color=tool_colors[t := tool],
                        marker=tool_markers[t],
                        markersize=6,
                        linewidth=1.2,
                        label=tool,
                        alpha=0.85,
                        zorder=3
                    )
                
                # Formatting subplot
                ax.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", alpha=0.7)
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
                
                if row_idx == 0:
                    ax.set_title(LANG_TITLES.get(lang, lang), fontsize=13, fontweight="bold", pad=8)
                    ax.set_ylim(-0.05, 1.00)
                else:
                    # Specificity: zoom to minimum specificity up to 1.00
                    y_min_val = df["specificity"].min()
                    y_lo = float(np.floor(y_min_val * 20) / 20)  # round down to nearest 0.05
                    y_lo = max(0.0, y_lo)
                    ax.set_ylim(y_lo, 1.005)
                
                if col_idx == 0:
                    ax.set_ylabel(metric_labels[metric], fontsize=11, fontweight="bold")
                
                if row_idx == 1:
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(family_order, rotation=45, ha="right", fontsize=9)
                    ax.set_xlabel("CWE Family", fontsize=10)

        # Shared legend
        tool_handles = [
            Line2D([], [], marker=tool_markers[t], linestyle="-", markersize=8,
                   color=tool_colors[t], linewidth=1.2, label=t)
            for t in sorted(all_tools)
        ]
        
        fig.legend(
            handles=tool_handles,
            title="Tool",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=min(len(all_tools), 8),
            frameon=False,
            fontsize=10,
            title_fontsize=10,
        )

        fig.suptitle(
            "CWE Performance Profile per Tool — Sensitivity and Specificity Comparison\n"
            "(CWEs sorted by support descending; lines crossing highlight trade-off crossovers between tools)",
            fontsize=12,
            y=1.02,
        )

        fig.tight_layout(rect=(0, 0.02, 1, 1))

        png_path = out_stem.with_suffix(".png")
        svg_path = out_stem.with_suffix(".svg")
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        print(f"Saved profile: {png_path}")
        return [png_path, svg_path]

if __name__ == "__main__":
    args = plot_cli_args("Calibration profile plot.", default_langs=["python", "java", "c_cpp"])

    make_calibration_profile(
        langs=args.langs,
        results_root=args.results_dir,
        out_stem=args.out_stem,
        min_tools=args.min_tools,
        use_sans=True,
    )
