"""Calibration bar plot: grouped bars (sensitivity, specificity) per (tool, CWE).

Grid layout:
- Rows: Languages (Python, Java, C/C++)
- Columns: Top 6 CWE families per language (ordered numerically)
- Inside each subplot: X-axis lists the tools for that language, showing Sensitivity
  and Specificity side-by-side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.plot_style import LANG_TITLES, load_calibration, plot_cli_args, rc_style

# Colorblind-friendly pairing for the two metrics
_METRIC_COLORS = {
    "sensitivity": "#56B4E9",  # Sky Blue
    "specificity": "#E69F00",  # Orange
}

def make_calibration_bars(
    langs: Sequence[str] = ("c_cpp", "java", "python"),
    results_root: str | Path = "data/results",
    out_stem: str | Path | None = None,
    min_tools: int = 2,
    use_sans: bool = True,
) -> list[Path]:
    results_root = Path(results_root)
    if out_stem is None:
        out_stem = results_root / "calibration_bars"
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, pd.DataFrame] = {}
    lang_tools: dict[str, list[str]] = {}
    lang_cwes: dict[str, list[str]] = {}
    lang_cwe_support: dict[str, dict[str, int]] = {}

    for lang in langs:
        df = load_calibration(lang, results_root, min_tools=min_tools)
        data[lang] = df
        if not df.empty:
            lang_tools[lang] = sorted(df["tool"].unique().tolist())
            
            # Unique CWEs in the filtered set
            unique_cwes = df["family"].unique().tolist()
            
            # Find maximum positive support for each CWE family
            support_map = df.groupby("family")["positive_support"].max().to_dict()
            lang_cwe_support[lang] = {k: int(round(v)) for k, v in support_map.items()}
            
            # Sort CWEs numerically by their number
            def get_cwe_num(cwe_str: str) -> int:
                try:
                    return int(cwe_str.replace("CWE-", ""))
                except ValueError:
                    return 999999
            
            sorted_cwes = sorted(unique_cwes, key=get_cwe_num)
            lang_cwes[lang] = sorted_cwes
        else:
            lang_tools[lang] = []
            lang_cwes[lang] = []
            lang_cwe_support[lang] = {}

    rc = rc_style(use_sans, {
        "font.size": 16, "axes.labelsize": 14, "xtick.labelsize": 16, "ytick.labelsize": 16,
    })

    n_langs = len(langs)
    n_cwes = 6  # Show top 6 CWEs

    with plt.rc_context(rc):
        # Grid: rows = languages, columns = CWEs (6 columns). sharey=False to show Y-axis on every subplot
        fig, axes = plt.subplots(
            n_langs,
            n_cwes,
            figsize=(3.6 * n_cwes, 3.6 * n_langs),
            sharey=False,
            squeeze=False
        )

        for row_idx, lang in enumerate(langs):
            df = data[lang]
            tools = lang_tools[lang]
            cwes = lang_cwes[lang]
            supports = lang_cwe_support[lang]

            for col_idx in range(n_cwes):
                ax = axes[row_idx, col_idx]
                
                # Hide unused axes
                if col_idx >= len(cwes):
                    ax.set_visible(False)
                    continue

                cwe = cwes[col_idx]
                cwe_df = df[df["family"] == cwe].set_index("tool").reindex(tools).reset_index()

                x = np.arange(len(tools))
                width = 0.35

                # Fill NaNs with 0 for plotting
                sens_vals = cwe_df["sensitivity"].fillna(0).values
                spec_vals = cwe_df["specificity"].fillna(0).values

                # Draw bars with black borders and textures/hatching
                sens_bars = ax.bar(
                    x - width/2,
                    sens_vals,
                    width,
                    color=_METRIC_COLORS["sensitivity"],
                    edgecolor="black",
                    linewidth=0.7,
                    hatch="//",
                    label="Sensitivity"
                )
                spec_bars = ax.bar(
                    x + width/2,
                    spec_vals,
                    width,
                    color=_METRIC_COLORS["specificity"],
                    edgecolor="black",
                    linewidth=0.7,
                    hatch="..",
                    label="Specificity"
                )

                # Add value labels above the bars, omitting NaN cases
                sens_labels = [f"{v:.2f}" if not pd.isna(orig_v) else "" 
                               for orig_v, v in zip(cwe_df["sensitivity"], sens_vals)]
                spec_labels = [f"{v:.2f}" if not pd.isna(orig_v) else "" 
                               for orig_v, v in zip(cwe_df["specificity"], spec_vals)]

                # Slightly larger label sizes above bars (fontsize=14)
                sens_labels_objs = ax.bar_label(sens_bars, labels=sens_labels, fontsize=14, padding=2)
                spec_labels_objs = ax.bar_label(spec_bars, labels=spec_labels, fontsize=14, padding=2)
                
                # Rotate the labels above the bars to 90 degrees
                for label_obj in list(sens_labels_objs) + list(spec_labels_objs):
                    label_obj.set_rotation(90)

                # Subplot title: e.g. "CWE-74 (N = 142)"
                cwe_clean = cwe if cwe.startswith("CWE-") else f"CWE-{cwe}"
                cwe_title = f"{cwe_clean} (N = {supports[cwe]})"
                
                # Increased pad to 15 for 90-degree labels
                ax.set_title(cwe_title, fontsize=18, fontweight="bold", pad=15)
                
                # Set limit to 1.25 to leave padding room above bars for their vertical text labels
                ax.set_ylim(0.0, 1.25)
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                
                ax.set_xticks(x)
                ax.set_xticklabels(tools, rotation=45, ha="right", fontsize=16)
                
                ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#cccccc")
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)

                # Language row label on the leftmost column
                if col_idx == 0:
                    ax.set_ylabel(f"{LANG_TITLES.get(lang, lang)}", fontsize=20, fontweight="bold")

                # Horizontal median reference lines
                if not df.empty:
                    sens_med = float(df["sensitivity"].median())
                    spec_med = float(df["specificity"].median())
                    ax.axhline(sens_med, color=_METRIC_COLORS["sensitivity"], linestyle="--", linewidth=0.6, alpha=0.5)
                    ax.axhline(spec_med, color=_METRIC_COLORS["specificity"], linestyle="--", linewidth=0.6, alpha=0.5)

        # Unified Legend at the bottom
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=_METRIC_COLORS["sensitivity"], hatch="//", edgecolor="black", label="Sensitivity"),
            plt.Rectangle((0, 0), 1, 1, facecolor=_METRIC_COLORS["specificity"], hatch="..", edgecolor="black", label="Specificity")
        ]
        
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=2,
            frameon=False,
            fontsize=24
        )

        fig.tight_layout()

        png_path = out_stem.with_suffix(".png")
        svg_path = out_stem.with_suffix(".svg")
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        print(f"Saved bar plot: {png_path}")
        return [png_path, svg_path]

if __name__ == "__main__":
    args = plot_cli_args("Calibration bar plot.", default_langs=["c_cpp", "java", "python"])

    make_calibration_bars(
        langs=args.langs,
        results_root=args.results_dir,
        out_stem=args.out_stem,
        min_tools=args.min_tools,
        use_sans=True,
    )
