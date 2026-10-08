"""Calibration scatter plot: (sensitivity, specificity) per (tool, CWE-family).

One subplot per language (3 columns). Each point is a (tool, CWE) pair averaged
over folds. Only the top 6 CWEs by support (that are supported by at least 2 tools)
are shown. Points use colorblind-friendly colors and distinct markers for each tool,
sized by positive support, and labelled with the CWE family.

Label strategy & Visual Association
-----------------------------------
Each CWE is labelled at the centroid of its tool-points. To resolve crowding in the
top-left region (high specificity, low sensitivity), we show a zoomed-in inset plot
in the upper-right quadrant of each subplot.
To ensure every point is clearly associated with its CWE:
1. In the main plot, all tool-points for a CWE are connected to its main centroid by thin lines.
2. In the inset plot, all tool-points falling inside the zoom region are connected to their
   local zoom centroid.
3. A greedy repulsion algorithm is run on both axes to avoid overlaps.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.plot_style import LANG_TITLES, load_calibration, plot_cli_args, rc_style, tool_palette

# Language-specific zoom windows for the high-density regions
ZOOM_LIMITS = {
    "python": {
        "x": (-0.015, 0.08),
        "y": (0.98, 1.001)
    },
    "java": {
        "x": (-0.015, 0.08),
        "y": (0.98, 1.001)
    },
    "c_cpp": {
        "x": (-0.01, 0.04),
        "y": (0.98, 1.001)
    }
}


# ── jitter helpers ────────────────────────────────────────────────────────────

def _circular_offsets(n: int, rx: float, ry: float) -> list[tuple[float, float]]:
    if n == 1:
        return [(0.0, 0.0)]
    angles = [2 * math.pi * i / n for i in range(n)]
    return [(rx * math.cos(a), ry * math.sin(a)) for a in angles]


def _apply_jitter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["x_jit"] = df["sensitivity"].astype(float)
    df["y_jit"] = df["specificity"].astype(float)
    x_span = max(df["sensitivity"].max() - df["sensitivity"].min(), 0.1)
    y_span = max(df["specificity"].max() - df["specificity"].min(), 0.05)
    rx = x_span * 0.012
    ry = y_span * 0.012
    for fam, group in df.groupby("family"):
        n = len(group)
        if n <= 1:
            continue
        offsets = _circular_offsets(n, rx, ry)
        for idx, row_idx in enumerate(group.index):
            df.at[row_idx, "x_jit"] += offsets[idx][0]
            df.at[row_idx, "y_jit"] += offsets[idx][1]
    return df


# ── greedy label repulsion ────────────────────────────────────────────────────

def _repulse_labels(
    positions: list[tuple[float, float]],
    x_range: float,
    y_range: float,
    iterations: int = 60,
    min_dist_frac: float = 0.07,
) -> list[tuple[float, float]]:
    """Iteratively push label positions apart so they do not overlap."""
    pos = [(x / x_range, y / y_range) for x, y in positions]
    md = min_dist_frac

    for _ in range(iterations):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx = pos[j][0] - pos[i][0]
                dy = pos[j][1] - pos[i][1]
                dist = math.hypot(dx, dy)
                if dist < md and dist > 1e-9:
                    push = (md - dist) / 2.0
                    nx = dx / dist * push
                    ny = dy / dist * push
                    pos[i] = (pos[i][0] - nx, pos[i][1] - ny)
                    pos[j] = (pos[j][0] + nx, pos[j][1] + ny)
                    moved = True
        if not moved:
            break

    return [(x * x_range, y * y_range) for x, y in pos]


# ── single-axis plot ───────────────────────────────────────────────────────────

def _plot_lang(
    ax: plt.Axes,
    df: pd.DataFrame,
    tool_colors: dict[str, str],
    tool_markers: dict[str, str],
    lang: str,
    size_range: tuple[float, float] = (60, 500),
) -> None:
    if df.empty:
        ax.set_title(LANG_TITLES.get(lang, lang), fontsize=14, fontweight="bold")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="#999999")
        return

    df = _apply_jitter(df)

    # Size scaling
    support = df["positive_support"].clip(lower=1)
    s_min, s_max = float(support.min()), float(support.max())
    span = s_max - s_min if s_max > s_min else 1.0
    sizes = size_range[0] + (support - s_min) / span * (size_range[1] - size_range[0])

    # Draw main plot points
    for tool in sorted(df["tool"].unique()):
        sub = df[df["tool"] == tool]
        ax.scatter(
            sub["x_jit"], sub["y_jit"],
            s=sizes[sub.index],
            color=tool_colors.get(tool, "#888888"),
            marker=tool_markers.get(tool, "o"),
            edgecolors="white", linewidths=0.7,
            alpha=0.88, zorder=3, label=tool,
        )

    # Language-specific zoom boundaries
    limits = ZOOM_LIMITS.get(lang, {"x": (-0.015, 0.08), "y": (0.98, 1.001)})
    x_zoom_min, x_zoom_max = limits["x"]
    y_zoom_min, y_zoom_max = limits["y"]

    # Position of the inset axes: placed in the top-right quadrant of parent
    # bounds: [x0, y0, width, height] in axes fraction coordinates
    axins = ax.inset_axes([0.45, 0.45, 0.50, 0.48])

    # Draw zoomed points in the inset
    for tool in sorted(df["tool"].unique()):
        sub = df[df["tool"] == tool]
        axins.scatter(
            sub["x_jit"], sub["y_jit"],
            s=sizes[sub.index] * 0.75,
            color=tool_colors.get(tool, "#888888"),
            marker=tool_markers.get(tool, "o"),
            edgecolors="white", linewidths=0.6,
            alpha=0.88, zorder=3,
        )

    axins.set_xlim(x_zoom_min, x_zoom_max)
    axins.set_ylim(y_zoom_min, y_zoom_max)
    
    # Configure ticks on inset: 2 decimal places precision for both axes
    axins.tick_params(labelsize=6.5, length=1.5, pad=1)
    axins.grid(True, linestyle=":", linewidth=0.4, color="#cccccc", alpha=0.8)
    axins.xaxis.set_major_locator(plt.MaxNLocator(3))
    axins.yaxis.set_major_locator(plt.MaxNLocator(3))
    axins.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    axins.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Add indicate box with connector lines (handled compatibly for Matplotlib < 3.10 and >= 3.10)
    zoom_indicator = ax.indicate_inset_zoom(axins, edgecolor="#888888", alpha=0.3, lw=0.7)
    if zoom_indicator is not None:
        rectpatch = getattr(zoom_indicator, "rectangle", None)
        if rectpatch is None:
            # Fallback for older Matplotlib versions returning a tuple
            try:
                rectpatch = zoom_indicator[0]
            except Exception:
                rectpatch = None
        if rectpatch:
            rectpatch.set_linestyle("--")

    # Set parent axis limits: X-axis stops exactly at 1.00, Y-axis stops exactly at 1.00
    y_min_val = df["specificity"].min()
    y_lo = float(np.floor(y_min_val * 20) / 20)  # round down to nearest 0.05
    y_lo = max(0.0, y_lo)
    
    ax.set_xlim(-0.05, 1.00)
    ax.set_ylim(y_lo, 1.00)

    x_range = 1.05  # -0.05 to 1.00
    y_range = 1.00 - y_lo

    # Split CWE labels: main plot vs inset plot
    centroids = df.groupby("family")[["sensitivity", "specificity"]].mean()
    


    # ── Labeling: Draw labels OUTSIDE zoom on the main plot ──
    # We label all 6 CWEs in the main plot at their main centroids
    fams_out = []
    anchors_out = []
    for fam, row in centroids.iterrows():
        label_text = str(fam)
        if not label_text.startswith("CWE-"):
            label_text = f"CWE-{label_text}"
        fams_out.append(label_text)
        anchors_out.append((float(row["sensitivity"]), float(row["specificity"])))
        
    mean_x = sum(a[0] for a in anchors_out) / len(anchors_out)
    mean_y = sum(a[1] for a in anchors_out) / len(anchors_out)
    text_pos_out = []
    label_offset = 0.045 * x_range
    for ax_, ay_ in anchors_out:
        dx = ax_ - mean_x
        dy = ay_ - mean_y
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            text_pos_out.append((ax_ + label_offset, ay_))
        else:
            text_pos_out.append((ax_ + dx / dist * label_offset,
                                 ay_ + dy / dist * label_offset * (y_range / x_range)))
                                 
    text_pos_out = _repulse_labels(text_pos_out, x_range, y_range, iterations=60, min_dist_frac=0.08)
    
    for (ax_, ay_), (tx, ty), fam in zip(anchors_out, text_pos_out, fams_out):
        arrow_dist = math.hypot(tx - ax_, ty - ay_)
        use_arrow = arrow_dist > 0.02 * x_range
        ax.annotate(
            fam,
            xy=(ax_, ay_),
            xytext=(tx, ty),
            fontsize=7.5,
            color="#111111",
            fontweight="bold",
            zorder=6,
            ha="center", va="center",
            clip_on=False,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            arrowprops=dict(
                arrowstyle="-",
                color="#aaaaaa",
                lw=0.6,
            ) if use_arrow else None,
        )

    # ── Labeling: Draw labels for points inside the zoom region in the inset axes ──
    # We only label CWEs that have at least one point inside the zoom region
    inside_zoom = []
    for fam, group in df.groupby("family"):
        sub_group = group[
            (group["x_jit"] >= x_zoom_min) & (group["x_jit"] <= x_zoom_max) &
            (group["y_jit"] >= y_zoom_min) & (group["y_jit"] <= y_zoom_max)
        ]
        if not sub_group.empty:
            label_text = str(fam)
            if not label_text.startswith("CWE-"):
                label_text = f"CWE-{label_text}"
            inside_zoom.append((label_text, float(sub_group["sensitivity"].mean()), float(sub_group["specificity"].mean())))

    if inside_zoom:
        fams_in = [item[0] for item in inside_zoom]
        anchors_in = [(item[1], item[2]) for item in inside_zoom]
        
        x_ins_range = x_zoom_max - x_zoom_min
        y_ins_range = y_zoom_max - y_zoom_min
        
        mean_x_in = sum(a[0] for a in anchors_in) / len(anchors_in)
        mean_y_in = sum(a[1] for a in anchors_in) / len(anchors_in)
        
        rel_anchors = [(ax_ - x_zoom_min, ay_ - y_zoom_min) for ax_, ay_ in anchors_in]
        rel_mean_x = mean_x_in - x_zoom_min
        rel_mean_y = mean_y_in - y_zoom_min
        
        rel_text_pos = []
        label_offset_in = 0.08 * x_ins_range
        for rx, ry_ in rel_anchors:
            dx = rx - rel_mean_x
            dy = ry_ - rel_mean_y
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                rel_text_pos.append((rx + label_offset_in, ry_))
            else:
                rel_text_pos.append((rx + dx / dist * label_offset_in,
                                     ry_ + dy / dist * label_offset_in * (y_ins_range / x_ins_range)))
                                     
        rel_repulsed = _repulse_labels(rel_text_pos, x_ins_range, y_ins_range, iterations=80, min_dist_frac=0.18)
        text_pos_in = [(tx + x_zoom_min, ty + y_zoom_min) for tx, ty in rel_repulsed]
        
        for (ax_, ay_), (tx, ty), fam in zip(anchors_in, text_pos_in, fams_in):
            arrow_dist = math.hypot(tx - ax_, ty - ay_)
            use_arrow = arrow_dist > 0.02 * x_ins_range
            axins.annotate(
                fam,
                xy=(ax_, ay_),
                xytext=(tx, ty),
                fontsize=7.5,
                color="#111111",
                fontweight="bold",
                zorder=6,
                ha="center", va="center",
                clip_on=False,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                arrowprops=dict(
                    arrowstyle="-",
                    color="#bbbbbb",
                    lw=0.6,
                ) if use_arrow else None,
            )

    # Main plot labels + formatting
    ax.set_xlabel("Sensitivity (recall)", fontsize=10)
    ax.set_ylabel("Specificity", fontsize=10)
    ax.set_title(LANG_TITLES.get(lang, lang), fontsize=13, fontweight="bold", pad=8)

    ax.grid(True, linestyle=":", linewidth=0.6, color="#cccccc", alpha=0.9, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Median reference lines
    sens_med = float(df["sensitivity"].median())
    spec_med = float(df["specificity"].median())
    ax.axvline(sens_med, color="#bbbbbb", linewidth=0.7, linestyle="--", zorder=1)
    ax.axhline(spec_med, color="#bbbbbb", linewidth=0.7, linestyle="--", zorder=1)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[1] - 0.005 * (xlim[1]-xlim[0]), spec_med + 0.003 * (ylim[1]-ylim[0]),
            f"med={spec_med:.2f}", fontsize=6.5, color="#777777", ha="right", va="bottom")
    ax.text(sens_med + 0.005 * (xlim[1]-xlim[0]), ylim[0] + 0.003 * (ylim[1]-ylim[0]),
            f"med={sens_med:.2f}", fontsize=6.5, color="#777777", ha="left", va="bottom")


# ── size legend ───────────────────────────────────────────────────────────────

def _size_legend_handles(
    support_values: pd.Series,
    size_range: tuple[float, float] = (60, 500),
    n: int = 3,
) -> list[Line2D]:
    s_min, s_max = float(support_values.min()), float(support_values.max())
    span = s_max - s_min if s_max > s_min else 1.0
    targets = np.linspace(s_min, s_max, n)
    handles = []
    for t in targets:
        sz = size_range[0] + (t - s_min) / span * (size_range[1] - size_range[0])
        h = Line2D(
            [], [], marker="o", linestyle="None",
            markersize=float(np.sqrt(sz)) * 0.55,
            color="#888888", alpha=0.7,
            label=f"n≈{int(t)}",
        )
        handles.append(h)
    return handles


# ── main figure ───────────────────────────────────────────────────────────────

def make_calibration_scatter(
    langs: Sequence[str] = ("python", "java", "c_cpp"),
    results_root: str | Path = "data/results",
    out_stem: str | Path | None = None,
    min_tools: int = 2,
    use_sans: bool = True,
) -> list[Path]:
    results_root = Path(results_root)
    if out_stem is None:
        out_stem = results_root / "calibration_scatter"
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
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, n_langs, figsize=(6.5 * n_langs, 5.5), sharey=False)
        if n_langs == 1:
            axes = [axes]

        size_range: tuple[float, float] = (60, 480)
        for ax, lang in zip(axes, langs):
            _plot_lang(ax, data[lang], tool_colors, tool_markers, lang, size_range=size_range)

        tool_handles = [
            Line2D([], [], marker=tool_markers[t], linestyle="None", markersize=8,
                   color=tool_colors[t], markeredgecolor="white", markeredgewidth=0.5,
                   label=t)
            for t in sorted(all_tools)
        ]
        all_support = pd.concat(
            [df["positive_support"] for df in data.values() if not df.empty],
            ignore_index=True,
        ).clip(lower=1)
        size_handles = _size_legend_handles(all_support, size_range=size_range, n=3)

        leg_tools = fig.legend(
            handles=tool_handles, title="Tool",
            loc="lower left", bbox_to_anchor=(0.01, -0.16),
            ncol=min(len(all_tools), 9),
            frameon=False, fontsize=9, title_fontsize=9,
            handletextpad=0.4, columnspacing=1.2,
        )
        fig.legend(
            handles=size_handles, title="GT support",
            loc="lower right", bbox_to_anchor=(0.99, -0.16),
            ncol=3, frameon=False, fontsize=9, title_fontsize=9,
            handletextpad=0.4, columnspacing=1.2,
        )
        fig.add_artist(leg_tools)

        fig.suptitle(
            "Per-tool, per-CWE calibration — sensitivity vs. specificity\n"
            "(each point = one tool on one CWE family, averaged over folds; "
            "size ∝ GT support; showing top 6 CWE families per language; dashed lines = per-subplot medians)",
            fontsize=10, y=1.03,
        )

        fig.tight_layout(rect=(0, 0.08, 1, 1))

        # Save primary output as clean name (sans-serif)
        png_path = out_stem.with_suffix(".png")
        svg_path = out_stem.with_suffix(".svg")
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        
        # Also save with suffix for backwards compatibility
        suffix = "sans" if use_sans else "serif"
        png_compat = out_stem.parent / f"{out_stem.name}_{suffix}.png"
        svg_compat = out_stem.parent / f"{out_stem.name}_{suffix}.svg"
        fig.savefig(png_compat, dpi=180, bbox_inches="tight")
        fig.savefig(svg_compat, bbox_inches="tight")
        
        print(f"Saved: {png_path}")
        print(f"Saved: {svg_path}")
        return [png_path, svg_path]


if __name__ == "__main__":
    args = plot_cli_args(
        "Calibration scatter plot (sens vs spec per tool x CWE).",
        default_langs=["python", "java", "c_cpp"],
    )
    make_calibration_scatter(
        langs=args.langs, results_root=args.results_dir,
        out_stem=args.out_stem, min_tools=args.min_tools, use_sans=True,
    )
