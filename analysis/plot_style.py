"""Shared styling, tool colors/markers, and CLI plumbing for the calibration plot scripts.

Used by calibration_bar_plot.py, calibration_profile_plot.py and calibration_scatter_plot.py
so the three don't each redefine the same rc dict, palette, and --results-dir/--out-stem/
--min-tools/--langs argument parser.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

# ── Colorblind-friendly palette & markers ─────────────────────────────────────
TOOL_COLORS = {
    "bandit": "#E69F00",      # orange
    "codeql": "#56B4E9",      # sky blue
    "cppcheck": "#009E73",    # bluish green
    "flawfinder": "#F0E442",  # yellow
    "ikos": "#0072B2",        # blue
    "joern": "#D55E00",       # vermillion
    "pylint": "#CC79A7",      # reddish purple
    "semgrep": "#000000",     # black
}

TOOL_MARKERS = {
    "bandit": "o",       # Circle
    "codeql": "s",       # Square
    "cppcheck": "^",     # Triangle Up
    "flawfinder": "D",   # Diamond
    "ikos": "v",         # Triangle Down
    "joern": "<",        # Triangle Left
    "pylint": ">",       # Triangle Right
    "semgrep": "X",      # X
}

FALLBACK_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
FALLBACK_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "X"]

LANG_TITLES = {"python": "Python", "java": "Java", "c_cpp": "C/C++"}


def load_calibration(lang: str, results_root: Path, min_tools: int = 2) -> pd.DataFrame:
    """Load calibration CSV, average over folds, filter to top 6 CWEs by support."""
    path = results_root / lang / "pillar_child" / "full" / "calibration_reliability.csv"
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    grp = (
        raw.groupby(["tool", "family"], as_index=False)
        .agg(
            sensitivity=("sensitivity", "mean"),
            specificity=("specificity", "mean"),
            positive_support=("positive_support", "mean"),
            supported=("supported", "max"),
        )
    )
    grp = grp[grp["supported"].astype(bool)].copy()

    # Keep only CWEs supported by >= min_tools
    cwe_n = grp.groupby("family")["tool"].nunique()
    valid_cwes = cwe_n[cwe_n >= min_tools].index
    grp = grp[grp["family"].isin(valid_cwes)].copy()

    # Filter to top 6 CWEs by support
    if not grp.empty:
        family_support = grp.groupby("family")["positive_support"].max()
        top_6_families = family_support.sort_values(ascending=False).head(6).index
        grp = grp[grp["family"].isin(top_6_families)].copy()

    return grp.reset_index(drop=True)


def tool_palette(tools: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    """(colors, markers) for an arbitrary tool set: known tools get their fixed
    color/marker, unknown ones cycle through the fallback palette in sorted order."""
    colors: dict[str, str] = {}
    markers: dict[str, str] = {}
    for i, t in enumerate(sorted(tools)):
        colors[t] = TOOL_COLORS.get(t, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        markers[t] = TOOL_MARKERS.get(t, FALLBACK_MARKERS[i % len(FALLBACK_MARKERS)])
    return colors, markers


def rc_style(use_sans: bool = True, overrides: dict | None = None) -> dict:
    """Base matplotlib rc dict shared by every calibration plot, plus per-script overrides."""
    rc = {
        "font.family": "sans-serif" if use_sans else "serif",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#222222",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
    if use_sans:
        rc["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
    else:
        rc["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    if overrides:
        rc.update(overrides)
    return rc


def plot_cli_args(description: str, default_langs: Sequence[str] = ("c_cpp", "java", "python")) -> argparse.Namespace:
    """Shared --results-dir/--out-stem/--min-tools/--langs CLI for the calibration plot scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--out-stem", default=None)
    parser.add_argument("--min-tools", type=int, default=2)
    parser.add_argument("--langs", nargs="+", default=list(default_langs))
    return parser.parse_args()
