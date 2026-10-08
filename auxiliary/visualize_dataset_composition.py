"""
visualize_dataset_composition.py
---------------------------------
Single stacked horizontal bar chart showing the CWE-family composition
of the three merged datasets (C/C++, Java, Python).

Layout
------
  • One bar per language (3 rows: C/C++, Java, Python)
  • Each coloured segment = one canonical CWE family (pillar_child)
  • Colour is consistent across all three bars
  • Right-side legend: "CWE-XXX: Full Name" with sample count
  • Only families with >= MIN_VULN_COUNT vuln samples (mirrors tier-full)

For C/C++ uses c_cpp_merged.parquet (PrimeVul sampled to 10k).

Run from FEAST project root:
    uv run python auxiliary/visualize_dataset_composition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.canonical import CweCanonicalizer, CANONICAL_LEVEL
from analysis.dataset import as_list
from ingestion.cwe_navigator import CWENavigator

# ── configuration ─────────────────────────────────────────────────────────────

MERGED_DIR   = ROOT / "data" / "merged"
CWE_XML      = ROOT / "data" / "cwec_latest.xml"
OUT_DIR      = ROOT / "auxiliary" / "figures"
OUT_FILENAME = "dataset_composition_overview.pdf"

# Only show families with >= this many vuln samples (mirrors tier-full threshold)
MIN_VULN_COUNT = 100

LANGUAGES = [
    ("C/C++",  "c_cpp_merged.parquet"),
    ("Java",   "java_merged.parquet"),
    ("Python", "python_merged.parquet"),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def canonicalise_cwes(cwes_series: pd.Series, canon: CweCanonicalizer) -> pd.Series:
    def _map(raw_list) -> list[str]:
        families = {canon.family(c) for c in as_list(raw_list)}
        return sorted(f for f in families if f is not None)
    return cwes_series.map(_map)


def cwe_vuln_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count vuln samples per canonical family."""
    counts: dict[str, int] = {}
    for _, row in df[df["label"] == 1].iterrows():
        for fam in row["canonical_cwes"]:
            counts[fam] = counts.get(fam, 0) + 1
    return counts


# Overrides for names that are too long for the legend
NAME_OVERRIDES: dict[str, str] = {
    "CWE-74": "Improper Neutralization of Special Elements ('Injection')",
}


def cwe_name(nav: CWENavigator, cwe_id: str) -> str:
    """Return a (possibly shortened) MITRE name for a canonical family ID."""
    if cwe_id in NAME_OVERRIDES:
        return NAME_OVERRIDES[cwe_id]
    num  = cwe_id.removeprefix("CWE-")
    name = nav.get_element_name(num)
    return name if name != "Unknown" else ""


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CWE canonicalizer and navigator ...")
    canon = CweCanonicalizer.from_xml(CANONICAL_LEVEL, str(CWE_XML))
    nav   = CWENavigator(str(CWE_XML))

    # ── load data ─────────────────────────────────────────────────────────────
    lang_counts: dict[str, dict[str, int]] = {}   # lang_label -> {family -> count}

    for lang_label, fname in LANGUAGES:
        path = MERGED_DIR / fname
        if not path.exists():
            print(f"[SKIP] {path} not found")
            continue
        print(f"Loading {fname} ...")
        df = pd.read_parquet(path)
        df["canonical_cwes"] = canonicalise_cwes(df["cwes"], canon)
        counts = cwe_vuln_counts(df)
        lang_counts[lang_label] = counts

    if not lang_counts:
        print("No data found. Aborting.")
        return

    # ── collect all families that pass the threshold in ANY language ──────────
    all_families_global: set[str] = set()
    for counts in lang_counts.values():
        for fam, cnt in counts.items():
            if cnt >= MIN_VULN_COUNT:
                all_families_global.add(fam)

    # Order families by total vuln count across all languages (descending)
    family_totals = {
        fam: sum(lang_counts[lang].get(fam, 0) for lang in lang_counts)
        for fam in all_families_global
    }
    families = sorted(all_families_global, key=lambda f: family_totals[f], reverse=True)

    print(f"\nFamilies passing threshold (>= {MIN_VULN_COUNT}): {len(families)}")
    for fam in families:
        name = cwe_name(nav, fam)
        print(f"  {fam}: {name}  (total={family_totals[fam]:,})")

    # ── build colour palette ──────────────────────────────────────────────────
    # Use a high-contrast qualitative palette
    n = len(families)
    # Combine tab10 + tab20b for up to ~30 distinct colours
    palette_src = (
        plt.get_cmap("tab10").colors +
        plt.get_cmap("tab20b").colors +
        plt.get_cmap("tab20c").colors
    )
    colours = {fam: palette_src[i % len(palette_src)] for i, fam in enumerate(families)}

    # ── figure ────────────────────────────────────────────────────────────────
    n_langs      = len(lang_counts)
    bar_h        = 0.55
    y_gap        = 1.0
    n_legend     = len(families)
    ncol         = 3
    legend_rows  = (n_legend + ncol - 1) // ncol  # 3-column legend
    legend_h_in  = legend_rows * 0.31 + 0.75      # height of legend box
    bars_h_in    = n_langs * y_gap + 1.6          # extra space for labels
    fig_h        = bars_h_in + legend_h_in

    # fraction of figure height reserved for the bars area
    bars_frac    = bars_h_in / fig_h

    fig, ax = plt.subplots(figsize=(16, fig_h))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F0F4F8")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(axis="x", color="white", linewidth=1.4, zorder=0)
    # Place the axes in the top portion, leaving the bottom for the legend.
    # We add a small pad above and below the bars area.
    pad = 0.10  # balanced pad to clear space for the X-axis label
    ax_bottom = (legend_h_in + pad) / fig_h   # bottom edge of bar axes
    ax_top    = (fig_h - 0.1) / fig_h          # top edge of bar axes
    ax.set_position([0.08, ax_bottom, 0.89, ax_top - ax_bottom])

    # C/C++ on top → reverse
    lang_order      = [lbl for lbl, _ in LANGUAGES if lbl in lang_counts]
    lang_order_plot = list(reversed(lang_order))
    y_positions: dict[str, float] = {}

    for i, lang_label in enumerate(lang_order_plot):
        counts    = lang_counts[lang_label]
        y         = i * y_gap
        y_positions[lang_label] = y

        bar_total = sum(counts.get(f, 0) for f in families)
        if bar_total == 0:
            continue

        left = 0.0
        for fam in families:
            val = counts.get(fam, 0)
            if val == 0:
                continue
            pct = val / bar_total * 100.0
            ax.barh(
                y, pct, left=left,
                height=bar_h,
                color=colours[fam],
                edgecolor="white", linewidth=0.6,
                zorder=2,
            )
            left += pct

        ax.text(
            101.5, y,
            f"n={sum(counts.get(f,0) for f in families):,}",
            va="center", ha="left",
            fontsize=9, color="#1E293B", fontweight="bold",
            zorder=3,
        )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()),
                       fontsize=12, fontweight="bold", color="#1E293B")
    ax.set_xlim(0, 115)
    ax.set_ylim(-0.65, (n_langs - 1) * y_gap + 0.65)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}%")
    )
    ax.set_xlabel(
        f"Proportion of vulnerable samples",
        fontsize=10, color="#475569", labelpad=8,
    )
    ax.tick_params(axis="x", labelsize=9, colors="#475569")
    
    # ax.set_title(
    #     "CWE-family composition of merged datasets",
    #     fontsize=14, fontweight="bold", color="#0F172A", pad=14,
    # )

    # ── legend below the bars, in figure coordinates ──────────────────────
    legend_handles = []
    for fam in families:
        name  = cwe_name(nav, fam)
        label = f"{fam}  –  {name}"
        patch = mpatches.Patch(color=colours[fam], label=label)
        legend_handles.append(patch)

    legend = fig.legend(
        handles=legend_handles,
        title="CWE Family",
        title_fontsize=9.5,
        fontsize=8.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.15),
        ncol=3,
        framealpha=0.95,
        edgecolor="#CBD5E1",
        borderpad=0.8,
        labelspacing=0.5,
        handlelength=1.2,
        handleheight=0.85,
    )
    legend.get_title().set_fontweight("bold")

    # ── save ──────────────────────────────────────────────────────────────────
    out_pdf = OUT_DIR / OUT_FILENAME
    out_png = OUT_DIR / OUT_FILENAME.replace(".pdf", ".png")
    # Do NOT use bbox_inches="tight" – we want fixed figure dimensions
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15, facecolor=fig.get_facecolor(), dpi=150)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.15, dpi=200, facecolor=fig.get_facecolor())
    print(f"\nSaved -> {out_pdf}")
    print(f"Saved -> {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()

