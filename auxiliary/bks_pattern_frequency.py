"""BKS fire-pattern frequency heatmap: substantiates Reviewer #2's Minor Issue #2.

The reviewer asks why BKS degrades significantly on Java (N=3 tools, 2^3=8 cells) but
not on Python, despite Python also using a small tool set (N=4, 2^4=16 cells), and
suggests the root cause is the distribution of fire patterns across the cell space
(some combinations may never occur in Java) rather than cell count alone.

BKS (analysis/fusion/bks.py) fits, per family, one empirical P(vulnerable | pattern)
table over all 2^N possible tool fire patterns. A pattern with few or zero calibration
observations produces an unreliable (or fully prior-shrunk) estimate for that cell. This
script reuses the exact same pattern definition as bks_predictions and reports, per
(language, family), how the data population is spread across the 2^N cells -- which is
the frequency distribution the reviewer asked to see, not a hypothesis about it.

Runs on every family in the tier's restriction by default (not just the six most
populated ones shown in Fig. 3), so the sparsity picture is not survivorship-biased
toward the families that already have the most data; pass --top-n to restrict it.

Requires data/enriched/<lang>.parquet and data/cwec_latest.xml (neither ships in the
git repo).

    uv run python auxiliary/bks_pattern_frequency.py --lang java
    uv run python auxiliary/bks_pattern_frequency.py --lang java,python --tier full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.calibration import gt_family_support
from analysis.experiment import prepare_canonical
from analysis.fusion import TIER_MIN_COUNT
from analysis.fusion.common import build_fire_index, precompute_labels


def _pattern_bitstring(pattern: tuple[bool, ...]) -> str:
    return "".join("1" if p else "0" for p in pattern)


def run_lang(lang: str, tier: str, top_n: int | None, seed: int) -> pd.DataFrame:
    min_cwe_count = TIER_MIN_COUNT.get(tier, 5)
    try:
        prep = prepare_canonical(lang, "pillar_child", n_splits=5, min_cwe_count=min_cwe_count, seed=seed)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[{lang}] could not prepare data -- {exc}")
        return pd.DataFrame()

    cdf, tools, families = prep.cdf, prep.tools, prep.families
    n_tools = len(tools)
    if not families:
        print(f"[{lang}] no families survive restriction.")
        return pd.DataFrame()

    support = gt_family_support(cdf, families)
    ranked = sorted(families, key=lambda f: support.get(f, 0), reverse=True)
    top_families = ranked if top_n is None else ranked[:top_n]

    fire, _ = build_fire_index(cdf, tools)
    labels = precompute_labels(cdf, families)
    n = len(cdf)

    rows = []
    for family in top_families:
        counts: dict[tuple, int] = {}
        pos: dict[tuple, int] = {}
        for i in range(n):
            p = tuple(family in fire[(i, t)] for t in tools)
            counts[p] = counts.get(p, 0) + 1
            if labels[(i, family)]:
                pos[p] = pos.get(p, 0) + 1

        all_patterns = [tuple(bool(int(b)) for b in format(k, f"0{n_tools}b")) for k in range(2 ** n_tools)]
        n_empty = sum(1 for p in all_patterns if counts.get(p, 0) == 0)
        n_singleton = sum(1 for p in all_patterns if counts.get(p, 0) == 1)
        dominant_pattern, dominant_count = max(counts.items(), key=lambda kv: kv[1])

        for p in all_patterns:
            c = counts.get(p, 0)
            rows.append({
                "language": lang, "family": family, "pattern": _pattern_bitstring(p),
                "count": c, "n_pos": pos.get(p, 0),
                "p_vuln_empirical": (pos.get(p, 0) / c) if c else None,
                "n_cells": 2 ** n_tools, "n_empty_cells": n_empty, "n_singleton_cells": n_singleton,
                "dominant_pattern": _pattern_bitstring(dominant_pattern),
                "dominant_pattern_share": dominant_count / n if n else None,
                "family_support": support.get(family, 0),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="java,python", help="comma-separated: java,python,c_cpp")
    ap.add_argument("--tier", default="full", choices=["base", "medium", "full"])
    ap.add_argument("--top-n", type=int, default=None, help="restrict to the N most-supported families; default runs on every family in scope")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    langs = [s.strip() for s in args.lang.split(",") if s.strip()]
    frames = [run_lang(lang, args.tier, args.top_n, args.seed) for lang in langs]
    result = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()

    if result.empty:
        print("\nNo data available -- see the messages above.")
        return

    print("\n=== BKS fire-pattern population, per (language, family) ===")
    summary = result.drop_duplicates(["language", "family"])[
        ["language", "family", "family_support", "n_cells", "n_empty_cells",
         "n_singleton_cells", "dominant_pattern", "dominant_pattern_share"]
    ]
    print(summary.to_string(index=False))

    print("\n=== Overall sparsity summary per language ===")
    overall = summary.groupby("language").agg(
        n_families=("family", "count"),
        avg_empty_cells=("n_empty_cells", "mean"),
        avg_singleton_cells=("n_singleton_cells", "mean"),
        avg_dominant_pattern_share=("dominant_pattern_share", "mean"),
    ).reset_index()
    print(overall.to_string(index=False))

    out_dir = ROOT / "data" / "results" / "_cross_language"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "bks_pattern_frequency.csv", index=False)
    print(f"\nSaved per-cell breakdown to {out_dir / 'bks_pattern_frequency.csv'}")

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        for lang in result["language"].unique():
            sub = result[result["language"] == lang]
            # Families heaviest first, so the reader meets the ones that carry the result.
            fams = list(sub.groupby("family")["family_support"].first()
                        .sort_values(ascending=False).index)
            # Drop patterns no family ever observes. With 6 tools that is most of the 64
            # columns, and keeping them makes the axis unreadable while showing nothing:
            # an empty column is exactly the absence the figure is meant to convey, and
            # the count of dropped columns says it more precisely than 40 blank stripes.
            populated = (sub.groupby("pattern")["count"].sum() > 0)
            patterns = sorted(populated[populated].index)
            n_dropped = int((~populated).sum())

            grid = np.full((len(fams), len(patterns)), np.nan)
            for _, r in sub.iterrows():
                if r["pattern"] in patterns and r["count"] > 0:
                    grid[fams.index(r["family"]), patterns.index(r["pattern"])] = r["count"]

            fig, ax = plt.subplots(figsize=(max(6, len(patterns) * 1.05 + 2.5),
                                            max(3, len(fams) * 0.65 + 1.8)))
            cmap = plt.get_cmap("viridis").with_extremes(bad="#f2f2f2")
            im = ax.imshow(np.log10(grid), aspect="auto", cmap=cmap)
            # Annotate each populated cell: on this data the interesting quantity is how
            # few samples most cells hold, which a colour scale cannot convey.
            for i in range(len(fams)):
                for j in range(len(patterns)):
                    if not np.isnan(grid[i, j]):
                        v = int(grid[i, j])
                        ax.text(j, i, f"{v:,}" if v < 10000 else f"{v/1000:.0f}k",
                                ha="center", va="center", fontsize=13,
                                color="#000000" if np.log10(v) > 3.2 else "#ffffff")
            ax.set_xticks(range(len(patterns)))
            ax.set_xticklabels(patterns, rotation=90, fontsize=14, family="monospace")
            ax.set_yticks(range(len(fams)))
            ax.set_yticklabels(fams, fontsize=15)
            cbar = fig.colorbar(im, ax=ax, label="log10(samples in cell)")
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label("log10(samples in cell)", fontsize=13)
            fig.tight_layout()
            out_path = out_dir / f"bks_pattern_heatmap_{lang}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved heatmap to {out_path}")
    except ImportError:
        print("matplotlib not available -- CSV written, skipping heatmap image.")


if __name__ == "__main__":
    main()
