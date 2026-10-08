import argparse
from pathlib import Path

import pandas as pd

from analysis.mean_difference_ci import (
    STATUS_INCONCLUSIVE,
    STATUS_LABELS,
    STATUS_STYLES,
    annotate_pvalue,
    strategy_label,
)
from analysis.plot_style import rc_style


def make_combined_plot(use_sans=False, results_root="data/results", metric="f1"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    rc = rc_style(use_sans, {
        "axes.edgecolor": "#000000", "xtick.color": "#000000", "ytick.color": "#000000",
        "text.color": "#000000", "font.size": 18, "axes.labelsize": 18,
        "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 22,
    })

    with plt.rc_context(rc):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 11))
        axes = [ax1, ax2, ax3]
        langs = ["c_cpp", "java", "python"]
        lang_titles = {"c_cpp": "C/C++", "java": "Java", "python": "Python"}

        for ax, lang in zip(axes, langs):
            csv_path = (
                Path(results_root)
                / lang / "pillar_child" / "full" / "plots"
                / f"mean_difference_ci_{metric}.csv"
            )
            if not csv_path.exists():
                print(f"Warning: {csv_path} does not exist. Skipping.")
                continue

            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            if "baseline" in df.columns:
                unique_baselines = df["baseline"].unique()
                if "traditional_2" in unique_baselines:
                    df = df[df["baseline"] == "traditional_2"].copy()
                elif "traditional_3" in unique_baselines:
                    df = df[df["baseline"] == "traditional_3"].copy()
                elif "traditional" in unique_baselines:
                    df = df[df["baseline"] == "traditional"].copy()

            strategies = df["strategy"].astype(str).tolist()
            labels = [strategy_label(s) for s in strategies]
            y_positions = list(range(len(strategies)))

            for y in y_positions:
                if y % 2:
                    ax.axhspan(y - 0.5, y + 0.5, color="#f7f7f7", zorder=0)
            ax.axvline(0.0, color="#000000", linewidth=0.9, linestyle="--", zorder=1)

            for y, (_idx, row) in zip(y_positions, df.iterrows()):
                hl = float(row["hodges_lehmann"])
                ci_low = float(row["ci_low"])
                ci_high = float(row["ci_high"])
                status = str(row["status"])
                style = STATUS_STYLES.get(status, STATUS_STYLES[STATUS_INCONCLUSIVE])

                if pd.isna(hl) or pd.isna(ci_low) or pd.isna(ci_high):
                    continue

                ax.errorbar(
                    hl, y,
                    xerr=[[hl - ci_low], [ci_high - hl]],
                    fmt=style["marker"],
                    color=style["color"],
                    ecolor=style["color"],
                    elinewidth=1.1,
                    capsize=2.5,
                    capthick=1.0,
                    markersize=6.0,
                    markeredgewidth=0.8,
                    markeredgecolor="#000000",
                    zorder=3,
                )

                annotate_pvalue(ax, hl, y, row)

            ax.set_yticks(y_positions, labels)
            ax.invert_yaxis()
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(lang_titles[lang], fontsize=26, fontweight="bold", pad=12)
            ax.grid(axis="x", linestyle=":", linewidth=0.6, color="#bdbdbd", alpha=0.8)
            ax.tick_params(axis="both", labelsize=18)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

        legend_handles = [
            mlines.Line2D(
                [], [], color=style["color"], marker=style["marker"], linestyle="None",
                markeredgecolor="#000000", markeredgewidth=0.8, markersize=8,
                label=STATUS_LABELS[status],
            )
            for status, style in STATUS_STYLES.items()
        ]

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=3,
            frameon=False,
            fontsize=28,
            handletextpad=0.4,
            columnspacing=1.5,
        )

        fig.tight_layout(rect=(0, 0.02, 1, 0.98))

        suffix = "sans" if use_sans else "serif"
        stem = "combined_ci" if metric == "f1" else f"combined_ci_{metric}"
        out_stem = Path(results_root) / f"{stem}_{suffix}"
        fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined plot: {out_stem.with_suffix('.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="data/results")
    parser.add_argument("--metric", type=str, default="f1",
                         help="Metric whose mean_difference_ci_<metric>.csv to plot (default: f1)")
    args = parser.parse_args()

    make_combined_plot(use_sans=False, results_root=args.results_dir, metric=args.metric)
    make_combined_plot(use_sans=True, results_root=args.results_dir, metric=args.metric)
