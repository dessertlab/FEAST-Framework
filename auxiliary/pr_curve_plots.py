"""Support-weighted precision-recall curves: three panels, one per language.

Each curve is built by the *same* aggregation rule as every headline number in the paper
(see ``analysis/aggregation.py``): precision and recall are computed inside each CWE
family, averaged over the 5 held-out folds, then averaged across families weighted by
family support. The curve is parameterised by the decision threshold tau, so each plotted
point is literally the support-weighted (recall, precision) pair the headline table would
report at that tau.

Why threshold-averaging and not the usual vertical averaging
------------------------------------------------------------
The instinctive way to average PR curves -- fix a recall grid, read each family's
precision there, average -- does not survive contact with this data:

* interpolating between PR operating points is wrong in general (precision does not vary
  linearly with recall between two points), so most of a fine recall grid would be
  fabricated rather than measured;
* the scores here are extremely discrete. Every fusion score is a deterministic function
  of an at-most-2^N-valued binary fire vector (N <= 6 tools), so a single family's curve
  has very few genuine operating points;
* families cover different recall ranges, so a shared recall grid would extrapolate for
  whichever families never reach the high-recall end.

Averaging along tau instead -- weight *both* coordinates at each threshold -- introduces
no interpolation at all and applies the project's weighting rule unchanged.

Reading the figure
------------------
* **The baseline is a single point, not a curve.** ``traditional_K_of_N`` is a fixed
  voting rule with no threshold to sweep, so it has exactly one operating point. It is
  drawn as a cross.
* **Curves have different lengths.** A strategy whose score saturates never reaches high
  recall at any tau; its curve is genuinely short, not truncated.
* **The points are written to ``pr_curve_weighted_points.csv``** next to the figure, with
  the baseline as a tau=NaN row, so any claim about the curves can be rechecked without
  re-running the fusion.
* **The area under these curves is NOT the PR-AUC.** The scalar PR-AUC reported elsewhere
  is the support-weighted mean of each family's average precision, a different quantity
  (see ``auxiliary/pr_auc_ci_report.py``); the area itself is deliberately not quantified.

This re-runs calibration + fusion from scratch (raw per-row scores are not persisted by
the main pipeline) using the same 5-fold split, tools and family set as
``main.py fusion --tier full --calibration sensitivity,specificity``. The curve keeps the
pipeline's tau *range* [0.1, 0.9] but refines its *resolution*: at the reported 0.1 step
the interesting part of the curve is sampled badly (weighted recall can halve between two
adjacent taus). This changes nothing about the reported results -- tau selection still
happens on the pipeline's own {0.1..0.9} grid, in the pipeline.

    uv run python auxiliary/pr_curve_plots.py --lang all
    uv run python auxiliary/pr_curve_plots.py --lang java --tau-step 0.01
    uv run python auxiliary/pr_curve_plots.py --from-points   # redraw only, no fusion re-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.aggregation import SUPPORT_COLUMN, aggregate_fusion_metrics
from analysis.calibration import compute_reliability
from analysis.experiment import prepare_canonical
from analysis.folds import split_train_validation
from analysis.fusion import run_fusion
from analysis.fusion.common import canonical_strategy_order, split_tau_strategy
from analysis.fusion.predictions import evaluate_predictions
from analysis.mean_difference_ci import fusion_strategies, strategy_label
from analysis.plot_style import FALLBACK_COLORS, LANG_TITLES, rc_style

# The fusers' continuous score does not depend on tau, so one materialised tau variant is
# enough to recover it; the curve re-thresholds that score on its own grid.
REPRESENTATIVE_TAU = 0.5
# The grid the pipeline reports and selects on; marked on the curve so the reader can see
# how few operating points there really are behind it.
REPORTED_TAUS = np.round(np.arange(0.1, 0.91, 0.1), 6)
BASELINE_PREFIX = "traditional_"
CURVE_KEY_SEP = "::"


def _fine_taus(step: float, tau_min: float = 0.1, tau_max: float = 0.9) -> np.ndarray:
    """Threshold grid for the curve, over the same tau range the pipeline reports.

    Only the *resolution* is refined, not the range: staying inside [0.1, 0.9] keeps the
    curve on the interval the paper declares, and avoids the extreme-tau region where the
    weighted precision collapses for a reason that has nothing to do with the ranking --
    see ``_pr_at_thresholds`` on zero-filling.
    """
    if not 0.0 < step <= 0.5:
        raise ValueError(f"--tau-step must be in (0, 0.5], got {step}")
    if not 0.0 <= tau_min <= tau_max <= 1.0:
        raise ValueError(f"tau range must satisfy 0 <= min <= max <= 1, got [{tau_min}, {tau_max}]")
    return np.round(np.arange(tau_min, tau_max + step / 2, step), 6)


def _pr_at_thresholds(
    labels: np.ndarray,
    scores: np.ndarray,
    taus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall of ``scores >= tau`` for every tau, in one pass.

    Sorting once and reading cumulative counts at each threshold keeps this O(n log n)
    instead of one full scan per tau, which matters because the per-family slices are
    large and the grid is deliberately fine.

    Precision is 0.0 where the strategy makes no positive prediction at that tau -- the
    project-wide ``zero_fill`` convention, applied here at the point the metric is born.
    This shapes the left end of every curve: as tau rises, more families fall silent and
    enter the weighted mean as 0.0, so the weighted precision drops even though the
    surviving predictions are getting *more* selective. That is the convention working as
    intended (abandoning a family is a failure, not a gap), but it means the low-recall end
    of a curve reads as "how many families has this strategy given up on", not "how precise
    is it when it speaks".
    Recall is 0/0 only when the slice holds no positives at all, which cannot happen for a
    family that passed the support filter, but is returned as NaN if it ever does.
    """
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)

    cum_tp = np.concatenate(([0], np.cumsum(sorted_labels)))
    n_positive = int(sorted_labels.sum())

    # Number of samples scoring >= tau: the sorted scores descend, so searching the
    # negated array gives the cut point directly.
    k = np.searchsorted(-sorted_scores, -taus, side="right")
    tp = cum_tp[k].astype(float)
    predicted_positive = k.astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted_positive > 0, tp / predicted_positive, 0.0)
        recall = np.full_like(tp, np.nan) if n_positive == 0 else tp / n_positive
    return precision, recall


def _curve_points(
    per_fold_scores: list[pd.DataFrame],
    taus: np.ndarray,
) -> pd.DataFrame:
    """Support-weighted (recall, precision) per (strategy, tau), via the two-stage rule.

    Builds a per-(fold, strategy+tau, family) table shaped exactly like the pipeline's own
    per-family metrics, then hands it to ``aggregate_fusion_metrics`` so the fold-mean and
    the support-weighted family mean are the shared implementation, not a copy of it.
    """
    rows: list[dict] = []
    for fold_index, frame in enumerate(per_fold_scores):
        for (strategy, family), group in frame.groupby(["base_strategy", "family"], sort=False):
            labels = group["label"].to_numpy(dtype=bool)
            precision, recall = _pr_at_thresholds(labels, group["score"].to_numpy(dtype=float), taus)
            n_positive = int(labels.sum())
            for tau, p, r in zip(taus, precision, recall, strict=True):
                rows.append({
                    "fold": fold_index,
                    "strategy": f"{strategy}{CURVE_KEY_SEP}{tau:.4f}",
                    "family": family,
                    "precision": p,
                    "recall": r,
                    SUPPORT_COLUMN: n_positive,
                })
    if not rows:
        return pd.DataFrame(columns=["base_strategy", "tau", "recall", "precision"])

    _per_family, overall = aggregate_fusion_metrics(
        pd.DataFrame(rows), metric_columns=("precision", "recall")
    )
    keys = overall["strategy"].astype(str).str.rsplit(CURVE_KEY_SEP, n=1, expand=True)
    overall["base_strategy"] = keys[0]
    overall["tau"] = keys[1].astype(float)
    return (
        overall[["base_strategy", "tau", "recall_weighted", "precision_weighted"]]
        .rename(columns={"recall_weighted": "recall", "precision_weighted": "precision"})
        .sort_values(["base_strategy", "tau"])
        .reset_index(drop=True)
    )


def _collect(language: str, tier: str, n_splits: int, seed: int) -> dict:
    """Run the 5 folds and return everything the figure needs for one language."""
    from analysis.fusion import TIER_MIN_COUNT

    prep = prepare_canonical(language, "pillar_child", n_splits=n_splits,
                             min_cwe_count=TIER_MIN_COUNT.get(tier, n_splits), seed=seed)
    cdf, tools, families, folds = prep.cdf, prep.tools, prep.families, prep.folds

    per_fold_scores: list[pd.DataFrame] = []
    per_fold_metrics: list[pd.DataFrame] = []
    for fold in sorted(folds["fold"].unique()):
        print(f"  [{language}] fold {fold}: calibrating + fusing …")
        cal_df, val_df = split_train_validation(cdf, folds, validation_fold=fold)
        reliability = compute_reliability(cal_df, tools, families)
        predictions = run_fusion(
            cal_df, val_df, reliability, tools, families,
            calibration_metrics=["sensitivity", "specificity"], tier=tier, seed=seed,
        )
        base, tau = zip(*predictions["strategy"].map(split_tau_strategy), strict=True)
        predictions = predictions.assign(base_strategy=list(base), tau=list(tau))
        # Baselines carry no tau suffix; for the fusers any single materialised variant
        # recovers the same continuous score.
        keep = predictions["tau"].isna() | (predictions["tau"] == REPRESENTATIVE_TAU)
        predictions = predictions[keep]

        fusers = set(fusion_strategies(predictions["strategy"].astype(str).unique()))
        fuser_rows = predictions[predictions["strategy"].isin(fusers)]
        per_fold_scores.append(fuser_rows[["base_strategy", "family", "label", "score"]].copy())

        # The baselines' single operating point comes from the pipeline's own scorer,
        # grouped per (strategy, family) as everywhere else.
        metrics = evaluate_predictions(
            predictions.assign(strategy=predictions["base_strategy"]),
            group_cols=("strategy", "family"),
        )
        metrics.insert(0, "fold", fold)
        per_fold_metrics.append(metrics)

    _per_family, overall = aggregate_fusion_metrics(pd.concat(per_fold_metrics, ignore_index=True))
    baselines = overall[overall["strategy"].astype(str).str.startswith(BASELINE_PREFIX)]
    return {
        "tools": tools,
        "overall": overall.set_index("strategy"),
        "per_fold_scores": per_fold_scores,
        "baseline": str(baselines["strategy"].iloc[0]) if not baselines.empty else None,
    }


def make_pr_curve_figure(
    languages: list[str],
    results_root: Path,
    *,
    tier: str = "full",
    n_splits: int = 5,
    seed: int = 42,
    tau_step: float = 0.02,
    use_sans: bool = True,
) -> list[Path]:
    """Re-run the folds, persist the curve points, then draw the figure from them."""
    taus = _fine_taus(tau_step)
    collected = {lang: _collect(lang, tier, n_splits, seed) for lang in languages}

    # Persist the points before drawing: a figure cannot be checked, and every claim made
    # about these curves (which operating points exist, whether any of them dominates the
    # baseline) has to be answerable without re-running the fusion to recover the scores.
    points = []
    for lang, data in collected.items():
        frame = _curve_points(data["per_fold_scores"], taus)
        frame.insert(0, "language", lang)
        baseline = data["baseline"]
        if baseline is not None and baseline in data["overall"].index:
            row = data["overall"].loc[baseline]
            frame = pd.concat([frame, pd.DataFrame([{
                "language": lang, "base_strategy": baseline, "tau": float("nan"),
                "recall": row["recall_weighted"], "precision": row["precision_weighted"],
            }])], ignore_index=True)
        points.append(frame)
    points_table = pd.concat(points, ignore_index=True)
    results_root.mkdir(parents=True, exist_ok=True)
    points_path = results_root / "pr_curve_weighted_points.csv"
    points_table.to_csv(points_path, index=False)
    print(f"wrote {points_path}  ({len(points_table)} punti)")

    tools = {lang: data["tools"] for lang, data in collected.items()}
    pr_auc = {
        lang: _pr_auc_by_base(data["overall"]["pr_auc_weighted"]) for lang, data in collected.items()
    }
    return [points_path, *_draw_figure(points_table, languages, tools, pr_auc, results_root, use_sans)]


def redraw_from_points(languages: list[str], results_root: Path, *, tier: str = "full",
                       use_sans: bool = True) -> list[Path]:
    """Redraw the figure from the persisted points, without re-running the fusion.

    The curves come from ``pr_curve_weighted_points.csv``; the tool list and the PR-AUC
    come from each language's pipeline outputs (``config.json`` and
    ``fusion_metrics_overall.csv``), i.e. the same numbers the paper's tables report.
    """
    points_table = pd.read_csv(results_root / "pr_curve_weighted_points.csv")
    tools, pr_auc = {}, {}
    for lang in languages:
        run_dir = results_root / lang / "pillar_child" / tier
        tools[lang] = json.loads((run_dir / "config.json").read_text())["tools"]
        overall = pd.read_csv(run_dir / "fusion_metrics_overall.csv").set_index("strategy")
        pr_auc[lang] = _pr_auc_by_base(overall["pr_auc_weighted"])
    return _draw_figure(points_table, languages, tools, pr_auc, results_root, use_sans)


def _pr_auc_by_base(pr_auc_weighted: pd.Series) -> dict[str, float]:
    """Support-weighted PR-AUC per base strategy.

    PR-AUC is the average precision of the continuous score, which does not depend on tau,
    so every tau variant of a base strategy carries the same value and any one of them
    stands for the base.
    """
    out: dict[str, float] = {}
    for strategy, value in pr_auc_weighted.items():
        out.setdefault(split_tau_strategy(str(strategy))[0], float(value))
    return out


def _pr_auc_text(base: str, languages: list[str], pr_auc: dict[str, dict[str, float]]) -> str:
    values = [
        f"{LANG_TITLES.get(lang, lang)} {pr_auc[lang][base]:.3f}"
        for lang in languages
        if base in pr_auc[lang] and not np.isnan(pr_auc[lang][base])
    ]
    return f"PR-AUC: {' · '.join(values)}" if values else ""


def _draw_figure(
    points_table: pd.DataFrame,
    languages: list[str],
    tools: dict[str, list[str]],
    pr_auc: dict[str, dict[str, float]],
    results_root: Path,
    use_sans: bool,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    is_baseline_row = points_table["tau"].isna()
    curves = {
        lang: points_table[(points_table["language"] == lang) & ~is_baseline_row]
        for lang in languages
    }
    baselines = {
        lang: points_table[(points_table["language"] == lang) & is_baseline_row]
        for lang in languages
    }

    ordered_bases: list[str] = []
    for lang, curve in curves.items():
        present = list(curve["base_strategy"].unique())
        for strategy in canonical_strategy_order(tools[lang], present):
            if strategy not in ordered_bases:
                ordered_bases.append(strategy)
    colors = {b: FALLBACK_COLORS[i % len(FALLBACK_COLORS)] for i, b in enumerate(ordered_bases)}
    dashes = {b: ["-", "--", "-.", ":"][i // len(FALLBACK_COLORS) % 4] for i, b in enumerate(ordered_bases)}

    fig_height = 6.0
    rc = rc_style(use_sans, {"font.size": 10})
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(languages), figsize=(4.4 * len(languages), fig_height),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes)

        for ax, lang in zip(axes, languages, strict=True):
            curve = curves[lang]
            for base in ordered_bases:
                points = curve[curve["base_strategy"] == base]
                if points.empty:
                    continue
                # Thin line for the shape, markers on the taus the pipeline actually
                # reports. The connecting segments are NOT measured: on C/C++ the scores
                # are discrete enough that weighted recall falls from 0.997 to 0.647 to
                # 0.262 across three adjacent taus, so a plain line would draw two long
                # straight jumps and read as a smooth curve that was never observed.
                # The markers show where the real operating points are.
                ax.plot(points["recall"], points["precision"], color=colors[base],
                        linestyle=dashes[base], linewidth=1.0, alpha=0.55, zorder=3)
                reported = points[np.isclose(
                    points["tau"].to_numpy()[:, None], REPORTED_TAUS[None, :], atol=1e-6
                ).any(axis=1)]
                ax.plot(reported["recall"], reported["precision"], color=colors[base],
                        linestyle="none", marker="o", markersize=3.6, alpha=0.95, zorder=4)

            for row in baselines[lang].itertuples():
                ax.plot(row.recall, row.precision, marker="X",
                        markersize=11, color="#000000", linestyle="none", zorder=6,
                        markeredgecolor="#FFFFFF", markeredgewidth=0.9)

            ax.set_title(LANG_TITLES.get(lang, lang), fontsize=12, fontweight="bold")
            ax.set_xlabel("Recall (support-weighted)")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, linestyle=":", linewidth=0.5, color="#cccccc")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[0].set_ylabel("Precision (support-weighted)")

        handles = []
        for base in ordered_bases:
            auc_text = _pr_auc_text(base, languages, pr_auc)
            label = f"{strategy_label(base)} — {auc_text}" if auc_text else strategy_label(base)
            handles.append(Line2D([], [], color=colors[base], linestyle=dashes[base],
                                  linewidth=1.5, label=label))
        handles.append(Line2D([], [], color="#000000", marker="X", linestyle="none",
                              markersize=9, label="2ooN baseline (single operating point)"))
        # Reserve the bottom strip for the legend instead of pushing it below the
        # figure with negative offsets, which bbox_inches="tight" turns into a huge
        # white band and a squashed set of axes.
        # Each legend entry carries the per-language PR-AUC on one line, so two wide
        # columns; the strip is sized per row and hangs from just under the x-axis labels.
        n_legend_cols = 2
        n_legend_rows = -(-len(handles) // n_legend_cols)
        # Spacing in inches, converted to figure fractions, so the legend keeps its size
        # whatever the figure height.
        xlabel_h, row_h = 0.26 / fig_height, 0.22 / fig_height
        bottom = xlabel_h + row_h * n_legend_rows
        fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, bottom - xlabel_h),
                   ncol=n_legend_cols, columnspacing=3.0, frameon=False, fontsize=8.5)
        written: list[Path] = []
        for suffix, kwargs in ((".svg", {}), (".png", {"dpi": 200})):
            path = results_root / f"pr_curve_weighted{suffix}"
            fig.savefig(path, bbox_inches="tight", **kwargs)
            written.append(path)
            print(f"wrote {path}")
        plt.close(fig)
        return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="all", choices=["c_cpp", "java", "python", "all"])
    ap.add_argument("--tier", default="full", choices=["base", "medium", "full"])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau-step", type=float, default=0.02,
                    help="threshold grid step for the curve (default 0.02; the pipeline's "
                         "own reported grid stays at 0.1 and is unaffected)")
    ap.add_argument("--serif", action="store_true", help="use a serif font (paper style)")
    ap.add_argument("--results-dir", type=str, default="data/results")
    ap.add_argument("--from-points", action="store_true",
                    help="redraw from the persisted pr_curve_weighted_points.csv instead of "
                         "re-running calibration + fusion (needs only data/results)")
    args = ap.parse_args()

    langs = ["c_cpp", "java", "python"] if args.lang == "all" else [args.lang]
    if args.from_points:
        redraw_from_points(langs, Path(args.results_dir), tier=args.tier, use_sans=not args.serif)
        return
    make_pr_curve_figure(langs, Path(args.results_dir), tier=args.tier, n_splits=args.n_splits,
                         seed=args.seed, tau_step=args.tau_step, use_sans=not args.serif)


if __name__ == "__main__":
    main()
