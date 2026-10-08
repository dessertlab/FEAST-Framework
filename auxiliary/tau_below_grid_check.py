"""Does F1 keep rising below the reported tau grid?

Naive Bayes and BKS on C/C++ both select tau=0.1 -- the *minimum* of the pipeline's
{0.1..0.9} grid -- and all five outer folds choose it independently. An optimum sitting on
the boundary of the search space is not evidence that it is the optimum: it is evidence
that the search space may be too narrow. This script extends the grid downward to
[0.01..0.09] at 0.01 steps and reports what happens to the quantity tau selection actually
maximises (the unweighted mean F1 across CWE families), alongside the support-weighted
mean and the paired Wilcoxon result against the K-of-N baseline.

Why this cannot be done with `main.py fusion --tau-min`: ``taus_in_range`` only clips the
fixed 0.1-step ``DEFAULT_TAUS``, and ``tau_suffix`` formats with %.1f, so 0.01 through 0.09
would all collide on the strategy name ``tau_0_0``. The fine grid is therefore applied by
re-thresholding the continuous scores here, exactly as auxiliary/pr_curve_plots.py does.

Scope: this is a *diagnostic on held-out folds*, not a re-selection. It answers "is the
reported configuration on a plateau or on a cliff", which is what a reviewer asks when the
chosen tau is at the edge. It deliberately does not feed back into tau selection, which
stays nested inside the calibration split.

The pipeline's reported grid stays {0.1..0.9}: this script exists to quantify what that
choice costs, not to replace it. On C/C++ it costs a lot -- Naive Bayes and BKS both
select the grid minimum in all five folds, and their F1 is 44% higher at tau=0.02 -- but
the gain has to be read against a trivial always-fire predictor, which at the reported tau
they are already below. Both figures belong in the limitations, not in a new headline.

Once ``fusion_score_histogram.csv`` exists for a run, this analysis can be recomputed from
it in seconds via ``analysis.fusion.metrics_from_histogram`` instead of re-running the
fusion, which is what the rest of this module still does.

    uv run python auxiliary/tau_below_grid_check.py --lang c_cpp
    uv run python auxiliary/tau_below_grid_check.py --lang c_cpp --strategies naive_bayes,bks
"""

from __future__ import annotations

import argparse
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
from analysis.fusion import TIER_MIN_COUNT, run_fusion
from analysis.fusion.common import split_tau_strategy
from analysis.fusion.predictions import binary_metrics, evaluate_predictions
from analysis.mean_difference_ci import mean_difference_ci

REPRESENTATIVE_TAU = 0.5  # fuser scores do not depend on tau; any materialised variant recovers them
BASELINE_PREFIX = "traditional_"
DEFAULT_STRATEGIES = ("naive_bayes", "bks")


def _grid(step_below: float = 0.01) -> np.ndarray:
    """The extension [step..0.09] plus the pipeline's own {0.1..0.9}, for continuity."""
    below = np.round(np.arange(step_below, 0.1, step_below), 6)
    reported = np.round(np.arange(0.1, 0.91, 0.1), 6)
    return np.concatenate([below, reported])


def _per_family_at_taus(frames: list[pd.DataFrame], taus: np.ndarray) -> pd.DataFrame:
    """Per-(fold, strategy@tau, family) metrics, re-thresholding the stored scores."""
    rows: list[dict] = []
    for fold, frame in enumerate(frames):
        for (strategy, family), group in frame.groupby(["base_strategy", "family"], sort=False):
            labels = group["label"].astype(bool).tolist()
            scores = group["score"].astype(float).to_numpy()
            for tau in taus:
                m = binary_metrics(labels, (scores >= tau).tolist(), scores.tolist())
                rows.append({
                    "fold": fold, "strategy": f"{strategy}@{tau:.4f}", "family": family,
                    "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
                    SUPPORT_COLUMN: m["positive_support"],
                })
    return pd.DataFrame(rows)


def run(language: str, tier: str, strategies: list[str], step: float, seed: int) -> pd.DataFrame:
    prep = prepare_canonical(language, "pillar_child", n_splits=5,
                             min_cwe_count=TIER_MIN_COUNT.get(tier, 5), seed=seed)
    cdf, tools, families, folds = prep.cdf, prep.tools, prep.families, prep.folds
    taus = _grid(step)

    score_frames, baseline_frames = [], []
    for fold in sorted(folds["fold"].unique()):
        print(f"  [{language}] fold {fold}: calibrating + fusing …")
        cal_df, val_df = split_train_validation(cdf, folds, validation_fold=fold)
        reliability = compute_reliability(cal_df, tools, families)
        predictions = run_fusion(cal_df, val_df, reliability, tools, families,
                                 calibration_metrics=["sensitivity", "specificity"],
                                 tier=tier, seed=seed)
        base, tau = zip(*predictions["strategy"].map(split_tau_strategy), strict=True)
        predictions = predictions.assign(base_strategy=list(base), tau=list(tau))

        wanted = predictions[predictions["base_strategy"].isin(strategies)
                             & (predictions["tau"] == REPRESENTATIVE_TAU)]
        score_frames.append(wanted[["base_strategy", "family", "label", "score"]].copy())

        # The baseline is a fixed rule: score it as it stands, once per fold.
        baseline_rows = predictions[predictions["base_strategy"].str.startswith(BASELINE_PREFIX)]
        metrics = evaluate_predictions(
            baseline_rows.assign(strategy=baseline_rows["base_strategy"]),
            group_cols=("strategy", "family"),
        )
        metrics.insert(0, "fold", fold)
        baseline_frames.append(metrics)

    per_fold = pd.concat([_per_family_at_taus(score_frames, taus),
                          pd.concat(baseline_frames, ignore_index=True)], ignore_index=True)
    per_family, overall = aggregate_fusion_metrics(per_fold, metric_columns=("f1", "precision", "recall"))

    baseline = next(s for s in per_family["strategy"].unique() if s.startswith(BASELINE_PREFIX))
    swept = [s for s in per_family["strategy"].unique() if "@" in s]
    ci = mean_difference_ci(per_family, metric="f1", preselected_strategies=swept)

    out = overall[overall["strategy"].isin(swept)].merge(
        ci[["strategy", "mean_difference", "weighted_difference", "n_pairs",
            "n_nonzero_pairs", "p_value", "status"]],
        on="strategy", how="left",
    )
    out["base_strategy"] = out["strategy"].str.split("@").str[0]
    out["tau"] = out["strategy"].str.split("@").str[1].astype(float)
    out.insert(0, "language", language)
    out.insert(1, "baseline", baseline)
    cols = ["language", "baseline", "base_strategy", "tau", "f1_macro", "f1_weighted",
            "precision_weighted", "recall_weighted", "mean_difference", "weighted_difference",
            "n_pairs", "n_nonzero_pairs", "p_value", "status"]
    return out[cols].sort_values(["base_strategy", "tau"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="c_cpp")
    ap.add_argument("--tier", default="full", choices=["base", "medium", "full"])
    ap.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    ap.add_argument("--step", type=float, default=0.01, help="step below 0.1 (default 0.01)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-dir", type=str, default="data/results")
    args = ap.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    frames = [run(lang, args.tier, strategies, args.step, args.seed)
              for lang in (l.strip() for l in args.lang.split(",") if l.strip())]
    table = pd.concat(frames, ignore_index=True)

    out_dir = ROOT / args.results_dir / "_cross_language"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tau_below_grid.csv"
    table.to_csv(path, index=False)

    show = ["base_strategy", "tau", "f1_macro", "f1_weighted", "mean_difference",
            "weighted_difference", "n_nonzero_pairs", "p_value", "status"]
    print(f"\n=== F1 below and across the reported tau grid ({args.lang}) ===")
    print(table[show].to_string(index=False))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
