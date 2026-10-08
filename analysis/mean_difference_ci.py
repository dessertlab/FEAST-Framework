"""Wilcoxon signed-rank test and Hodges-Lehmann confidence intervals for fusion-strategy differences.

This module uses the non-parametric **Wilcoxon signed-rank test** in place of the
parametric paired t-test used by an earlier revision of the pipeline.

Rationale for switching:
  Pairing is per CWE family (N = number of families, e.g. 15-20): for each family, the
  mean F1 across the 5 folds is compared between strategy and baseline (paper §5.3). The
  paired t-test relies on a normality assumption that cannot be verified and is frequently
  violated for F1/MCC differences. The Wilcoxon signed-rank test is distribution-free and
  more appropriate here; it also preserves statistical validity when the underlying
  distributions are skewed or heavy-tailed.

Statistical procedure:
  For each (strategy, baseline) pair, a vector of N per-family paired differences d_i is
  computed (N = number of surviving CWE families). Zero differences are dropped first
  (matching scipy's default ``zero_method="wilcox"``): the p-value, the Hodges-Lehmann
  estimate and the CI are all computed on the same n_nonzero <= N differences, so
  ``n_nonzero_pairs`` (reported next to ``n_pairs``) is the test's real basis. A family
  where neither the strategy nor the baseline detects anything differs by exactly 0: it
  says nothing about which of the two is better, but it does shrink that basis.
  - **Test statistic**: scipy ``wilcoxon(d, alternative='two-sided', method='auto')``.
    With n_nonzero<=25 and no ties the exact distribution is used; otherwise the normal
    approximation applies.
  - **Point estimate**: Hodges-Lehmann estimator — median of all n_nonzero(n_nonzero+1)/2
    Walsh averages (d_i + d_j)/2, i<=j. This is the natural point estimate dual to the
    Wilcoxon test.
  - **95% CI**: the classic Walsh-average interval (Hollander & Wolfe, *Nonparametric
    Statistical Methods*): sort the Walsh averages and take the pair of order statistics
    whose rank is set by the alpha/2 quantile of the null distribution of the signed-rank
    statistic W+. That null distribution is computed exactly, via the standard subset-sum
    recursion, when n_nonzero<=25 with no ties among |d_i|; otherwise a normal
    approximation with the usual tie correction is used -- mirroring the same exact/
    asymptotic split scipy's own ``method="auto"`` makes for the p-value, so by
    construction the CI excludes 0 iff the two-sided p-value is below alpha. (Note: scipy
    itself has never exposed a ``confidence_level``/``.confidence_interval`` for
    ``wilcoxon`` in any released version -- an earlier revision of this function assumed
    it existed under a ``try/except TypeError``, which meant the CI silently always fell
    through to a normal approximation with an incorrect standard error, regardless of the
    installed scipy version.)
  - **Status**: identical classification logic as the t-test version: if the entire CI
    lies above 0 the strategy is classified as a statistically significant improvement;
    if entirely below 0, a degradation; otherwise inconclusive.

Note: an earlier revision of this module also supported a fold-level pairing mode (N=5,
support-weighted across families per fold). It was never used by the pipeline (both
``run_language_level`` and ``regenerate_plots`` always paired by family) and has been
removed: with N=5 and no ties, the exact two-sided Wilcoxon test cannot reach p<0.05
(minimum achievable p is 2/2^5=0.0625), which alone rules it out as the source of the
paper's reported significant results.
"""

from __future__ import annotations

from math import sqrt
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from analysis.aggregation import REPORT_METRICS, SUPPORT_COLUMN, ZERO_FILL_METRICS, zero_fill
from analysis.fusion.common import FUSER_ORDER, canonical_strategy_order, split_tau_strategy

LOWER_IS_BETTER = {"fpr", "fnr"}
DEFAULT_BASELINES = ("traditional",)
CI_LEVEL = 0.95

STATUS_IMPROVEMENT = "incremento_statistico"
STATUS_INCONCLUSIVE = "inconcludente"
STATUS_DEGRADATION = "degradazione_statistica"

STATUS_LABELS = {
    STATUS_IMPROVEMENT: "Statistical Improvement",
    STATUS_INCONCLUSIVE: "Inconclusive",
    STATUS_DEGRADATION: "Statistical Degradation",
}

# Okabe-Ito palette: high-contrast and commonly used as colorblind-friendly.
STATUS_STYLES = {
    STATUS_IMPROVEMENT: {"color": "#0072B2", "marker": "^"},
    STATUS_INCONCLUSIVE: {"color": "#999999", "marker": "o"},
    STATUS_DEGRADATION: {"color": "#D55E00", "marker": "v"},
}


def _exact_wplus_cdf(n: int) -> list[float]:
    """P(W+ <= s) for s=0..n(n+1)/2 under H0, no ties, via the subset-sum recursion.

    W+ is the sum of ranks 1..n assigned to the positive differences; under H0 every one
    of the 2**n sign patterns is equally likely, so dp[s] (subsets of {1..n} summing to s)
    gives the exact null distribution -- the same recursion scipy uses internally for the
    exact Wilcoxon p-value.
    """
    max_sum = n * (n + 1) // 2
    dp = [0] * (max_sum + 1)
    dp[0] = 1
    for rank in range(1, n + 1):
        for s in range(max_sum, rank - 1, -1):
            dp[s] += dp[s - rank]
    total = 2**n
    cdf = []
    cumulative = 0
    for count in dp:
        cumulative += count
        cdf.append(cumulative / total)
    return cdf


def _walsh_rank(cdf: Sequence[float], m: int, alpha: float) -> int:
    """Largest 1-indexed rank c such that (w_(c), w_(m+1-c)) has level >= 1-alpha.

    ``cdf`` is P(W+ <= k) for k=0..m (m = n(n+1)/2, matching the Walsh-average count).
    Finds the largest k with cdf[k] <= alpha/2; c = k+1. When no such k exists (the
    discrete null distribution can't reach alpha/2 at this n -- the same limitation noted
    above for N=5 exact tests), falls back to c=1, i.e. the full observed range of Walsh
    averages as the most conservative interval available.
    """
    half_alpha = alpha / 2.0
    k_star = -1
    for k, value in enumerate(cdf):
        if value <= half_alpha:
            k_star = k
        else:
            break
    c = k_star + 1 if k_star >= 0 else 1
    # Keep the two order-statistic indices from crossing (only possible at tiny n where
    # the discrete distribution is coarse relative to m).
    return max(1, min(c, (m + 1) // 2))


def _asymptotic_walsh_rank(n: int, abs_d: "object", alpha: float, m: int) -> int:
    """Normal-approximation counterpart of ``_walsh_rank`` for n>25 or tied |d_i|.

    Mirrors the tie-corrected normal approximation scipy's ``wilcoxon`` uses for the
    p-value when ``method="auto"`` falls back to the asymptotic branch.
    """
    import numpy as np
    from scipy.stats import norm

    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    _values, counts = np.unique(abs_d, return_counts=True)
    tie_sizes = counts[counts > 1]
    if len(tie_sizes):
        var -= float(((tie_sizes**3 - tie_sizes) / 48.0).sum())
    sigma = sqrt(max(var, 0.0))
    if sigma == 0:
        return 1
    z = norm.ppf(alpha / 2.0)
    k_star = int((mu + z * sigma - 0.5) // 1)
    c = k_star + 1 if k_star >= 0 else 1
    return max(1, min(c, (m + 1) // 2))


def _wilcoxon_ci(
    diffs: "pd.Series",
    confidence_level: float = 0.95,
) -> tuple[float, float, float, float]:
    """Hodges-Lehmann point estimate and Walsh-average CI via the Wilcoxon signed-rank test.

    Returns (p_value, hl_estimate, ci_low, ci_high). Zero differences are dropped first
    (matching scipy's default ``zero_method="wilcox"``) so the p-value, the HL estimate
    and the CI share the same n_nonzero basis. For n_nonzero <= 1 all values are NaN --
    see the module docstring for the exact/asymptotic CI construction.
    """
    import numpy as np
    from scipy.stats import wilcoxon

    d_all = diffs.dropna().to_numpy(dtype=float)
    d = d_all[d_all != 0]
    n = len(d)
    if n <= 1:
        return float("nan"), float("nan"), float("nan"), float("nan")

    result = wilcoxon(d, alternative="two-sided", method="auto")
    p_value = float(result.pvalue)

    idx = np.triu_indices(n)
    walsh = np.sort((d[idx[0]] + d[idx[1]]) / 2.0)
    m = len(walsh)
    hl = float(np.median(walsh))

    alpha = 1.0 - confidence_level
    abs_d = np.abs(d)
    has_ties = len(np.unique(abs_d)) < n
    if n <= 25 and not has_ties:
        c = _walsh_rank(_exact_wplus_cdf(n), m, alpha)
    else:
        c = _asymptotic_walsh_rank(n, abs_d, alpha, m)

    ci_low = float(walsh[c - 1])
    ci_high = float(walsh[m - c])
    return p_value, hl, ci_low, ci_high


def _resolve_baseline(present: set[str], baseline: str) -> str:
    baseline = baseline.strip()
    if baseline in present:
        return baseline
    if baseline == "traditional":
        matches = sorted(s for s in present if s.startswith("traditional_"))
        if len(matches) > 1:
            matches_2 = [m for m in matches if m.startswith("traditional_2_of_")]
            if matches_2:
                return matches_2[0]
    elif baseline == "or":
        matches = sorted(s for s in present if s.startswith("or_1_of_"))
    else:
        matches = sorted(s for s in present if s.startswith(baseline))
    if not matches:
        raise ValueError(f"baseline strategy not found: {baseline}")
    if len(matches) > 1:
        raise ValueError(f"baseline {baseline!r} is ambiguous: {matches}")
    return matches[0]


def fusion_strategies(present: Iterable[str]) -> list[str]:
    fusers = set(FUSER_ORDER)
    strategies: list[str] = []
    for strategy in present:
        base, _tau = split_tau_strategy(str(strategy))
        if base in fusers:
            strategies.append(str(strategy))
    return strategies


def best_tau_per_strategy(
    per_family: pd.DataFrame,
    strategies: list[str],
    metric: str,
) -> list[str]:
    """For each base strategy keep only the tau variant with the best mean metric.

    Strategies without a tau suffix are passed through unchanged. Missing values follow
    the shared ``analysis.aggregation.zero_fill`` convention, so a variant that abandons
    a family scores 0.0 there rather than dropping out of its own mean. This matters most
    here: ``pandas.Series.mean()`` is skipna, so without the fill, raising tau until a
    variant goes silent on most families (their F1 becomes NaN, not low) would shrink the
    average onto an ever-easier subset -- rewarding coverage loss instead of penalising
    it. With the fill, tau selection optimises the same quantity ``_paired_difference``
    later scores. Note this is the *unweighted* mean over families: tau selection and the
    paired test are macro, while the headline table is support-weighted (see
    ``analysis.aggregation``).
    When all variants of a base strategy have empty data for the metric, the first
    variant is kept as a fallback so the base strategy still appears in the plot.
    """
    ascending = metric in LOWER_IS_BETTER
    groups: dict[str, list[str]] = {}
    for s in strategies:
        base, _tau = split_tau_strategy(s)
        groups.setdefault(base, []).append(s)

    selected: list[str] = []
    for _base, variants in groups.items():
        if len(variants) == 1:
            selected.append(variants[0])
            continue
        scores: dict[str, float] = {}
        for v in variants:
            vals = zero_fill(per_family.loc[per_family["strategy"] == v, metric], metric)
            scores[v] = float(vals.mean()) if not vals.empty else float("nan")
        valid = {v: s for v, s in scores.items() if not pd.isna(s)}
        if not valid:
            selected.append(variants[0])
        else:
            best = min(valid, key=lambda v: valid[v] if ascending else -valid[v])
            selected.append(best)
    return selected


def _paired_difference(
    per_family: pd.DataFrame,
    strategy: str,
    baseline: str,
    metric: str,
) -> tuple["pd.Series", "pd.Series", int, int]:
    """Return (diffs, weights, n_coverage_gain, n_coverage_loss).

    ``weights`` is the family support aligned to ``diffs``, so the caller can report the
    support-weighted difference (the headline estimand) alongside the unweighted vector
    the Wilcoxon test consumes. It is empty when ``per_family`` carries no support column.

    Missing values follow the shared ``analysis.aggregation.zero_fill`` convention,
    applied symmetrically to both sides so that coverage-loss pairs (strategy silent,
    baseline detects) enter as negative differences instead of being silently dropped.
    Pairs where both sides are NaN carry no information and are excluded.

    The coverage counters only make sense for ``ZERO_FILL_METRICS``; for any other metric
    NaN is a property of the family, not of the strategy, so both are reported as 0.
    ``n_coverage_gain`` counts families the baseline does not cover (metric 0 after
    zero-filling, whether it stayed silent or fired without ever being right) but the
    strategy does (metric>0); ``n_coverage_loss`` is the mirror case.
    """
    support_cols = [SUPPORT_COLUMN] if SUPPORT_COLUMN in per_family.columns else []
    strategy_values = (
        per_family.loc[per_family["strategy"] == strategy, ["family", metric, *support_cols]]
        .rename(columns={metric: "strategy_value"})
    )
    baseline_values = (
        per_family.loc[per_family["strategy"] == baseline, ["family", metric]]
        .rename(columns={metric: "baseline_value"})
    )
    paired = strategy_values.merge(baseline_values, on="family", how="inner")
    strategy_metric_raw = pd.to_numeric(paired["strategy_value"], errors="coerce")
    baseline_metric_raw = pd.to_numeric(paired["baseline_value"], errors="coerce")
    baseline_nan = baseline_metric_raw.isna()
    strategy_nan = strategy_metric_raw.isna()
    strategy_metric = zero_fill(strategy_metric_raw, metric)
    baseline_metric = zero_fill(baseline_metric_raw, metric)
    if metric in ZERO_FILL_METRICS:
        # "Covered" means the family yields at least one true positive, so read the
        # zero-filled values, not the raw ones. Silence (NaN) and firing without ever
        # being right (a genuine 0.0) are the same outcome for coverage, and since
        # ``mean_over_folds`` now zero-fills before averaging, a silent family reaches
        # here as 0.0 and never as NaN -- testing isna() here counted nothing at all.
        n_coverage_gain = int(((baseline_metric == 0) & (strategy_metric > 0)).sum())
        n_coverage_loss = int(((strategy_metric == 0) & (baseline_metric > 0)).sum())
    else:
        n_coverage_gain = n_coverage_loss = 0
    # Pairs carrying no information: both sides NaN, or (for a non-zero-filled metric)
    # a difference that stays NaN because one side is undefined for that family.
    keep = ~(baseline_nan & strategy_nan)
    diffs = (strategy_metric - baseline_metric)[keep]
    keep = keep & diffs.notna().reindex(keep.index, fill_value=False)
    weights = (
        pd.to_numeric(paired.loc[keep, SUPPORT_COLUMN], errors="coerce")
        if support_cols else pd.Series(dtype=float)
    )
    return (
        diffs.dropna().reset_index(drop=True),
        weights.reset_index(drop=True),
        n_coverage_gain,
        n_coverage_loss,
    )


def _support_weighted_difference(
    diffs: "pd.Series",
    weights: "pd.Series",
) -> tuple[float, float]:
    """Support-weighted paired difference and Kish's effective sample size.

    The weighted value is the headline estimand -- "how much does the ensemble gain on the
    bulk of real positives" -- and is reported next to the unweighted vector the Wilcoxon
    test actually consumes. The two answer different questions and are not interchangeable:
    the weighted one is dominated by the largest families (on C/C++ a single CWE carries
    ~44% of the positives), the unweighted one gives every family the same voice.

    ``n_effective`` = (sum w)^2 / sum(w^2) makes that concentration explicit: it is the
    number of *equally weighted* families the weighted mean is worth. Report it whenever
    the weighted difference is quoted, so a large weighted gain resting on one family
    cannot be mistaken for a gain spread over all of them.
    """
    if diffs.empty or weights.empty or len(weights) != len(diffs):
        return float("nan"), float("nan")
    w = pd.to_numeric(weights, errors="coerce")
    mask = w.notna() & (w > 0) & diffs.notna()
    if not mask.any():
        return float("nan"), float("nan")
    w = w[mask].astype(float)
    total = float(w.sum())
    weighted = float((diffs[mask].astype(float) * w).sum() / total)
    n_effective = float(total**2 / float((w**2).sum()))
    return weighted, n_effective


def _classify_difference(metric: str, ci_low: float, ci_high: float) -> str:
    if pd.isna(ci_low) or pd.isna(ci_high):
        return STATUS_INCONCLUSIVE
    lower_is_better = metric in LOWER_IS_BETTER
    if ci_low > 0:
        return STATUS_DEGRADATION if lower_is_better else STATUS_IMPROVEMENT
    if ci_high < 0:
        return STATUS_IMPROVEMENT if lower_is_better else STATUS_DEGRADATION
    return STATUS_INCONCLUSIVE


def mean_difference_ci(
    per_family: pd.DataFrame,
    metric: str = "f1",
    baselines: Sequence[str] = DEFAULT_BASELINES,
    tools: Sequence[str] = (),
    best_tau_only: bool = True,
    preselected_strategies: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute paired 95% CIs for each fusion strategy against each baseline.

    Pairing is per CWE family (N = number of surviving families): for each family, the
    fold-averaged metric in ``per_family`` is compared between strategy and baseline
    (paper §5.3). ``difference`` is always ``strategy metric - baseline metric``. The
    ``status`` column accounts for metric direction, so lower-is-better metrics treat
    negative intervals as improvements.

    ``best_tau_only``: when True (default) each base strategy contributes only its
    best-mean-metric tau variant.  Tau selection uses ``per_family`` (fold-averaged
    per-family values) UNLESS ``preselected_strategies`` is given.

    ``preselected_strategies``: tau-suffixed strategy names chosen ahead of time from
    calibration-internal data (see ``analysis.experiment._select_tau_nested``), so the
    selection never touches the held-out fold values being reported here. When given,
    this replaces the ``best_tau_per_strategy`` call entirely -- ``best_tau_only`` is
    ignored. Passing the outcome of a proper nested split closes the "which held-out
    data selected tau" question raised in review: tau is fixed before ``per_family`` is
    ever consulted for anything other than reporting.
    """
    required = {"strategy", "family", metric}
    missing = required - set(per_family.columns)
    if missing:
        raise ValueError(f"per_family is missing required columns: {sorted(missing)}")
    if metric not in REPORT_METRICS:
        raise ValueError(f"unsupported metric {metric!r}; expected one of {REPORT_METRICS}")

    present = set(per_family["strategy"].astype(str))
    if baselines == DEFAULT_BASELINES:
        trad_strategies = sorted(s for s in present if s.startswith("traditional_"))
        if len(trad_strategies) == 1:
            resolved = [("traditional", trad_strategies[0])]
        elif len(trad_strategies) > 1:
            # When multiple traditional_* variants exist, always prefer the
            # 2-of-N variant as the sole comparison baseline and discard the
            # rest (e.g. traditional_3_of_6 is never used).
            matches_2 = [s for s in trad_strategies if s.startswith("traditional_2_of_")]
            if matches_2:
                resolved = [("traditional_2", matches_2[0])]
            else:
                # No 2-of-N available: fall back to all variants as before.
                resolved = []
                for s in trad_strategies:
                    parts = s.split("_")
                    if len(parts) >= 4 and parts[0] == "traditional" and parts[2] == "of":
                        resolved.append((f"traditional_{parts[1]}", s))
                    else:
                        resolved.append((s, s))
        else:
            resolved = [(name, _resolve_baseline(present, name)) for name in baselines]
    else:
        resolved = [(name, _resolve_baseline(present, name)) for name in baselines]
    all_fusion = canonical_strategy_order(tools, fusion_strategies(present))
    if preselected_strategies is not None:
        strategies = list(preselected_strategies)
    elif best_tau_only:
        strategies = best_tau_per_strategy(per_family, all_fusion, metric)
    else:
        strategies = all_fusion
    strategies = canonical_strategy_order(tools, strategies)

    rows: list[dict] = []
    for baseline_name, baseline_strategy in resolved:
        for strategy in strategies:
            diffs, weights, n_coverage_gain, n_coverage_loss = _paired_difference(
                per_family, strategy, baseline_strategy, metric
            )
            n = int(len(diffs))
            # scipy's wilcoxon discards zero differences (zero_method="wilcox", its
            # default), so the test's real basis is the non-zero count, not n_pairs.
            # The gap is not cosmetic: on C/C++ both Naive Bayes and BKS pair 20 families
            # of which 12 differ by exactly 0 -- neither the strategy nor the baseline
            # scores a single true positive there -- leaving 8 usable observations that
            # split 4 positive / 4 negative. That is what puts the Hodges-Lehmann estimate
            # exactly on zero and makes the interval symmetric, and reading it as
            # "20 families disagree mildly" gets the result backwards.
            n_nonzero = int((diffs != 0).sum())
            mean = float(diffs.mean()) if n else float("nan")
            weighted, n_effective = _support_weighted_difference(diffs, weights)
            p_value, hl_estimate, ci_low, ci_high = _wilcoxon_ci(diffs, confidence_level=CI_LEVEL)
            rows.append({
                "baseline": baseline_name,
                "baseline_strategy": baseline_strategy,
                "strategy": strategy,
                "metric": metric,
                "n_pairs": n,
                "n_nonzero_pairs": n_nonzero,
                "mean_difference": mean,
                "weighted_difference": weighted,
                "n_effective": n_effective,
                "hodges_lehmann": hl_estimate,
                "p_value": p_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_level": CI_LEVEL,
                "status": _classify_difference(metric, ci_low, ci_high),
                "n_coverage_gain": n_coverage_gain,
                "n_coverage_loss": n_coverage_loss,
            })
    return pd.DataFrame(rows)


def strategy_label(strategy: str) -> str:
    base, tau = split_tau_strategy(strategy)
    if base.startswith("weighted_fire_") and "_silence_" in base:
        label = "Weighted Voting"
    elif base.startswith("dst_") and "_fire_" in base and "_silence_" in base:
        rule, _rest = base.removeprefix("dst_").split("_fire_", 1)
        rule_label = rule.title() if rule.lower() not in ("pcr6",) else "PCR6"
        label = f"DST {rule_label}"
    else:
        mapping = {
            "logistic_regression": "Logistic Regression",
            "decision_tree": "Decision Tree",
            "random_forest": "Random Forest",
            "gradient_boosting": "Gradient Boosting",
            "xgboost": "XGBoost",
            "naive_bayes": "Naive Bayes",
            "logistic_interactions": "Logistic Interactions",
            "bks": "BKS",
        }
        # Pattern-based labels for N-of-M voting strategies; handles any tool count.
        m_or = re.match(r"or_(\d+)_of_(\d+)$", base)
        m_trad = re.match(r"traditional_(\d+)_of_(\d+)$", base)
        if m_or:
            label = f"OR {m_or.group(1)}-of-{m_or.group(2)}"
        elif m_trad:
            label = f"Traditional {m_trad.group(1)}-of-{m_trad.group(2)}"
        else:
            label = mapping.get(base, base.replace("_", " ").title())
            for acr in ["Dst", "Pcr6", "Bks", "Or"]:
                if acr in label:
                    label = label.replace(acr, acr.upper())
    if tau is not None:
        label = f"{label}, $\\tau$={tau:.1f}"
    return label


def _format_p_value(p_val: float) -> str:
    if p_val >= 0.001:
        return f"p={p_val:.3f}"
    s = f"{p_val:.1e}"
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    return f"p={s}"


def annotate_pvalue(
    ax: "matplotlib.axes.Axes",
    hl: float,
    y: float,
    row: "pd.Series",
) -> None:
    """Draw the p-value (+ optional Δc coverage indicator) below a CI point.

    For degenerate tests (p=NaN, e.g. n<=2), draws ``p=*`` when the coverage
    delta is non-zero, so that meaningful coverage information is not lost.
    """
    gain = int(row.get("n_coverage_gain", 0) or 0)
    loss = int(row.get("n_coverage_loss", 0) or 0)
    delta_c = gain - loss

    p_val_raw = row.get("p_value") if "p_value" in row else None
    p_is_nan = p_val_raw is None or pd.isna(p_val_raw)

    if p_is_nan:
        if delta_c == 0:
            return  # nothing informative to show
        p_str = "p=*"
        weight = "normal"
    else:
        p_val = float(p_val_raw)
        p_str = _format_p_value(p_val)
        weight = "bold" if p_val < 0.05 else "normal"

    if delta_c > 0:
        p_str += f" (+{delta_c})"
    elif delta_c < 0:
        p_str += f" ({delta_c})"

    ax.text(
        hl, y - 0.15, p_str,
        ha="center", va="bottom",
        fontsize=16, color="#000000",
        fontweight=weight,
        zorder=4,
    )


def save_mean_difference_ci_plot(
    intervals: pd.DataFrame,
    out_stem: Path,
) -> list[Path]:
    """Save a paper-style CI forest plot as SVG and PNG.

    Every point carries its p-value and coverage delta. These are not decoration: the
    paper's own Figure 4 caption describes them as part of the figure, so leaving them
    behind an off-by-default switch meant the documented command did not reproduce the
    published plot. ``annotate_pvalue`` stays silent where there is nothing to say.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    required = {"baseline", "strategy", "metric", "mean_difference", "ci_low", "ci_high", "status"}
    missing = required - set(intervals.columns)
    if missing:
        raise ValueError(f"intervals is missing required columns: {sorted(missing)}")
    if intervals.empty:
        return []

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    unique_baselines = sorted(intervals["baseline"].astype(str).unique())

    if len(unique_baselines) <= 1:
        preferred = unique_baselines[0] if unique_baselines else "traditional"
        frame = intervals[intervals["baseline"].astype(str) == preferred].copy()
        if frame.empty:
            return []

        strategies = frame["strategy"].astype(str).tolist()
        labels = [strategy_label(strategy) for strategy in strategies]

        y_positions = list(range(len(strategies)))
        fig_h = max(4.4, 0.34 * len(strategies) + 1.8)
        fig_w = 8.2
        rc = {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.edgecolor": "#000000",
            "axes.linewidth": 0.8,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "text.color": "#000000",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "font.size": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            for y in y_positions:
                if y % 2:
                    ax.axhspan(y - 0.5, y + 0.5, color="#f7f7f7", zorder=0)
            ax.axvline(0.0, color="#000000", linewidth=0.9, linestyle="--", zorder=1)

            for y, (_idx, row) in zip(y_positions, frame.iterrows(), strict=False):
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
                    markersize=5.2,
                    markeredgewidth=0.8,
                    markeredgecolor="#222222",
                    zorder=3,
                )
                annotate_pvalue(ax, hl, y, row)

            ax.set_yticks(y_positions, labels)
            ax.invert_yaxis()
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
            ax.grid(axis="x", linestyle=":", linewidth=0.6, color="#bdbdbd", alpha=0.8)
            ax.tick_params(axis="both", labelsize=14)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

            legend_handles = [
                mlines.Line2D(
                    [], [], color=style["color"], marker=style["marker"], linestyle="None",
                    markeredgecolor="#222222", markeredgewidth=0.8, markersize=6,
                    label=STATUS_LABELS[status],
                )
                for status, style in STATUS_STYLES.items()
            ]
            ax.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=3,
                frameon=False,
                fontsize=14,
                handletextpad=0.4,
                columnspacing=1.2,
            )
            fig.tight_layout(rect=(0, 0.04, 1, 1))

            svg_path = out_stem.with_suffix(".svg")
            png_path = out_stem.with_suffix(".png")
            fig.savefig(svg_path, bbox_inches="tight")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return [svg_path, png_path]
    else:
        baseline_1, baseline_2 = unique_baselines[:2]
        frame_1 = intervals[intervals["baseline"].astype(str) == baseline_1].copy()
        frame_2 = intervals[intervals["baseline"].astype(str) == baseline_2].copy()

        if frame_1.empty or frame_2.empty:
            return []

        strategies = frame_1["strategy"].astype(str).tolist()
        labels = [strategy_label(strategy) for strategy in strategies]

        # Align frame_2 to the same strategies order as frame_1
        frame_2 = frame_2.set_index("strategy").reindex(strategies).reset_index()

        y_positions = list(range(len(strategies)))
        fig_h = max(4.4, 0.34 * len(strategies) + 1.8)
        fig_w = 12.0
        rc = {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#222222",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "font.size": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
        with plt.rc_context(rc):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=True)

            # Left subplot
            for y in y_positions:
                if y % 2:
                    ax1.axhspan(y - 0.5, y + 0.5, color="#f7f7f7", zorder=0)
            ax1.axvline(0.0, color="#222222", linewidth=0.9, linestyle="--", zorder=1)

            for y, (_idx, row) in zip(y_positions, frame_1.iterrows(), strict=False):
                hl = float(row["hodges_lehmann"])
                ci_low = float(row["ci_low"])
                ci_high = float(row["ci_high"])
                status = str(row["status"])
                style = STATUS_STYLES.get(status, STATUS_STYLES[STATUS_INCONCLUSIVE])
                if pd.isna(hl) or pd.isna(ci_low) or pd.isna(ci_high):
                    continue
                ax1.errorbar(
                    hl, y,
                    xerr=[[hl - ci_low], [ci_high - hl]],
                    fmt=style["marker"],
                    color=style["color"],
                    ecolor=style["color"],
                    elinewidth=1.1,
                    capsize=2.5,
                    capthick=1.0,
                    markersize=5.2,
                    markeredgewidth=0.8,
                    markeredgecolor="#222222",
                    zorder=3,
                )
                annotate_pvalue(ax1, hl, y, row)

            ax1.set_yticks(y_positions, labels)
            ax1.invert_yaxis()
            ax1.set_xlabel("")
            ax1.set_title("")
            ax1.grid(axis="x", linestyle=":", linewidth=0.6, color="#bdbdbd", alpha=0.8)
            ax1.tick_params(axis="both", labelsize=14)
            for spine in ("top", "right"):
                ax1.spines[spine].set_visible(False)

            # Right subplot
            for y in y_positions:
                if y % 2:
                    ax2.axhspan(y - 0.5, y + 0.5, color="#f7f7f7", zorder=0)
            ax2.axvline(0.0, color="#222222", linewidth=0.9, linestyle="--", zorder=1)

            for y, (_idx, row) in zip(y_positions, frame_2.iterrows(), strict=False):
                hl = float(row["hodges_lehmann"])
                ci_low = float(row["ci_low"])
                ci_high = float(row["ci_high"])
                status = str(row["status"])
                style = STATUS_STYLES.get(status, STATUS_STYLES[STATUS_INCONCLUSIVE])
                if pd.isna(hl) or pd.isna(ci_low) or pd.isna(ci_high):
                    continue
                ax2.errorbar(
                    hl, y,
                    xerr=[[hl - ci_low], [ci_high - hl]],
                    fmt=style["marker"],
                    color=style["color"],
                    ecolor=style["color"],
                    elinewidth=1.1,
                    capsize=2.5,
                    capthick=1.0,
                    markersize=5.2,
                    markeredgewidth=0.8,
                    markeredgecolor="#222222",
                    zorder=3,
                )
                annotate_pvalue(ax2, hl, y, row)

            ax2.set_xlabel("")
            ax2.set_title("")
            ax2.grid(axis="x", linestyle=":", linewidth=0.6, color="#bdbdbd", alpha=0.8)
            ax2.tick_params(axis="both", labelsize=14)
            for spine in ("top", "right"):
                ax2.spines[spine].set_visible(False)

            # Legend for overall figure
            legend_handles = [
                mlines.Line2D(
                    [], [], color=style["color"], marker=style["marker"], linestyle="None",
                    markeredgecolor="#222222", markeredgewidth=0.8, markersize=6,
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
                fontsize=14,
                handletextpad=0.4,
                columnspacing=1.2,
            )
            fig.tight_layout(rect=(0, 0.04, 1, 0.95))

            svg_path = out_stem.with_suffix(".svg")
            png_path = out_stem.with_suffix(".png")
            fig.savefig(svg_path, bbox_inches="tight")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return [svg_path, png_path]


def save_mean_difference_ci_report(
    per_family: pd.DataFrame,
    out_dir: Path,
    *,
    metric: str = "f1",
    baselines: Sequence[str] = DEFAULT_BASELINES,
    tools: Sequence[str] = (),
    stem: str | None = None,
    best_tau_only: bool = True,
    preselected_strategies: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Compute intervals, write the CSV, and save the SVG/PNG forest plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    intervals = mean_difference_ci(
        per_family, metric=metric, baselines=baselines, tools=tools,
        best_tau_only=best_tau_only, preselected_strategies=preselected_strategies,
    )
    name = stem or f"mean_difference_ci_{metric}"
    csv_path = out_dir / f"{name}.csv"
    intervals.to_csv(csv_path, index=False)
    plot_paths = save_mean_difference_ci_plot(intervals, out_dir / name)
    return intervals, [csv_path, *plot_paths]
