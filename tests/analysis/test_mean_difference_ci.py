import pandas as pd
from pytest import approx

from analysis.mean_difference_ci import (
    STATUS_DEGRADATION,
    STATUS_IMPROVEMENT,
    STATUS_INCONCLUSIVE,
    _paired_difference,
    mean_difference_ci,
    save_mean_difference_ci_report,
)


def _rows(strategy, values):
    return [
        {"strategy": strategy, "family": family, "f1": value}
        for family, value in zip(["CWE-1", "CWE-2", "CWE-3"], values, strict=False)
    ]


def _per_family():
    return pd.DataFrame([
        *_rows("or_1_of_4", [0.50, 0.50, 0.50]),
        *_rows("traditional_2_of_4", [0.50, 0.50, 0.50]),
        *_rows("weighted_fire_ppv_silence_npv_tau_0_5", [0.70, 0.80, 0.90]),
        *_rows("naive_bayes_tau_0_5", [0.60, 0.40, 0.50]),
        *_rows("bks_tau_0_5", [0.30, 0.20, 0.10]),
    ])


def test_mean_difference_ci_classifies_paired_intervals_against_traditional_baseline():
    intervals = mean_difference_ci(_per_family(), metric="f1", tools=["a", "b", "c", "d"])

    by_pair = {
        (row["baseline"], row["strategy"]): row["status"]
        for row in intervals.to_dict("records")
    }

    assert set(intervals["baseline"]) == {"traditional"}
    assert by_pair[("traditional", "weighted_fire_ppv_silence_npv_tau_0_5")] == STATUS_IMPROVEMENT
    assert by_pair[("traditional", "naive_bayes_tau_0_5")] == STATUS_INCONCLUSIVE
    assert by_pair[("traditional", "bks_tau_0_5")] == STATUS_DEGRADATION


def test_mean_difference_ci_report_writes_csv_svg_and_png(tmp_path):
    intervals, paths = save_mean_difference_ci_report(_per_family(), tmp_path, metric="f1")
    names = {path.name for path in paths}

    assert not intervals.empty
    assert names == {"mean_difference_ci_f1.csv", "mean_difference_ci_f1.svg", "mean_difference_ci_f1.png"}
    assert (tmp_path / "mean_difference_ci_f1.csv").is_file()
    assert (tmp_path / "mean_difference_ci_f1.svg").is_file()
    assert (tmp_path / "mean_difference_ci_f1.png").is_file()
    svg = (tmp_path / "mean_difference_ci_f1.svg").read_text(encoding="utf-8")
    assert "Weighted Voting" in svg
    assert "or_1_of_4" not in svg


def _per_family_with_gaps():
    # The strategy abandons CWE-3 (f1 NaN); the baseline abandons CWE-2.
    # roc_auc is NaN on CWE-3 for both: that family is degenerate, nobody failed.
    return pd.DataFrame([
        {"strategy": "traditional_2_of_4", "family": "CWE-1", "f1": 0.50, "roc_auc": 0.60},
        {"strategy": "traditional_2_of_4", "family": "CWE-2", "f1": 0.40, "roc_auc": 0.55},
        {"strategy": "traditional_2_of_4", "family": "CWE-3", "f1": 0.30, "roc_auc": None},
        {"strategy": "bks_tau_0_5", "family": "CWE-1", "f1": 0.70, "roc_auc": 0.80},
        {"strategy": "bks_tau_0_5", "family": "CWE-2", "f1": None, "roc_auc": 0.75},
        {"strategy": "bks_tau_0_5", "family": "CWE-3", "f1": None, "roc_auc": None},
    ])


def test_paired_difference_scores_abandoned_families_as_zero():
    diffs, _weights, gain, loss = _paired_difference(
        _per_family_with_gaps(), "bks_tau_0_5", "traditional_2_of_4", "f1"
    )
    # Three pairs survive: +0.20, 0 - 0.40, 0 - 0.30. No pair is silently dropped.
    assert sorted(round(d, 2) for d in diffs) == [-0.40, -0.30, 0.20]
    # CWE-2 and CWE-3: the baseline covers them, the strategy does not
    assert (gain, loss) == (0, 2)


def test_paired_difference_keeps_degenerate_metric_nan_out_of_the_pairs():
    diffs, _weights, gain, loss = _paired_difference(
        _per_family_with_gaps(), "bks_tau_0_5", "traditional_2_of_4", "roc_auc"
    )
    # CWE-3 is NaN on both sides and drops out; the other two pairs stay.
    assert sorted(round(d, 2) for d in diffs) == [0.20, 0.20]
    # NaN does not mean "silent" for roc_auc, so the coverage counters stay at zero.
    assert (gain, loss) == (0, 0)


def _per_family_weighted():
    # CWE-1 is 9x the support of CWE-2 and is where the strategy loses; the unweighted
    # mean is positive, the support-weighted one negative.
    return pd.DataFrame([
        {"strategy": "traditional_2_of_4", "family": "CWE-1", "f1": 0.50, "positive_support": 90},
        {"strategy": "traditional_2_of_4", "family": "CWE-2", "f1": 0.10, "positive_support": 10},
        {"strategy": "bks_tau_0_5", "family": "CWE-1", "f1": 0.40, "positive_support": 90},
        {"strategy": "bks_tau_0_5", "family": "CWE-2", "f1": 0.60, "positive_support": 10},
    ])


def test_weighted_difference_and_effective_n_are_reported():
    row = mean_difference_ci(
        _per_family_weighted(), metric="f1", preselected_strategies=["bks_tau_0_5"]
    ).iloc[0]
    # unweighted: mean(-0.10, +0.50) = +0.20 ; weighted: (-0.10*90 + 0.50*10)/100 = -0.04
    assert row["mean_difference"] == approx(0.20)
    assert row["weighted_difference"] == approx(-0.04)
    # Kish: 100^2 / (90^2 + 10^2) = 1.22 of 2 families
    assert row["n_effective"] == approx(100**2 / (90**2 + 10**2))
    assert row["n_effective"] < 2


def test_weighted_difference_is_nan_without_support_column():
    per_family = _per_family_weighted().drop(columns="positive_support")
    row = mean_difference_ci(
        per_family, metric="f1", preselected_strategies=["bks_tau_0_5"]
    ).iloc[0]
    assert pd.isna(row["weighted_difference"]) and pd.isna(row["n_effective"])


def test_weights_stay_aligned_when_pairs_are_dropped():
    """A dropped pair must drop its weight too, or the weighted mean silently shifts."""
    per_family = pd.DataFrame([
        # CWE-2 is NaN on one side only: for roc_auc that pair carries no information
        # and is dropped -- its (large) weight must not be applied to another family.
        {"strategy": "traditional_2_of_4", "family": "CWE-1", "roc_auc": 0.50, "positive_support": 10},
        {"strategy": "traditional_2_of_4", "family": "CWE-2", "roc_auc": 0.50, "positive_support": 900},
        {"strategy": "traditional_2_of_4", "family": "CWE-3", "roc_auc": 0.50, "positive_support": 10},
        {"strategy": "bks_tau_0_5", "family": "CWE-1", "roc_auc": 0.70, "positive_support": 10},
        {"strategy": "bks_tau_0_5", "family": "CWE-2", "roc_auc": None, "positive_support": 900},
        {"strategy": "bks_tau_0_5", "family": "CWE-3", "roc_auc": 0.90, "positive_support": 10},
    ])
    diffs, weights, _gain, _loss = _paired_difference(
        per_family, "bks_tau_0_5", "traditional_2_of_4", "roc_auc"
    )
    assert len(diffs) == len(weights) == 2
    assert sorted(round(d, 2) for d in diffs) == [0.20, 0.40]
    assert sorted(weights) == [10, 10]          # the 900-weight pair left with its pair

    row = mean_difference_ci(
        per_family, metric="roc_auc", preselected_strategies=["bks_tau_0_5"]
    ).iloc[0]
    assert row["weighted_difference"] == approx(0.30)   # not skewed by the dropped weight
    assert row["n_pairs"] == 2


def test_zero_differences_are_counted_separately_from_pairs():
    """n_pairs overstates the Wilcoxon's basis: the test discards zero differences."""
    rows = []
    for family, (strategy_f1, baseline_f1) in {
        # four families nobody detects: difference exactly 0, no information either way
        "CWE-1": (0.0, 0.0), "CWE-2": (0.0, 0.0), "CWE-3": (0.0, 0.0), "CWE-4": (0.0, 0.0),
        # two where the strategy genuinely wins
        "CWE-5": (0.6, 0.2), "CWE-6": (0.5, 0.1),
    }.items():
        rows.append({"strategy": "bks_tau_0_5", "family": family, "f1": strategy_f1,
                     "positive_support": 10})
        rows.append({"strategy": "traditional_2_of_4", "family": family, "f1": baseline_f1,
                     "positive_support": 10})

    row = mean_difference_ci(
        pd.DataFrame(rows), metric="f1", preselected_strategies=["bks_tau_0_5"]
    ).iloc[0]
    assert row["n_pairs"] == 6
    assert row["n_nonzero_pairs"] == 2


def test_coverage_counters_survive_zero_filled_input():
    """mean_over_folds zero-fills before averaging, so 'silent' arrives here as 0.0."""
    per_family = pd.DataFrame([
        # baseline covers nothing on CWE-1 (already zero-filled upstream), strategy does
        {"strategy": "traditional_2_of_4", "family": "CWE-1", "f1": 0.0, "positive_support": 10},
        {"strategy": "bks_tau_0_5", "family": "CWE-1", "f1": 0.40, "positive_support": 10},
        # mirror case on CWE-2
        {"strategy": "traditional_2_of_4", "family": "CWE-2", "f1": 0.30, "positive_support": 10},
        {"strategy": "bks_tau_0_5", "family": "CWE-2", "f1": 0.0, "positive_support": 10},
        # neither covers CWE-3: not a gain, not a loss
        {"strategy": "traditional_2_of_4", "family": "CWE-3", "f1": 0.0, "positive_support": 10},
        {"strategy": "bks_tau_0_5", "family": "CWE-3", "f1": 0.0, "positive_support": 10},
    ])
    _diffs, _w, gain, loss = _paired_difference(
        per_family, "bks_tau_0_5", "traditional_2_of_4", "f1"
    )
    assert (gain, loss) == (1, 1)
