import pandas as pd
from pytest import approx

from analysis.aggregation import (
    aggregate_fusion_metrics,
    mean_over_folds,
    support_weighted_over_families,
    zero_fill,
)


def _per_family_per_fold():
    # One strategy, two families, two folds. f1 metric chosen so the maths is checkable.
    return pd.DataFrame([
        {"fold": 0, "strategy": "S", "family": "CWE-1", "f1": 0.8, "positive_support": 8},
        {"fold": 1, "strategy": "S", "family": "CWE-1", "f1": 0.6, "positive_support": 8},
        {"fold": 0, "strategy": "S", "family": "CWE-2", "f1": 0.2, "positive_support": 2},
        {"fold": 1, "strategy": "S", "family": "CWE-2", "f1": 0.4, "positive_support": 2},
    ])


def test_stage1_means_metrics_and_sums_support():
    per_family = mean_over_folds(_per_family_per_fold(), metric_columns=("f1",))
    by_family = per_family.set_index("family")
    assert by_family.at["CWE-1", "f1"] == approx(0.7)   # mean(0.8, 0.6)
    assert by_family.at["CWE-2", "f1"] == approx(0.3)   # mean(0.2, 0.4)
    assert by_family.at["CWE-1", "positive_support"] == 16  # summed across folds
    assert by_family.at["CWE-2", "positive_support"] == 4


def test_stage2_support_weighted_macro_median():
    per_family = mean_over_folds(_per_family_per_fold(), metric_columns=("f1",))
    overall = support_weighted_over_families(per_family, metric_columns=("f1",)).iloc[0]
    # weighted by summed support 16 vs 4: (0.7*16 + 0.3*4) / 20 = 0.62
    assert abs(overall["f1_weighted"] - 0.62) < 1e-9
    assert abs(overall["f1_macro"] - 0.5) < 1e-9        # mean(0.7, 0.3)
    assert abs(overall["f1_median"] - 0.5) < 1e-9


def test_aggregate_fusion_metrics_returns_both_stages():
    per_family, overall = aggregate_fusion_metrics(_per_family_per_fold(), metric_columns=("f1",))
    assert set(per_family["family"]) == {"CWE-1", "CWE-2"}
    assert "f1_weighted" in overall.columns and len(overall) == 1


def _per_family_per_fold_with_gaps():
    # Strategy S is silent on CWE-2 in fold 1 (f1 NaN); roc_auc is NaN on CWE-2 in fold 0
    # because that slice is degenerate, not because S failed.
    return pd.DataFrame([
        {"fold": 0, "strategy": "S", "family": "CWE-1", "f1": 0.8, "roc_auc": 0.9, "positive_support": 8},
        {"fold": 1, "strategy": "S", "family": "CWE-1", "f1": 0.6, "roc_auc": 0.7, "positive_support": 8},
        {"fold": 0, "strategy": "S", "family": "CWE-2", "f1": 0.4, "roc_auc": None, "positive_support": 2},
        {"fold": 1, "strategy": "S", "family": "CWE-2", "f1": None, "roc_auc": 0.5, "positive_support": 2},
    ])


def test_zero_fill_applies_only_to_failure_metrics():
    values = pd.Series([0.5, None])
    assert zero_fill(values, "f1").tolist() == [0.5, 0.0]
    assert zero_fill(values, "precision").tolist() == [0.5, 0.0]
    assert zero_fill(values, "roc_auc").isna().tolist() == [False, True]
    assert zero_fill(values, "fnr").isna().tolist() == [False, True]


def test_stage1_zero_fills_silent_folds_but_not_degenerate_ones():
    per_family = mean_over_folds(_per_family_per_fold_with_gaps(), metric_columns=("f1", "roc_auc"))
    by_family = per_family.set_index("family")
    # f1: the silent fold counts as 0.0 -> mean(0.4, 0.0), not mean(0.4)
    assert by_family.at["CWE-2", "f1"] == approx(0.2)
    # roc_auc: the degenerate fold carries no information -> mean over the fold where it exists
    assert by_family.at["CWE-2", "roc_auc"] == approx(0.5)


def test_stage2_drops_nan_from_weighted_mean_without_nan_result():
    per_family = pd.DataFrame([
        {"strategy": "S", "family": "CWE-1", "roc_auc": 0.9, "positive_support": 8},
        {"strategy": "S", "family": "CWE-2", "roc_auc": None, "positive_support": 2},
    ])
    overall = support_weighted_over_families(per_family, metric_columns=("roc_auc",)).iloc[0]
    # CWE-2 is excluded and the weights renormalise; the result must not be NaN
    assert overall["roc_auc_weighted"] == approx(0.9)
    assert overall["roc_auc_macro"] == approx(0.9)
