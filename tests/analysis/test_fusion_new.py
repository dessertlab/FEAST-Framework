import numpy as np
import pandas as pd
from pytest import approx
from analysis.calibration import compute_reliability
from analysis.fusion import run_fusion
from analysis.fusion.common import DEFAULT_METRIC_PAIRS, build_fire_index, metric_lookup, metric_pairs_for_calibration, precompute_labels, taus_in_range
from analysis.fusion.bks import bks_predictions
from analysis.fusion.dst import _combine_dempster, _combine_yager
from analysis.fusion.predictions import DEFAULT_TAUS, tau_variant_table


def _df():
    return pd.DataFrame([
        {"sample_id": "a", "label": 1, "cwes": ["CWE-1"], "toolA": ["CWE-1"], "toolB": ["CWE-1"]},
        {"sample_id": "b", "label": 1, "cwes": ["CWE-1"], "toolA": ["CWE-1"], "toolB": []},
        {"sample_id": "c", "label": 0, "cwes": [],        "toolA": [],        "toolB": []},
        {"sample_id": "d", "label": 1, "cwes": ["CWE-2"], "toolA": ["CWE-2"], "toolB": ["CWE-2"]},
        {"sample_id": "e", "label": 0, "cwes": [],        "toolA": ["CWE-1"], "toolB": []},
    ])


def _ctx(df, tools=("toolA", "toolB"), families=("CWE-1", "CWE-2")):
    tools, families = list(tools), list(families)
    rel = compute_reliability(df, tools, families)
    return tools, families, metric_lookup(rel), precompute_labels(df, families), build_fire_index(df, tools), rel


def test_run_fusion_includes_explicit_tau_strategy_variants():
    df = _df()
    tools, families, _, _, _, rel = _ctx(df)
    preds = run_fusion(df, df, rel, tools, families, threshold=2)
    strategies = set(preds["strategy"])
    for expected in {
        "bks_tau_0_5",
        "dst_yager_fire_ppv_silence_npv_tau_0_5",
        "logistic_interactions_tau_0_5",
        "naive_bayes_tau_0_5",
        "weighted_fire_ppv_silence_npv_tau_0_5",
    }:
        assert expected in strategies


def test_metric_pair_expansion_excludes_hybrids_and_uses_requested_pairs():
    assert DEFAULT_METRIC_PAIRS == (
        ("ppv", "npv"),
        ("specificity", "sensitivity"),
        ("fpr", "fnr"),
    )

    df = _df()
    tools, families, _, _, _, rel = _ctx(df)
    strategies = set(run_fusion(df, df, rel, tools, families, threshold=2)["strategy"])

    assert "weighted_fire_fpr_silence_fnr_tau_0_5" in strategies
    assert "dst_yager_fire_specificity_silence_sensitivity_tau_0_5" in strategies
    hybrid_a = f"weighted_fire_{'ppv'}_silence_{'sensitivity'}_tau_0_5"
    hybrid_b = f"weighted_fire_{'specificity'}_silence_{'npv'}_tau_0_5"
    old_supported_names = ("_".join(("of", "supported")), "_".join(("out", "of", "supported")))
    assert hybrid_a not in strategies
    assert hybrid_b not in strategies
    assert not any(any(old in strategy for old in old_supported_names) for strategy in strategies)


def test_configured_calibration_metrics_and_tau_range_prune_strategy_variants():
    assert metric_pairs_for_calibration(["sensitivity", "specificity"]) == (("specificity", "sensitivity"),)
    assert taus_in_range(0.1, 0.8) == tuple(round(0.1 * i, 1) for i in range(1, 9))

    df = _df()
    tools, families, _, _, _, rel = _ctx(df)
    preds = run_fusion(
        df, df, rel, tools, families, threshold=2,
        calibration_metrics=["sensitivity", "specificity"],
        taus=taus_in_range(0.1, 0.8),
    )
    strategies = set(preds["strategy"])

    assert "weighted_fire_specificity_silence_sensitivity_tau_0_8" in strategies
    assert "dst_yager_fire_specificity_silence_sensitivity_tau_0_8" in strategies
    assert "bks_tau_0_8" in strategies
    assert "logistic_regression_tau_0_8" in strategies
    assert not any("_tau_0_9" in strategy for strategy in strategies)
    assert not any("weighted_fire_ppv_silence_npv" in strategy for strategy in strategies)
    assert not any("dst_yager_fire_ppv_silence_npv" in strategy for strategy in strategies)
    assert not any("weighted_fire_fpr_silence_fnr" in strategy for strategy in strategies)


def test_bks_scores_are_probabilities_and_cover_all_rows():
    df = _df()
    tools, families, _, labels, fire_index, _ = _ctx(df)
    preds = bks_predictions(df, df, tools, families, labels, fire_index)
    assert ((preds["score"] >= 0) & (preds["score"] <= 1)).all()
    assert len(preds) == len(df) * len(families)


def test_yager_unnormalised_mass_le_dempster():
    # On conflict, Yager leaves mass on ignorance: mV+mS <= the normalised Dempster masses.
    v = np.array([[0.6, 0.0]]); s = np.array([[0.0, 0.7]]); t = 1.0 - v - s
    yV, yS = _combine_yager(v, s, t)
    dV, dS = _combine_dempster(v, s, t)
    assert yV + yS <= dV + dS + 1e-9


def test_canonical_strategy_order():
    from analysis.fusion.common import canonical_strategy_order
    present = ["bks_tau_0_5", "tool:codeql", "logistic_regression_tau_0_1", "or_1_of_2",
               "tool:bandit", "traditional_2_of_2", "naive_bayes_tau_0_9", "always_vulnerable"]
    order = canonical_strategy_order(["bandit", "codeql"], present)
    # single tools first (in tool order), then OR, then traditional, then fusers, unknown last
    assert order[:2] == ["tool:bandit", "tool:codeql"]
    assert order[2] == "or_1_of_2"
    assert order[3] == "traditional_2_of_2"
    assert order.index("naive_bayes_tau_0_9") < order.index("logistic_regression_tau_0_1")
    assert order[-1] == "always_vulnerable"


def test_run_fusion_includes_single_tools_and_or():
    df = _df()
    tools, families, _, _, _, rel = _ctx(df)
    strategies = set(run_fusion(df, df, rel, tools, families)["strategy"])
    assert {"tool:toolA", "tool:toolB", "or_1_of_2"} <= strategies


def test_tau_variants_are_materialised_as_strategies():
    df = _df()
    tools, families, _, _, _, rel = _ctx(df)
    preds = run_fusion(df, df, rel, tools, families, threshold=2)
    bks = {s for s in preds["strategy"] if s.startswith("bks_tau_")}
    assert len(bks) == len(DEFAULT_TAUS)

    metrics = pd.DataFrame({"strategy": sorted(bks), "mcc": np.arange(len(bks), dtype=float)})
    table = tau_variant_table(metrics)
    assert set(table["tau"]) == set(DEFAULT_TAUS)
    assert set(table["base_strategy"]) == {"bks"}


def test_score_histogram_reproduces_per_family_metrics_exactly():
    """The persisted histogram must be a replacement for per-row scores, not a summary.

    If metrics recovered from the histogram differ from those computed on the rows, the
    file is a lossy digest and any later analysis built on it silently drifts from the
    pipeline it claims to describe.
    """
    import numpy as np
    from analysis.fusion.predictions import (
        binary_metrics,
        metrics_from_histogram,
        score_histogram,
    )

    rng = np.random.default_rng(7)
    n = 900
    # Few distinct scores, heavy ties: the real shape of a fire-vector-derived score.
    scores = rng.choice([0.02, 0.05, 0.13, 0.31, 0.68], size=n)
    predictions = pd.DataFrame({
        "strategy": rng.choice(["bks_tau_0_5", "naive_bayes_tau_0_5"], size=n),
        "family": rng.choice(["CWE-1", "CWE-2", "CWE-3"], size=n),
        "score": scores,
        "label": rng.random(n) < (0.1 + scores),
    })

    taus = [0.01, 0.05, 0.10, 0.31, 0.50, 0.90]
    recovered = metrics_from_histogram(score_histogram(predictions), taus).set_index(
        ["strategy", "family", "tau"]
    )

    base = predictions.assign(
        strategy=predictions["strategy"].str.replace(r"_tau_\d_\d$", "", regex=True)
    )
    for (strategy, family), group in base.groupby(["strategy", "family"]):
        for tau in taus:
            direct = binary_metrics(
                group["label"].tolist(),
                (group["score"] >= tau).tolist(),
                group["score"].tolist(),
            )
            row = recovered.loc[(strategy, family, tau)]
            assert row["positive_support"] == direct["positive_support"]
            assert row["recall"] == approx(direct["recall"])
            # binary_metrics returns None for 0/0 precision; the histogram zero-fills it,
            # which is the project-wide convention for "made no positive prediction".
            assert row["precision"] == approx(direct["precision"] or 0.0)
            assert row["f1"] == approx(direct["f1"] or 0.0)
