import numpy as np
import pandas as pd
from pytest import approx

from auxiliary.pr_curve_plots import CURVE_KEY_SEP, _curve_points, _fine_taus, _pr_at_thresholds


def _brute_force(labels, scores, taus):
    out = []
    for tau in taus:
        pred = scores >= tau
        tp = int((labels & pred).sum())
        precision = tp / int(pred.sum()) if pred.sum() else 0.0
        recall = tp / int(labels.sum())
        out.append((precision, recall))
    return out


def test_pr_at_thresholds_matches_brute_force_including_ties():
    rng = np.random.default_rng(0)
    # Heavy ties on purpose: fusion scores take few distinct values by construction.
    scores = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=400)
    labels = rng.random(400) < (0.2 + 0.5 * scores)
    taus = _fine_taus(0.05)

    precision, recall = _pr_at_thresholds(labels, scores, taus)
    expected = _brute_force(labels, scores, taus)
    for got_p, got_r, (want_p, want_r) in zip(precision, recall, expected, strict=True):
        assert got_p == approx(want_p)
        assert got_r == approx(want_r)


def test_precision_is_zero_filled_when_nothing_is_predicted():
    scores = np.array([0.1, 0.1, 0.1])
    labels = np.array([True, False, True])
    precision, recall = _pr_at_thresholds(labels, scores, np.array([0.9]))
    assert precision[0] == 0.0            # no positive prediction is a failure, not a gap
    assert recall[0] == approx(0.0)


def test_curve_points_weights_families_by_support():
    # One fold, one strategy, two families. At tau=0.5 the big family has precision 1.0
    # and the small one 0.0; the weighted point must lean on the big family.
    big = pd.DataFrame({
        "base_strategy": "s", "family": "CWE-1",
        "label": [True] * 9 + [False], "score": [0.9] * 9 + [0.1],
    })
    small = pd.DataFrame({
        "base_strategy": "s", "family": "CWE-2",
        "label": [True], "score": [0.1],
    })
    curve = _curve_points([pd.concat([big, small], ignore_index=True)], np.array([0.5]))

    row = curve.iloc[0]
    assert row["base_strategy"] == "s" and row["tau"] == approx(0.5)
    # weights are positive support: 9 vs 1 -> (1.0*9 + 0.0*1)/10
    assert row["precision"] == approx(0.9)
    assert row["recall"] == approx(0.9)
    assert CURVE_KEY_SEP not in row["base_strategy"]
