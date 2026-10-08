import numpy as np
import pandas as pd
from pytest import approx

from analysis.complementarity import (
    catch_combinations,
    coverage_report,
    diversity_report,
    family_matrices,
)


def test_coverage_report_recall_oracle_and_unique():
    pos = np.array([True, True, True, True])
    catch = {
        "A": np.array([True, True, False, False]),   # catches 0,1
        "B": np.array([False, False, True, False]),  # catches 2
    }
    per_tool, summary = coverage_report(catch, pos)
    rec = per_tool.set_index("tool")["recall"]
    assert rec["A"] == approx(0.5) and rec["B"] == approx(0.25)
    assert summary["best_tool"] == "A"
    assert summary["or_recall"] == approx(0.75)        # 0,1,2 caught
    assert summary["headroom"] == approx(0.25)         # 0.75 - 0.5
    assert summary["missed_by_all"] == 1               # instance 3
    assert per_tool.set_index("tool")["unique_catches"]["A"] == 2


def test_catch_combinations_counts_membership():
    pos = np.array([True, True, True, True])
    catch = {"A": np.array([True, True, False, False]), "B": np.array([False, True, True, False])}
    combos = catch_combinations(catch, pos).set_index("tools")["count"]
    assert combos["A"] == 1          # instance 0: only A
    assert combos["A+B"] == 1        # instance 1: both
    assert combos["B"] == 1          # instance 2: only B
    assert combos["∅ (missed)"] == 1 # instance 3: none


def test_diversity_report_double_fault_and_q():
    correct = {
        "A": np.array([True, True, False, False]),
        "B": np.array([True, False, True, False]),
    }
    row = diversity_report(correct).iloc[0]
    assert row["double_fault"] == approx(0.25)   # both wrong on instance 3
    assert row["disagreement"] == approx(0.5)
    assert row["q_statistic"] == approx(0.0)     # independent errors


def test_family_matrices_shapes_and_membership():
    cdf = pd.DataFrame([
        {"cwes": ["CWE-1"], "toolA": ["CWE-1"]},
        {"cwes": ["CWE-2"], "toolA": []},
    ])
    fires, gt = family_matrices(cdf, ["toolA"], ["CWE-1", "CWE-2"])
    assert gt.shape == (2, 2)
    assert gt[0, 0] and gt[1, 1]
    assert fires["toolA"][0, 0] and not fires["toolA"][1, 1]
