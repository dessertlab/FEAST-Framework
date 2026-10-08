import pandas as pd

from analysis.calibration import compute_reliability, gt_family_support


def _df():
    # Families already canonicalised. 4 rows, one tool.
    # row0: GT={F1}, tool fires F1   -> TP for F1
    # row1: GT={F1}, tool silent     -> FN for F1
    # row2: GT={},   tool fires F1   -> FP for F1
    # row3: GT={},   tool silent     -> TN for F1
    return pd.DataFrame([
        {"cwes": ["CWE-1"], "toolA": ["CWE-1"]},
        {"cwes": ["CWE-1"], "toolA": []},
        {"cwes": [],        "toolA": ["CWE-1"]},
        {"cwes": [],        "toolA": []},
    ])


def test_exact_confusion_counts():
    rel = compute_reliability(_df(), ["toolA"], ["CWE-1"])
    row = rel.iloc[0]
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (1, 1, 1, 1)
    assert row["ppv"] == 0.5 and row["npv"] == 0.5
    assert row["positive_support"] == 2
    assert bool(row["supported"]) is True


def test_unfired_family_is_unsupported():
    df = pd.DataFrame([{"cwes": ["CWE-2"], "toolA": []}, {"cwes": [], "toolA": []}])
    rel = compute_reliability(df, ["toolA"], ["CWE-2"])
    row = rel.iloc[0]
    assert (row["tp"], row["fp"]) == (0, 0)
    assert bool(row["supported"]) is False  # tp+fp == 0


def test_no_cross_family_credit():
    # tool fires F2 on a GT-F1 row: F1 gets an FN, F2 gets an FP — never a TP.
    df = pd.DataFrame([{"cwes": ["CWE-1"], "toolA": ["CWE-2"]}])
    rel = compute_reliability(df, ["toolA"], ["CWE-1", "CWE-2"]).set_index("family")
    assert (rel.at["CWE-1", "fn"], rel.at["CWE-1", "tp"]) == (1, 0)
    assert (rel.at["CWE-2", "fp"], rel.at["CWE-2", "tp"]) == (1, 0)


def test_gt_family_support_counts_occurrences():
    support = gt_family_support(_df(), ["CWE-1", "CWE-9"])
    assert support == {"CWE-1": 2, "CWE-9": 0}
