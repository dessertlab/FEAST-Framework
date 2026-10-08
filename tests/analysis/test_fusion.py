import pandas as pd

from analysis.calibration import compute_reliability
from analysis.fusion import detection_from_predictions, evaluate_predictions, run_fusion


def _canonicalised_df():
    # Families already assigned. Two tools, families CWE-1 / CWE-2.
    return pd.DataFrame([
        {"sample_id": "a", "label": 1, "cwes": ["CWE-1"], "toolA": ["CWE-1"], "toolB": ["CWE-1"]},
        {"sample_id": "b", "label": 1, "cwes": ["CWE-1"], "toolA": ["CWE-1"], "toolB": []},
        {"sample_id": "c", "label": 0, "cwes": [],        "toolA": [],        "toolB": []},
        {"sample_id": "d", "label": 1, "cwes": ["CWE-2"], "toolA": ["CWE-2"], "toolB": ["CWE-2"]},
        {"sample_id": "e", "label": 0, "cwes": [],        "toolA": ["CWE-1"], "toolB": []},
    ])


def test_run_fusion_emits_shared_schema_for_every_strategy():
    df = _canonicalised_df()
    tools, families = ["toolA", "toolB"], ["CWE-1", "CWE-2"]
    reliability = compute_reliability(df, tools, families)
    preds = run_fusion(df, df, reliability, tools, families, threshold=2)

    assert {"strategy", "family", "row_index", "prediction", "score", "label"} <= set(preds.columns)
    # every (strategy, family) pair scored on every row
    per_strategy_rows = preds.groupby("strategy").size()
    assert (per_strategy_rows == len(df) * len(families)).all()
    # labels follow exact family membership: row 0 (GT={CWE-1}) is positive for CWE-1,
    # negative for CWE-2, regardless of strategy.
    row0 = preds[preds["row_index"] == 0]
    assert row0[row0["family"] == "CWE-1"]["label"].all()
    assert not row0[row0["family"] == "CWE-2"]["label"].any()


def test_evaluate_predictions_groups_per_strategy_family():
    df = _canonicalised_df()
    tools, families = ["toolA", "toolB"], ["CWE-1", "CWE-2"]
    reliability = compute_reliability(df, tools, families)
    preds = run_fusion(df, df, reliability, tools, families, threshold=2)
    metrics = evaluate_predictions(preds, group_cols=("strategy", "family"))
    assert {"strategy", "family", "precision", "recall", "f2"} <= set(metrics.columns)
    assert len(metrics) == preds["strategy"].nunique() * len(families)


def test_detection_collapses_to_row_level():
    df = _canonicalised_df()
    tools, families = ["toolA", "toolB"], ["CWE-1", "CWE-2"]
    reliability = compute_reliability(df, tools, families)
    preds = run_fusion(df, df, reliability, tools, families, threshold=1)
    det = detection_from_predictions(preds, df)
    # one detection row per (strategy, dataframe row)
    assert len(det) == preds["strategy"].nunique() * len(df)
    assert set(det["label"].unique()) <= {True, False}
