
import pandas as pd

from analysis.folds import (
    cwe_support_counts,
    kept_cwes_by_min_count,
    kept_cwes_for_dataframe,
    multilabel_stratified_kfold,
    prune_rare_cwes,
    row_multilabels,
    split_train_validation,
)


def test_row_multilabels_uses_safe_label_for_non_vulnerable_samples():
    assert row_multilabels([]) == {"SAFE"}
    assert row_multilabels(["CWE-079", "89"]) == {"CWE-79", "CWE-89"}


def test_multilabel_stratified_kfold_assigns_all_rows_once():
    df = pd.DataFrame([
        {"sample_id": "a", "cwes": ["CWE-79"]},
        {"sample_id": "b", "cwes": ["CWE-79", "CWE-89"]},
        {"sample_id": "c", "cwes": ["CWE-89"]},
        {"sample_id": "d", "cwes": []},
        {"sample_id": "e", "cwes": []},
        {"sample_id": "f", "cwes": ["CWE-22"]},
    ])

    folds = multilabel_stratified_kfold(df, n_splits=3, random_state=7)

    assert sorted(folds["row_index"].tolist()) == list(range(len(df)))
    assert set(folds["fold"]) == {0, 1, 2}
    assert folds["sample_id"].tolist() == df["sample_id"].tolist()


def test_cwe_support_counts_excludes_safe_marker():
    labels = [{"CWE-79"}, {"CWE-79", "CWE-89"}, {"SAFE"}]
    assert cwe_support_counts(labels) == {"CWE-79": 2, "CWE-89": 1}


def test_kept_cwes_by_min_count_keeps_only_frequent_cwes():
    labels = [{"CWE-79"}, {"CWE-79", "CWE-89"}, {"CWE-89"}, {"CWE-22"}]
    assert kept_cwes_by_min_count(labels, min_count=2) == {"CWE-79", "CWE-89"}
    assert kept_cwes_by_min_count(labels, min_count=1) == {"CWE-79", "CWE-89", "CWE-22"}


def test_prune_rare_cwes_collapses_emptied_rows_to_safe():
    labels = [{"CWE-79", "CWE-22"}, {"CWE-22"}, {"SAFE"}]
    # CWE-22 occurs twice, CWE-79 once -> with min_count=2 only CWE-22 survives.
    pruned, kept = prune_rare_cwes(labels, min_count=2)
    assert kept == {"CWE-22"}
    # Row 0 keeps CWE-22; row 1 keeps CWE-22; row 2 stays SAFE.
    assert pruned == [{"CWE-22"}, {"CWE-22"}, {"SAFE"}]


def test_prune_rare_cwes_row_with_only_rare_cwe_becomes_safe():
    labels = [{"CWE-79"}, {"CWE-22"}, {"CWE-22"}]
    pruned, kept = prune_rare_cwes(labels, min_count=2)
    assert kept == {"CWE-22"}
    assert pruned[0] == {"SAFE"}  # CWE-79 pruned, nothing left -> SAFE


def test_kept_cwes_for_dataframe_matches_label_based_counting():
    df = pd.DataFrame([
        {"cwes": ["CWE-79"]},
        {"cwes": ["CWE-79", "CWE-89"]},
        {"cwes": []},
        {"cwes": ["CWE-22"]},
    ])
    assert kept_cwes_for_dataframe(df, min_count=2) == {"CWE-79"}


def test_min_cwe_count_prunes_stratification_without_dropping_rows():
    df = pd.DataFrame([
        {"sample_id": "a", "cwes": ["CWE-79"]},
        {"sample_id": "b", "cwes": ["CWE-79"]},
        {"sample_id": "c", "cwes": ["CWE-22"]},  # rare -> pruned from stratification
        {"sample_id": "d", "cwes": []},
        {"sample_id": "e", "cwes": []},
        {"sample_id": "f", "cwes": ["CWE-79"]},
    ])
    folds = multilabel_stratified_kfold(df, n_splits=2, random_state=0, min_cwe_count=2)
    # All rows are still assigned exactly once; pruning only changes balancing.
    assert sorted(folds["row_index"].tolist()) == list(range(len(df)))
    assert folds["sample_id"].tolist() == df["sample_id"].tolist()


def test_split_train_validation_uses_fold_assignments():
    df = pd.DataFrame([
        {"sample_id": "a", "cwes": ["CWE-79"]},
        {"sample_id": "b", "cwes": []},
        {"sample_id": "c", "cwes": ["CWE-89"]},
    ])
    folds = pd.DataFrame({"row_index": [0, 1, 2], "fold": [0, 1, 0]})

    train, validation = split_train_validation(df, folds, validation_fold=1)

    assert validation["sample_id"].tolist() == ["b"]
    assert train["sample_id"].tolist() == ["a", "c"]
