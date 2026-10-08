import csv
import pytest
from pathlib import Path
from ingestion.secvuleval import extract_secvuleval
from ingestion.schema import FunctionSample


FIXTURE_ROWS = [
    # Positive, single CWE -> valid
    {"func_body": "void vuln1() {}", "is_vulnerable": True,  "cwe_list": "['CWE-119']"},
    # Positive, multi-CWE proper list -> valid with 2 CWEs
    {"func_body": "void vuln2() {}", "is_vulnerable": True,  "cwe_list": "['CWE-787', 'CWE-119']"},
    # Positive, concatenated CWE string -> split into 2 separate CWEs
    {"func_body": "void vuln_concat() {}","is_vulnerable": True,  "cwe_list": "['CWE-20CWE-190']"},
    # Positive, NVD placeholder -> filtered out (no resolvable CWE)
    {"func_body": "void vuln_nvd() {}",  "is_vulnerable": True,  "cwe_list": "['NVD-CWE-Other']"},
    # Positive, empty cwe_list -> filtered out
    {"func_body": "void vuln3() {}",     "is_vulnerable": True,  "cwe_list": "[]"},
    # Negative -> kept as-is, no CWE
    {"func_body": "void safe1() {}",     "is_vulnerable": False, "cwe_list": "[]"},
    {"func_body": "void safe2() {}",     "is_vulnerable": False, "cwe_list": "nan"},
]


@pytest.fixture
def secvuleval_csv(tmp_path):
    path = tmp_path / "secvuleval.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["func_body", "is_vulnerable", "cwe_list"])
        writer.writeheader()
        writer.writerows(FIXTURE_ROWS)
    return path


def test_extracts_resolvable_positive(secvuleval_csv):
    samples = extract_secvuleval(secvuleval_csv)
    codes = [s.code for s in samples]
    assert "void vuln1() {}" in codes


def test_assigns_cwe_from_nvd_cache(secvuleval_csv):
    samples = extract_secvuleval(secvuleval_csv)
    vuln1 = next(s for s in samples if s.code == "void vuln1() {}")
    assert vuln1.cwes == ["CWE-119"]


def test_assigns_multi_cwe(secvuleval_csv):
    samples = extract_secvuleval(secvuleval_csv)
    vuln2 = next(s for s in samples if s.code == "void vuln2() {}")
    assert set(vuln2.cwes) == {"CWE-787", "CWE-119"}


def test_splits_concatenated_cwes(secvuleval_csv):
    samples = extract_secvuleval(secvuleval_csv)
    vuln = next(s for s in samples if s.code == "void vuln_concat() {}")
    assert set(vuln.cwes) == {"CWE-20", "CWE-190"}


def test_filters_nvd_placeholder(secvuleval_csv):
    samples = extract_secvuleval(secvuleval_csv)
    codes = [s.code for s in samples]
    assert "void vuln_nvd() {}" not in codes


def test_filters_unresolvable_positive(secvuleval_csv):
    samples = extract_secvuleval(secvuleval_csv)
    codes = [s.code for s in samples]
    assert "void vuln3() {}" not in codes


def test_keeps_all_negatives(secvuleval_csv):
    samples = extract_secvuleval(secvuleval_csv)
    negatives = [s for s in samples if s.label == 0]
    neg_codes = [s.code for s in negatives]
    assert "void safe1() {}" in neg_codes
    assert "void safe2() {}" in neg_codes


def test_negatives_have_empty_cwes(secvuleval_csv):
    samples = extract_secvuleval(secvuleval_csv)
    for s in samples:
        if s.label == 0:
            assert s.cwes == []


def test_returns_function_samples(secvuleval_csv):
    samples = extract_secvuleval(secvuleval_csv)
    assert all(isinstance(s, FunctionSample) for s in samples)


def test_handles_missing_cwe_list_column(tmp_path):
    # Empty CSV cell becomes NaN (a float) after pandas read_csv;
    # _parse_cwe_list must handle the float branch.
    import pandas as pd
    p = tmp_path / "secvul.csv"
    pd.DataFrame([
        {"func_body": "void v(){}", "is_vulnerable": True,  "cwe_list": None},
        {"func_body": "void s(){}", "is_vulnerable": False, "cwe_list": None},
    ]).to_csv(p, index=False)
    samples = extract_secvuleval(p)
    # The vulnerable row has no CWE -> filtered. Negative is kept.
    assert len(samples) == 1 and samples[0].label == 0


def test_handles_unparseable_list_literal(tmp_path):
    import pandas as pd
    p = tmp_path / "secvul.csv"
    pd.DataFrame([
        {"func_body": "void v(){}", "is_vulnerable": True,  "cwe_list": "[unparseable"},
    ]).to_csv(p, index=False)
    samples = extract_secvuleval(p)
    assert samples == []
