import json
import pytest
from pathlib import Path
from ingestion.primevul import extract_primevul
from ingestion.schema import FunctionSample


FIXTURE = [
    # commit "aaa": single target=1, CWE present -> valid positive
    {"func": "void vuln() { return buf[i]; }", "cwe": "CWE-119", "target": 1,
     "commit_id": "aaa", "project": "proj"},
    # commit "aaa": target=0 -> valid negative
    {"func": "void safe() { return 0; }", "cwe": "", "target": 0,
     "commit_id": "aaa", "project": "proj"},
    # commit "bbb": two target=1 entries -> both filtered out (multi-function commit)
    {"func": "void vuln2() {}", "cwe": "CWE-787", "target": 1,
     "commit_id": "bbb", "project": "proj"},
    {"func": "void vuln3() {}", "cwe": "CWE-119", "target": 1,
     "commit_id": "bbb", "project": "proj"},
    # commit "ccc": single target=1 but no CWE -> filtered out
    {"func": "void vuln4() {}", "cwe": "", "target": 1,
     "commit_id": "ccc", "project": "proj"},
    # commit "ddd": single target=1, multi-CWE string -> valid positive with 2 CWEs
    {"func": "void vuln5() {}", "cwe": "['CWE-119', 'CWE-787']", "target": 1,
     "commit_id": "ddd", "project": "proj"},
    # commit "eee": concatenated CWE string -> split into 2 CWEs
    {"func": "void vuln6() {}", "cwe": "['CWE-20CWE-190']", "target": 1,
     "commit_id": "eee", "project": "proj"},
    # commit "fff": NVD placeholder -> filtered out (no valid CWE)
    {"func": "void nvd() {}", "cwe": "['NVD-CWE-noinfo']", "target": 1,
     "commit_id": "fff", "project": "proj"},
    # commit "ggg": non-numeric CWE -> filtered out
    {"func": "void other() {}", "cwe": "CWE-Other", "target": 1,
     "commit_id": "ggg", "project": "proj"},
]


@pytest.fixture
def primevul_path(tmp_path):
    p = tmp_path / "primevul.json"
    p.write_text(json.dumps(FIXTURE))
    return p


def test_extracts_single_function_commit_positive(primevul_path):
    samples = extract_primevul(primevul_path)
    positives = [s for s in samples if s.label == 1]
    codes = [s.code for s in positives]
    assert "void vuln() { return buf[i]; }" in codes


def test_filters_multi_function_commit_positives(primevul_path):
    samples = extract_primevul(primevul_path)
    codes = [s.code for s in samples]
    assert "void vuln2() {}" not in codes
    assert "void vuln3() {}" not in codes


def test_filters_positive_with_no_cwe(primevul_path):
    samples = extract_primevul(primevul_path)
    codes = [s.code for s in samples]
    assert "void vuln4() {}" not in codes


def test_keeps_all_negatives(primevul_path):
    samples = extract_primevul(primevul_path)
    negatives = [s for s in samples if s.label == 0]
    assert any(s.code == "void safe() { return 0; }" for s in negatives)
    assert all(s.cwes == [] for s in negatives)


def test_parses_multi_cwe_string(primevul_path):
    samples = extract_primevul(primevul_path)
    multi = next(s for s in samples if s.code == "void vuln5() {}")
    assert set(multi.cwes) == {"CWE-119", "CWE-787"}


def test_returns_function_samples(primevul_path):
    samples = extract_primevul(primevul_path)
    assert all(isinstance(s, FunctionSample) for s in samples)


def test_splits_concatenated_cwes(primevul_path):
    """'CWE-20CWE-190' must be split into two separate CWE IDs."""
    samples = extract_primevul(primevul_path)
    concat = next(s for s in samples if s.code == "void vuln6() {}")
    assert set(concat.cwes) == {"CWE-20", "CWE-190"}


def test_filters_nvd_placeholder(primevul_path):
    """NVD placeholder CWEs must drop the positive entirely (no valid CWE left)."""
    samples = extract_primevul(primevul_path)
    codes = [s.code for s in samples]
    assert "void nvd() {}" not in codes


def test_filters_non_numeric_cwe(primevul_path):
    """'CWE-Other' (no numeric suffix) must drop the positive."""
    samples = extract_primevul(primevul_path)
    codes = [s.code for s in samples]
    assert "void other() {}" not in codes


def test_loads_jsonl(tmp_path):
    p = tmp_path / "primevul.jsonl"
    p.write_text(
        '{"func":"void v(){}","cwe":"CWE-119","target":1,"commit_id":"x","project":"p"}\n'
        '\n'
        '{"func":"void s(){}","cwe":"","target":0,"commit_id":"x","project":"p"}\n',
        encoding="utf-8",
    )
    samples = extract_primevul(p)
    assert any(s.label == 1 and s.cwes == ["CWE-119"] for s in samples)
    assert any(s.label == 0 for s in samples)


def test_loads_parquet(tmp_path):
    import pandas as pd
    p = tmp_path / "primevul.parquet"
    pd.DataFrame([
        {"func": "void v(){}", "cwe": "CWE-22", "target": 1, "commit_id": "u", "project": "p"},
    ]).to_parquet(p, index=False)
    samples = extract_primevul(p)
    assert samples and samples[0].cwes == ["CWE-22"]


def test_parse_cwes_handles_bad_list_literal(tmp_path):
    # Unparseable list literal -> []; positive is then filtered out (no CWE).
    p = tmp_path / "primevul.json"
    p.write_text(json.dumps([
        {"func": "void v(){}", "cwe": "[bad python literal", "target": 1, "commit_id": "z", "project": "p"},
    ]))
    samples = extract_primevul(p)
    assert samples == []


def test_parse_cwes_handles_none_value(tmp_path):
    p = tmp_path / "primevul.json"
    p.write_text(json.dumps([
        {"func": "void s(){}", "cwe": None, "target": 0, "commit_id": "z", "project": "p"},
    ]))
    samples = extract_primevul(p)
    assert samples and samples[0].label == 0
