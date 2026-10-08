import pandas as pd
import pytest
from pathlib import Path
from ingestion.megavul import extract_megavul
from ingestion.schema import FunctionSample


def _write_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    d = tmp_path / "megavul"
    d.mkdir()
    pd.DataFrame(rows).to_parquet(d / "train-00000.parquet", index=False)
    return d


@pytest.fixture
def megavul_dir(tmp_path):
    rows = [
        # hash_a -- single-function, valid CWE, high CVSS -> positive
        {"hash": "aaa", "vulnerable_code": "void vuln() {}",     "cwe_id": "CWE-119", "cvss3_base_score": 9.8},
        # hash_b -- two functions -> filtered (multi-function commit)
        {"hash": "bbb", "vulnerable_code": "void vuln2() {}",    "cwe_id": "CWE-787", "cvss3_base_score": 7.5},
        {"hash": "bbb", "vulnerable_code": "void vuln3() {}",    "cwe_id": "CWE-119", "cvss3_base_score": 7.5},
        # hash_c -- no CWE -> filtered
        {"hash": "ccc", "vulnerable_code": "void vuln4() {}",    "cwe_id": None,      "cvss3_base_score": 5.0},
        # hash_d -- single-function, low CVSS -> filtered when threshold set
        {"hash": "ddd", "vulnerable_code": "void low_cvss() {}", "cwe_id": "CWE-476", "cvss3_base_score": 3.1},
        # hash_e -- single-function, null CVSS -> filtered when threshold set
        {"hash": "eee", "vulnerable_code": "void null_cvss() {}","cwe_id": "CWE-476", "cvss3_base_score": None},
    ]
    return _write_parquet(tmp_path, rows)


def test_extracts_valid_positive(megavul_dir):
    samples = extract_megavul(megavul_dir)
    assert any(s.code == "void vuln() {}" for s in samples)


def test_filters_multi_function_commit(megavul_dir):
    samples = extract_megavul(megavul_dir)
    codes = [s.code for s in samples]
    assert "void vuln2() {}" not in codes
    assert "void vuln3() {}" not in codes


def test_filters_missing_cwe(megavul_dir):
    samples = extract_megavul(megavul_dir)
    codes = [s.code for s in samples]
    assert "void vuln4() {}" not in codes


def test_no_cvss_filter_by_default(megavul_dir):
    samples = extract_megavul(megavul_dir)
    codes = [s.code for s in samples]
    assert "void low_cvss() {}" in codes
    assert "void null_cvss() {}" in codes


def test_cvss_threshold_filters_low_and_null(megavul_dir):
    samples = extract_megavul(megavul_dir, cvss_threshold=7.0)
    codes = [s.code for s in samples]
    assert "void low_cvss() {}" not in codes
    assert "void null_cvss() {}" not in codes
    assert "void vuln() {}" in codes


def test_all_label_1(megavul_dir):
    samples = extract_megavul(megavul_dir)
    assert all(s.label == 1 for s in samples)


def test_single_file_input(tmp_path):
    p = tmp_path / "shard.parquet"
    pd.DataFrame([
        {"hash": "h1", "vulnerable_code": "void f(){}", "cwe_id": "CWE-89", "cvss3_base_score": 5.0},
    ]).to_parquet(p, index=False)
    samples = extract_megavul(p)
    assert len(samples) == 1


def test_empty_dir_raises(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="No .parquet files"):
        extract_megavul(d)
