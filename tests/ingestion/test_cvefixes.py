import pandas as pd
import pytest
from pathlib import Path
from ingestion.cvefixes import extract_cvefixes
from ingestion.schema import FunctionSample


def _write_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    """Write rows as a single Parquet file in a directory."""
    d = tmp_path / "cvefixes"
    d.mkdir()
    pd.DataFrame(rows).to_parquet(d / "train-00000.parquet", index=False)
    return d


@pytest.fixture
def cvefixes_dir(tmp_path):
    rows = [
        # hash_a -- single C function, valid CWE -> positive
        {"hash": "hash_a", "language": "C",   "vulnerable_code": "void vuln() {}",    "cwe_id": "CWE-119"},
        # hash_b -- two C functions -> filtered (multi-function commit)
        {"hash": "hash_b", "language": "C",   "vulnerable_code": "void multi1() {}",  "cwe_id": "CWE-787"},
        {"hash": "hash_b", "language": "C",   "vulnerable_code": "void multi2() {}",  "cwe_id": "CWE-787"},
        # hash_c -- no CWE -> filtered
        {"hash": "hash_c", "language": "C",   "vulnerable_code": "void no_cwe() {}",  "cwe_id": None},
        # hash_d -- C++ file, valid -> positive
        {"hash": "hash_d", "language": "C++", "vulnerable_code": "void cpp_vuln() {}","cwe_id": "CWE-476"},
        # hash_e -- Java, valid CWE -> filtered (non-C/C++ language)
        {"hash": "hash_e", "language": "Java","vulnerable_code": "void java() {}",    "cwe_id": "CWE-89"},
        # hash_f -- NVD placeholder CWE -> filtered
        {"hash": "hash_f", "language": "C",   "vulnerable_code": "void nvd() {}",     "cwe_id": "NVD-CWE-noinfo"},
    ]
    return _write_parquet(tmp_path, rows)


def test_extracts_single_function_c_positive(cvefixes_dir):
    samples = extract_cvefixes(cvefixes_dir)
    codes = [s.code for s in samples]
    assert "void vuln() {}" in codes


def test_filters_multi_function_commit(cvefixes_dir):
    samples = extract_cvefixes(cvefixes_dir)
    codes = [s.code for s in samples]
    assert "void multi1() {}" not in codes
    assert "void multi2() {}" not in codes


def test_filters_missing_cwe(cvefixes_dir):
    samples = extract_cvefixes(cvefixes_dir)
    codes = [s.code for s in samples]
    assert "void no_cwe() {}" not in codes


def test_accepts_cpp_files(cvefixes_dir):
    samples = extract_cvefixes(cvefixes_dir)
    codes = [s.code for s in samples]
    assert "void cpp_vuln() {}" in codes


def test_filters_non_c_files(cvefixes_dir):
    samples = extract_cvefixes(cvefixes_dir)
    codes = [s.code for s in samples]
    assert "void java() {}" not in codes


def test_filters_nvd_placeholder_cwe(cvefixes_dir):
    samples = extract_cvefixes(cvefixes_dir)
    codes = [s.code for s in samples]
    assert "void nvd() {}" not in codes


def test_all_positives_label_1(cvefixes_dir):
    samples = extract_cvefixes(cvefixes_dir)
    assert all(s.label == 1 for s in samples)


def test_cwe_assigned(cvefixes_dir):
    samples = extract_cvefixes(cvefixes_dir)
    vuln = next(s for s in samples if s.code == "void vuln() {}")
    assert vuln.cwes == ["CWE-119"]


def test_empty_directory_raises(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="No .parquet files"):
        extract_cvefixes(d)


def test_single_file_input(tmp_path):
    p = tmp_path / "shard.parquet"
    pd.DataFrame([
        {"hash": "h", "language": "C", "vulnerable_code": "void v(){}", "cwe_id": "CWE-89"},
    ]).to_parquet(p, index=False)
    samples = extract_cvefixes(p, language="C")
    assert len(samples) == 1


def test_python_language(tmp_path):
    p = tmp_path / "shard.parquet"
    pd.DataFrame([
        {"hash": "h1", "language": "Python", "vulnerable_code": "def v(): pass", "cwe_id": "CWE-89"},
        {"hash": "h2", "language": "C",      "vulnerable_code": "void f(){}",     "cwe_id": "CWE-89"},
    ]).to_parquet(p, index=False)
    samples = extract_cvefixes(p, language="Python")
    assert len(samples) == 1
    assert samples[0].language == "Python"
