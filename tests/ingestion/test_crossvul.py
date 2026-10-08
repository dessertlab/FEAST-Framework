import zipfile
import pytest
from pathlib import Path
from ingestion.crossvul import extract_crossvul
from ingestion.schema import FunctionSample


@pytest.fixture
def crossvul_path(tmp_path):
    p = tmp_path / "crossvul.zip"
    with zipfile.ZipFile(p, 'w') as zf:
        zf.writestr("dataset_final_sorted/CWE-119/c/bad_0001_0",   "void vuln() { buf[idx]; }")
        zf.writestr("dataset_final_sorted/CWE-119/c/good_0001_0",  "void safe() { return 0; }")
        zf.writestr("dataset_final_sorted/CWE-89/java/bad_0001_0",  "void sqli() {}")
        zf.writestr("dataset_final_sorted/CWE-89/java/good_0001_0", "void safej() {}")
        zf.writestr("dataset_final_sorted/CWE-22/py/bad_0001_0",   "x = open(path)")
        zf.writestr("dataset_final_sorted/CWE-22/py/good_0001_0",  "x = open(safe_path)")
        zf.writestr("dataset_final_sorted/NOTCWE/c/bad_0001_0",    "void skip() {}")
    return p


def test_returns_function_samples(crossvul_path):
    samples = extract_crossvul(crossvul_path, language="C/C++")
    assert all(isinstance(s, FunctionSample) for s in samples)


def test_filters_by_language(crossvul_path):
    c_samples      = extract_crossvul(crossvul_path, language="C/C++")
    java_samples   = extract_crossvul(crossvul_path, language="Java")
    python_samples = extract_crossvul(crossvul_path, language="Python")
    assert len(c_samples)      == 2
    assert len(java_samples)   == 2
    assert len(python_samples) == 2


def test_labels(crossvul_path):
    samples = extract_crossvul(crossvul_path, language="C/C++")
    assert {s.label for s in samples} == {0, 1}


def test_vulnerable_has_cwe_safe_has_none(crossvul_path):
    samples = extract_crossvul(crossvul_path, language="C/C++")
    for s in samples:
        if s.label == 1:
            assert s.cwes == ["CWE-119"]
        else:
            assert s.cwes == []


def test_skips_invalid_cwe_dir(crossvul_path):
    samples = extract_crossvul(crossvul_path, language="C/C++")
    assert len(samples) == 2  # NOTCWE dir skipped


def test_branch_and_language_fields(crossvul_path):
    samples = extract_crossvul(crossvul_path, language="C/C++")
    assert all(s.branch == "real" for s in samples)
    assert all(s.language == "C/C++" for s in samples)


def test_missing_zip_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="CrossVul ZIP not found"):
        extract_crossvul(tmp_path / "nope.zip", language="C/C++")


def test_skips_misc_zip_entries(tmp_path):
    p = tmp_path / "crossvul.zip"
    with zipfile.ZipFile(p, 'w') as zf:
        # Top-level README -> too-short path
        zf.writestr("README.md", "info")
        # Directory entry
        zf.writestr("dataset_final_sorted/CWE-79/c/", "")
        # Filename neither bad_ nor good_
        zf.writestr("dataset_final_sorted/CWE-79/c/random_file", "void r(){}")
        # Empty content
        zf.writestr("dataset_final_sorted/CWE-79/c/bad_0001_0", "")
        # Valid entry
        zf.writestr("dataset_final_sorted/CWE-79/c/bad_0001_1", "void v(){}")
    samples = extract_crossvul(p, language="C/C++")
    assert len(samples) == 1
    assert samples[0].cwes == ["CWE-79"]
