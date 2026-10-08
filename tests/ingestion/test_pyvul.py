import json
import pytest
from pathlib import Path
from ingestion.pyvul import extract_pyvul
from ingestion.schema import FunctionSample


@pytest.fixture
def pyvul_dir(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    cwe_map = {
        "https://github.com/x/y/commit/abc": "CWE-089",
        "https://github.com/x/y/commit/nvd": "NVD-CWE-noinfo",
    }
    (dataset / "commits_cwe_map.json").write_text(json.dumps(cwe_map))

    records = [
        # Python, valid CWE -> 1 positive + 1 negative
        {"programming_language": "Python", "commit": "https://github.com/x/y/commit/abc",
         "code_before": "def bad():\n    pass", "code_after": "def safe():\n    pass"},
        # Python, NVD placeholder -> skipped
        {"programming_language": "Python", "commit": "https://github.com/x/y/commit/nvd",
         "code_before": "def nvd():\n    pass", "code_after": ""},
        # Java -> filtered out
        {"programming_language": "Java", "commit": "https://github.com/x/y/commit/abc",
         "code_before": "void java(){}", "code_after": ""},
    ]
    lines = "\n".join(json.dumps(r) for r in records)
    (dataset / "function_level_dataset.out").write_text(lines)

    return tmp_path


def test_extracts_positive(pyvul_dir):
    samples = extract_pyvul(pyvul_dir)
    positives = [s for s in samples if s.label == 1]
    assert len(positives) == 1
    assert positives[0].cwes == ["CWE-89"]


def test_normalises_cwe(pyvul_dir):
    samples = extract_pyvul(pyvul_dir)
    positives = [s for s in samples if s.label == 1]
    assert all(not c.startswith("CWE-0") for s in positives for c in s.cwes)


def test_extracts_negative(pyvul_dir):
    samples = extract_pyvul(pyvul_dir)
    negatives = [s for s in samples if s.label == 0]
    assert len(negatives) == 1


def test_drops_nvd_placeholder(pyvul_dir):
    samples = extract_pyvul(pyvul_dir)
    codes = [s.code for s in samples]
    assert "def nvd():\n    pass" not in codes


def test_branch_and_language(pyvul_dir):
    samples = extract_pyvul(pyvul_dir)
    assert all(s.branch == "real" for s in samples)
    assert all(s.language == "Python" for s in samples)


def test_missing_dataset_file_raises(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "commits_cwe_map.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="function_level_dataset.out"):
        extract_pyvul(tmp_path)


def test_missing_cwe_map_raises(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "function_level_dataset.out").write_text("")
    with pytest.raises(FileNotFoundError, match="commits_cwe_map.json"):
        extract_pyvul(tmp_path)


def test_skips_blank_lines_and_unmapped_commits(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "commits_cwe_map.json").write_text(json.dumps({"u": "CWE-89"}))
    (dataset / "function_level_dataset.out").write_text(
        '\n'
        '{"programming_language":"Python","commit":"u","code_before":"def b(): pass","code_after":""}\n'
        '\n'
        '{"programming_language":"Python","commit":"unmapped","code_before":"def x(): pass","code_after":""}\n'
    )
    samples = extract_pyvul(tmp_path)
    assert len(samples) == 1 and samples[0].cwes == ["CWE-89"]
