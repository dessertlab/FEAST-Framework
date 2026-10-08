import json
import pytest
from ingestion.patcheval import extract_patcheval
from ingestion.schema import FunctionSample


@pytest.fixture
def patcheval_path(tmp_path):
    records = [
        {
            "cve_id": "CVE-2021-1",
            "cwe_info": {"CWE-22": "Path Traversal"},
            "vul_func": "def bad():\n    open(path)",
            "fix_func":  "def fixed():\n    open(safe_path)",
            "programming_language": "Python",
        },
        {
            "cve_id": "CVE-2021-2",
            "cwe_info": {"CWE-78": "Command Injection"},
            "vul_func": "void cmd(){exec(input);}",
            "fix_func":  "",
            "programming_language": "C",
        },
        {
            "cve_id": "CVE-2021-3",
            "cwe_info": {},
            "vul_func": "def no_cwe(): pass",
            "fix_func":  "",
            "programming_language": "Python",
        },
    ]
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(records))
    return p


def test_filters_python_only(patcheval_path):
    samples = extract_patcheval(patcheval_path)
    assert all(s.language == "Python" for s in samples)


def test_extracts_positive(patcheval_path):
    samples = extract_patcheval(patcheval_path)
    positives = [s for s in samples if s.label == 1]
    assert len(positives) == 1
    assert positives[0].cwes == ["CWE-22"]


def test_extracts_negative(patcheval_path):
    samples = extract_patcheval(patcheval_path)
    negatives = [s for s in samples if s.label == 0]
    assert len(negatives) == 1
    assert negatives[0].cwes == []


def test_drops_empty_cwe_info(patcheval_path):
    samples = extract_patcheval(patcheval_path)
    codes = [s.code for s in samples]
    assert "def no_cwe(): pass" not in codes


def test_branch_and_language(patcheval_path):
    samples = extract_patcheval(patcheval_path)
    assert all(s.branch == "real" for s in samples)


def test_extracts_from_directory_of_json_files(tmp_path):
    rec_a = {
        "cve_id": "CVE-A",
        "cwe_info": {"CWE-89": "SQLi"},
        "vul_func": "def a(): pass",
        "fix_func": "def a_fixed(): pass",
        "programming_language": "Python",
    }
    rec_b = [{
        "cve_id": "CVE-B",
        "cwe_info": "Has CWE-22 inside text",
        "vul_func": "def b(): pass",
        "fix_func": "",
        "programming_language": "Python",
    }]
    (tmp_path / "a.json").write_text(json.dumps(rec_a))
    (tmp_path / "b.json").write_text(json.dumps(rec_b))

    samples = extract_patcheval(tmp_path)
    cwes = sorted({c for s in samples for c in s.cwes})
    assert "CWE-22" in cwes and "CWE-89" in cwes


def test_string_cwe_info_fallback(tmp_path):
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps([{
        "cve_id": "CVE-X",
        "cwe_info": "Some text mentioning CWE-79 and CWE-89",
        "vul_func": "def x(): pass",
        "fix_func": "",
        "programming_language": "Python",
    }]))
    samples = extract_patcheval(p)
    positives = [s for s in samples if s.label == 1]
    assert positives and set(positives[0].cwes) == {"CWE-79", "CWE-89"}


def test_docker_verified_only_filter(tmp_path):
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps([
        {
            "cve_id": "CVE-1",
            "cwe_info": {"CWE-89": "SQLi"},
            "vul_func": "def a(): pass",
            "fix_func": "",
            "programming_language": "Python",
            "docker_verified": True,
        },
        {
            "cve_id": "CVE-2",
            "cwe_info": {"CWE-22": "Path"},
            "vul_func": "def b(): pass",
            "fix_func": "",
            "programming_language": "Python",
            "docker_verified": False,
        },
    ]))
    filtered = extract_patcheval(p, docker_verified_only=True)
    assert len(filtered) == 1 and filtered[0].cwes == ["CWE-89"]
    unfiltered = extract_patcheval(p, docker_verified_only=False)
    assert len(unfiltered) == 2


def test_string_cwe_info_dedup(tmp_path):
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps([{
        "cve_id": "CVE-D",
        "cwe_info": "CWE-79 CWE-79 CWE-79",
        "vul_func": "def d(): pass",
        "fix_func": "",
        "programming_language": "Python",
    }]))
    samples = extract_patcheval(p)
    assert samples[0].cwes == ["CWE-79"]
