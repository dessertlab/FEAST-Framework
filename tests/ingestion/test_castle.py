import json
import pytest
from ingestion.castle import extract_castle
from ingestion.schema import FunctionSample

FIXTURE = [
    {"code": "void vuln(){buf[100]=1;}",  "cwe": "CWE-121", "vulnerable": True},
    {"code": "void safe(){buf[0]=1;}",    "cwe": "",         "vulnerable": False},
    {"code": "void v2(){x=0;}",           "cwe": "NVD-CWE-noinfo", "vulnerable": True},
    {"code": "",                           "cwe": "CWE-89",  "vulnerable": True},
]


@pytest.fixture
def castle_path(tmp_path):
    p = tmp_path / "CASTLE-C250.json"
    p.write_text(json.dumps(FIXTURE))
    return p


def test_extracts_positive(castle_path):
    samples = extract_castle(castle_path)
    positives = [s for s in samples if s.label == 1]
    assert len(positives) == 1
    assert positives[0].cwes == ["CWE-121"]


def test_extracts_negative(castle_path):
    samples = extract_castle(castle_path)
    negatives = [s for s in samples if s.label == 0]
    assert len(negatives) == 1
    assert negatives[0].cwes == []


def test_drops_nvd_placeholder(castle_path):
    samples = extract_castle(castle_path)
    codes = [s.code for s in samples]
    assert "void v2(){x=0;}" not in codes


def test_branch_and_language(castle_path):
    samples = extract_castle(castle_path)
    assert all(s.branch == "synth" for s in samples)
    assert all(s.language == "C/C++" for s in samples)


def test_drops_empty_code(castle_path):
    samples = extract_castle(castle_path)
    assert all(s.code for s in samples)


def test_resolves_directory(tmp_path):
    p = tmp_path / "CASTLE-C250.json"
    p.write_text(json.dumps([
        {"code": "void v(){buf[100]=1;}", "cwe": 121, "vulnerable": True},
        {"code": "void s(){buf[0]=1;}",   "cwe": 0,   "vulnerable": False},
    ]))
    samples = extract_castle(tmp_path)
    assert len(samples) == 2
    pos = [s for s in samples if s.label == 1]
    assert pos[0].cwes == ["CWE-121"]


def test_directory_without_json_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .json files"):
        extract_castle(tmp_path)


def test_int_cwe_normalised(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps([
        {"code": "void f(){}", "cwe": 22, "vulnerable": True},
    ]))
    samples = extract_castle(p)
    assert samples[0].cwes == ["CWE-22"]


def test_top_level_dict_with_tests(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps({
        "tests": [
            {"code": "void f(){}", "cwe": 89, "vulnerable": True},
        ],
        "metadata": {"version": "1.0"},
    }))
    samples = extract_castle(p)
    assert samples[0].cwes == ["CWE-89"]


def test_vulnerable_with_unparseable_cwe_skipped(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps([
        {"code": "void f(){}", "cwe": "no-cwe-here", "vulnerable": True},
    ]))
    samples = extract_castle(p)
    assert samples == []
