import json
import pytest
from ingestion.security_eval import extract_security_eval
from ingestion.schema import FunctionSample


@pytest.fixture
def security_eval_jsonl(tmp_path):
    lines = [
        {"ID": "CWE-79-1",  "Insecure_code": "print(user_input)",   "CWE": "CWE-79"},
        {"ID": "CWE-22-1",  "Insecure_code": "open(path)",          "CWE": "CWE-22"},
        {"ID": "NO_CWE",    "Insecure_code": "x = 1",               "CWE": ""},
        {"ID": "EMPTY",     "Insecure_code": "",                     "CWE": "CWE-89"},
    ]
    p = tmp_path / "dataset.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in lines))
    return p


@pytest.fixture
def security_eval_dirs(tmp_path):
    cwe79 = tmp_path / "CWE-79"
    cwe79.mkdir()
    (cwe79 / "xss.py").write_text("print(user_input)")
    return tmp_path


def test_jsonl_extracts_positives(security_eval_jsonl):
    samples = extract_security_eval(security_eval_jsonl)
    assert len(samples) == 2
    assert all(s.label == 1 for s in samples)


def test_jsonl_drops_empty_cwe(security_eval_jsonl):
    samples = extract_security_eval(security_eval_jsonl)
    codes = [s.code for s in samples]
    assert "x = 1" not in codes


def test_jsonl_drops_empty_code(security_eval_jsonl):
    samples = extract_security_eval(security_eval_jsonl)
    assert all(s.code for s in samples)


def test_dir_layout(security_eval_dirs):
    samples = extract_security_eval(security_eval_dirs)
    assert len(samples) == 1
    assert samples[0].cwes == ["CWE-79"]


def test_branch_and_language(security_eval_jsonl):
    samples = extract_security_eval(security_eval_jsonl)
    assert all(s.branch == "ai" for s in samples)
    assert all(s.language == "Python" for s in samples)


def test_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "dataset.jsonl"
    p.write_text(
        '\n'
        + json.dumps({"ID": "CWE-89-1", "Insecure_code": "open()", "CWE": "CWE-89"}) + '\n'
        + '\n'
    )
    samples = extract_security_eval(p)
    assert len(samples) == 1


def test_jsonl_id_field_only(tmp_path):
    # CWE field absent; ID field carries the CWE.
    p = tmp_path / "dataset.jsonl"
    p.write_text(json.dumps({"ID": "CWE-79-1", "Insecure_code": "x"}))
    samples = extract_security_eval(p)
    assert samples and samples[0].cwes == ["CWE-79"]


def test_cwe_from_id_field_no_dash(tmp_path):
    # IDs like 'CWE79' (no dash) are parsed via the secondary regex.
    p = tmp_path / "dataset.jsonl"
    p.write_text(json.dumps({"ID": "CWE79", "Insecure_code": "x = 1"}))
    samples = extract_security_eval(p)
    assert samples and samples[0].cwes == ["CWE-79"]


def test_dir_layout_skips_non_cwe_dirs(tmp_path):
    # Only directories matching CWE-NNN are considered. Top-level files
    # (non-directories) and non-matching directory names are both skipped.
    (tmp_path / "README.md").write_text("info")  # top-level file -> not a dir
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "x.py").write_text("def x(): pass")
    cwe89 = tmp_path / "CWE-89"
    cwe89.mkdir()
    (cwe89 / "v.py").write_text("def v(): pass")
    samples = extract_security_eval(tmp_path)
    assert len(samples) == 1 and samples[0].cwes == ["CWE-89"]


def test_dir_layout_skips_empty_files(tmp_path):
    cwe = tmp_path / "CWE-22"
    cwe.mkdir()
    (cwe / "empty.py").write_text("")
    (cwe / "ok.py").write_text("def f(): pass")
    samples = extract_security_eval(tmp_path)
    assert len(samples) == 1 and samples[0].code == "def f(): pass"


def test_jsonl_blank_id_field_yields_no_cwe(tmp_path):
    p = tmp_path / "dataset.jsonl"
    p.write_text(json.dumps({"ID": "", "Insecure_code": "x", "CWE": ""}))
    samples = extract_security_eval(p)
    assert samples == []
