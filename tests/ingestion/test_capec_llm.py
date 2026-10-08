import json
import pytest
from ingestion.capec_llm import extract_capec_llm
from ingestion.schema import FunctionSample


@pytest.fixture
def capec_java_path(tmp_path):
    records = [
        {
            "capec_id": "CAPEC-1",
            "code_snippet": "public class Vuln { public void exec(String input) { Runtime.getRuntime().exec(input); } }",
            "description": "This maps to CWE-78 command injection.",
        },
        {
            "capec_id": "CAPEC-2",
            "code_snippet": "def run(cmd): import os; os.system(cmd)",
            "description": "Maps to CWE-78.",
        },
        {
            "capec_id": "CAPEC-3",
            "code_snippet": "public class X { void x(){} }",
            "description": "No CWE mentioned here.",
        },
    ]
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(records))
    return tmp_path


def test_extracts_java_only(capec_java_path):
    samples = extract_capec_llm(capec_java_path, language="Java")
    assert all(s.language == "Java" for s in samples)
    codes = [s.code for s in samples]
    assert not any("def run" in c for c in codes)


def test_drops_no_cwe(capec_java_path):
    samples = extract_capec_llm(capec_java_path, language="Java")
    # CAPEC-3 has no CWE in description -> dropped
    codes = [s.code for s in samples]
    assert not any("void x(){}" in c for c in codes)


def test_cwe_from_description(capec_java_path):
    samples = extract_capec_llm(capec_java_path, language="Java")
    assert all("CWE-78" in s.cwes for s in samples)


def test_branch_and_language(capec_java_path):
    samples = extract_capec_llm(capec_java_path, language="Java")
    assert all(s.branch == "ai" for s in samples)


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="CAPEC_LLM path not found"):
        extract_capec_llm(tmp_path / "missing")


def test_python_filter(capec_java_path):
    samples = extract_capec_llm(capec_java_path, language="Python")
    assert all(s.language == "Python" for s in samples)
    codes = [s.code for s in samples]
    assert any("def run" in c for c in codes)


def test_single_file_input(tmp_path):
    p = tmp_path / "single.json"
    p.write_text(json.dumps([
        {
            "capec_id": "CAPEC-1",
            "code_snippet": "public class V { void v(String s){Runtime.getRuntime().exec(s);} }",
            "description": "CWE-78",
        },
    ]))
    samples = extract_capec_llm(p, language="Java")
    assert len(samples) == 1 and samples[0].cwes == ["CWE-78"]


def test_single_file_with_top_level_dict(tmp_path):
    p = tmp_path / "single.json"
    p.write_text(json.dumps({
        "capec_id": "CAPEC-1",
        "code_snippet": "public class V { @Override public void run(){} }",
        "description": "CWE-79 risk",
    }))
    samples = extract_capec_llm(p, language="Java")
    assert len(samples) == 1 and samples[0].cwes == ["CWE-79"]


def test_directory_with_top_level_dict(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({
        "capec_id": "CAPEC-1",
        "code_snippet": "public class V { void f(){} }",
        "description": "CWE-89",
    }))
    samples = extract_capec_llm(tmp_path, language="Java")
    assert samples and samples[0].cwes == ["CWE-89"]


def test_empty_code_skipped(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps([
        {"capec_id": "CAPEC-99", "code_snippet": "", "description": "CWE-78"},
    ]))
    samples = extract_capec_llm(p, language="Java")
    assert samples == []


def test_unrecognised_language_dropped(tmp_path):
    # Snippet that doesn't match Java or Python heuristics -> filtered out.
    p = tmp_path / "data.json"
    p.write_text(json.dumps([
        {"capec_id": "CAPEC-1", "code_snippet": "<html>?</html>", "description": "CWE-79"},
    ]))
    java_samples = extract_capec_llm(p, language="Java")
    py_samples = extract_capec_llm(p, language="Python")
    assert java_samples == [] and py_samples == []
