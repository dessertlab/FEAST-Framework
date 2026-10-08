import pytest
from pathlib import Path
from ingestion.llmseceval import extract_llmseceval
from ingestion.schema import FunctionSample


@pytest.fixture
def llmseceval_dir(tmp_path):
    # Vulnerable C/C++ via zenodo gen_scenario structure
    gen_c121 = tmp_path / "zenodo" / "data" / "cwe-121" / "scenario1" / "gen_scenario"
    gen_c121.mkdir(parents=True)
    (gen_c121 / "test_copilot_1.c").write_text("void vuln(){buf[200]=1;}")

    gen_c89 = tmp_path / "zenodo" / "data" / "cwe-89" / "scenario1" / "gen_scenario"
    gen_c89.mkdir(parents=True)
    (gen_c89 / "test_copilot_1.c").write_text("void sql(){query(input);}")

    # Python file in C gen_scenario -- ignored for C/C++
    (gen_c121 / "script.py").write_text("import os")

    # Safe Python via CWE-NNN/Secure structure
    secure_dir = tmp_path / "CWE-121" / "Secure"
    secure_dir.mkdir(parents=True)
    (secure_dir / "safe.py").write_text("def safe(): pass")

    return tmp_path


def test_extracts_positives(llmseceval_dir):
    samples = extract_llmseceval(llmseceval_dir, language="C/C++")
    positives = [s for s in samples if s.label == 1]
    assert len(positives) == 2


def test_no_negatives_for_c(llmseceval_dir):
    # extractor only collects safe samples for Python, not C/C++
    samples = extract_llmseceval(llmseceval_dir, language="C/C++")
    assert all(s.label == 1 for s in samples)


def test_extracts_python_negatives(llmseceval_dir):
    samples = extract_llmseceval(llmseceval_dir, language="Python")
    negatives = [s for s in samples if s.label == 0]
    assert len(negatives) == 1
    assert negatives[0].cwes == []


def test_cwe_from_dir(llmseceval_dir):
    samples = extract_llmseceval(llmseceval_dir, language="C/C++")
    pos_121 = [s for s in samples if s.label == 1 and s.cwes == ["CWE-121"]]
    assert len(pos_121) == 1


def test_branch_and_language(llmseceval_dir):
    samples = extract_llmseceval(llmseceval_dir, language="C/C++")
    assert all(s.branch == "ai" for s in samples)
    assert all(s.language == "C/C++" for s in samples)


def test_language_filter_excludes_wrong_ext(llmseceval_dir):
    samples = extract_llmseceval(llmseceval_dir, language="C/C++")
    codes = [s.code for s in samples]
    assert "import os" not in codes


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="LLMSecEval directory not found"):
        extract_llmseceval(tmp_path / "missing")


def test_unsupported_language_raises(tmp_path):
    with pytest.raises(ValueError, match="Unsupported language"):
        extract_llmseceval(tmp_path, language="Rust")


def test_skips_empty_zenodo_files(tmp_path):
    gen = tmp_path / "zenodo" / "experiments_dow" / "cwe-121" / "scen" / "gen_scenario"
    gen.mkdir(parents=True)
    (gen / "empty_copilot_1.c").write_text("")
    (gen / "ok_copilot_2.c").write_text("void ok(){}")
    samples = extract_llmseceval(tmp_path, language="C/C++")
    assert len(samples) == 1


def test_skips_zenodo_files_without_cwe_in_path(tmp_path):
    # gen_scenario folder but no cwe-NNN directory in the path
    gen = tmp_path / "zenodo" / "no_cwe_here" / "scen" / "gen_scenario"
    gen.mkdir(parents=True)
    (gen / "x_copilot_1.c").write_text("void x(){}")
    samples = extract_llmseceval(tmp_path, language="C/C++")
    assert samples == []


def test_python_secure_dir_missing_is_skipped(tmp_path):
    # CWE-NNN dir exists but no Secure subdirectory.
    cwe_dir = tmp_path / "CWE-22"
    cwe_dir.mkdir()
    (cwe_dir / "loose_file.py").write_text("def x(): pass")
    samples = extract_llmseceval(tmp_path, language="Python")
    assert samples == []


def test_python_secure_dir_skips_non_py_and_empty(tmp_path):
    secure = tmp_path / "CWE-89" / "Secure"
    secure.mkdir(parents=True)
    (secure / "ok.py").write_text("def ok(): pass")
    (secure / "ignore.txt").write_text("not python")
    (secure / "empty.py").write_text("")
    samples = extract_llmseceval(tmp_path, language="Python")
    assert len(samples) == 1
    assert samples[0].code == "def ok(): pass"
