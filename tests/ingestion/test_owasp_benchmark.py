import pytest
from ingestion.owasp_benchmark import extract_owasp_benchmark
from ingestion.schema import FunctionSample


@pytest.fixture
def owasp_java_dir(tmp_path):
    csv_content = (
        "# test name,category,real vulnerability,cwe\n"
        "BenchmarkTest00001,sqli,1,CWE-89\n"
        "BenchmarkTest00002,xss,0,CWE-79\n"
        "BenchmarkTest00003,cmdi,1,CWE-78\n"
    )
    (tmp_path / "expectedresults-1.2.csv").write_text(csv_content)
    src = tmp_path / "src" / "main" / "java"
    src.mkdir(parents=True)
    (src / "BenchmarkTest00001.java").write_text("public class BenchmarkTest00001 { }")
    (src / "BenchmarkTest00002.java").write_text("public class BenchmarkTest00002 { }")
    (src / "BenchmarkTest00003.java").write_text("public class BenchmarkTest00003 { }")
    return tmp_path


def test_extracts_positives(owasp_java_dir):
    samples = extract_owasp_benchmark(owasp_java_dir, language="Java")
    positives = [s for s in samples if s.label == 1]
    assert len(positives) == 2


def test_extracts_negatives(owasp_java_dir):
    samples = extract_owasp_benchmark(owasp_java_dir, language="Java")
    negatives = [s for s in samples if s.label == 0]
    assert len(negatives) == 1
    assert negatives[0].cwes == []


def test_cwe_from_csv(owasp_java_dir):
    samples = extract_owasp_benchmark(owasp_java_dir, language="Java")
    pos_sql = [s for s in samples if s.label == 1 and "CWE-89" in s.cwes]
    assert len(pos_sql) == 1


def test_branch_and_language(owasp_java_dir):
    samples = extract_owasp_benchmark(owasp_java_dir, language="Java")
    assert all(s.branch == "synth" for s in samples)
    assert all(s.language == "Java" for s in samples)


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No expectedresults"):
        extract_owasp_benchmark(tmp_path, language="Java")


def test_python_layout(tmp_path):
    csv_content = (
        "# test name,category,real vulnerability,cwe\n"
        "BenchmarkTest00001,sqli,1,89\n"
    )
    (tmp_path / "expectedresults-0.1.csv").write_text(csv_content)
    testcode = tmp_path / "testcode"
    testcode.mkdir()
    (testcode / "BenchmarkTest00001.py").write_text("def vuln(): pass")
    samples = extract_owasp_benchmark(tmp_path, language="Python")
    assert len(samples) == 1
    assert samples[0].language == "Python"
    assert samples[0].cwes == ["CWE-89"]


def test_cwe_falls_back_to_category_when_csv_cwe_blank(tmp_path):
    # When cwe column is blank, the extractor maps the category name to a CWE.
    (tmp_path / "expectedresults-1.2.csv").write_text(
        "# test name,category,real vulnerability,cwe\n"
        "BenchmarkTest09999,xss,1,\n"
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "BenchmarkTest09999.java").write_text("class X {}")
    samples = extract_owasp_benchmark(tmp_path, language="Java")
    assert samples and samples[0].cwes == ["CWE-79"]


def test_skips_csv_blank_and_comment_lines(tmp_path):
    (tmp_path / "expectedresults-1.2.csv").write_text(
        "# test name,category,real vulnerability,cwe\n"
        "\n"
        "# This is just a comment line\n"
        "BenchmarkTest00001,sqli,1,89\n"
        "\n"
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "BenchmarkTest00001.java").write_text("class A {}")
    samples = extract_owasp_benchmark(tmp_path, language="Java")
    assert len(samples) == 1


def test_empty_test_name_row_is_skipped(tmp_path):
    (tmp_path / "expectedresults-1.2.csv").write_text(
        "# test name,category,real vulnerability,cwe\n"
        ",sqli,1,89\n"
        "BenchmarkTest00001,sqli,1,89\n"
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "BenchmarkTest00001.java").write_text("class A {}")
    samples = extract_owasp_benchmark(tmp_path, language="Java")
    assert len(samples) == 1


def test_files_without_csv_entry_are_skipped(tmp_path):
    (tmp_path / "expectedresults-1.2.csv").write_text(
        "# test name,category,real vulnerability,cwe\n"
        "BenchmarkTest00001,sqli,1,89\n"
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "BenchmarkTest00001.java").write_text("class A {}")
    (src / "Untracked.java").write_text("class U {}")
    samples = extract_owasp_benchmark(tmp_path, language="Java")
    assert len(samples) == 1


def test_skips_empty_source_files(tmp_path):
    (tmp_path / "expectedresults-1.2.csv").write_text(
        "# test name,category,real vulnerability,cwe\n"
        "BenchmarkTest00001,sqli,1,89\n"
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "BenchmarkTest00001.java").write_text("")
    samples = extract_owasp_benchmark(tmp_path, language="Java")
    assert samples == []


def test_vulnerable_without_cwe_or_category_is_skipped(tmp_path):
    # vulnerable=1 but no CWE and an unknown category -> skipped
    (tmp_path / "expectedresults-1.2.csv").write_text(
        "# test name,category,real vulnerability,cwe\n"
        "BenchmarkTest00001,bogusunknowncategory,1,\n"
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "BenchmarkTest00001.java").write_text("class A {}")
    samples = extract_owasp_benchmark(tmp_path, language="Java")
    assert samples == []


def test_cwe_normalisation_handles_full_label_in_csv(tmp_path):
    (tmp_path / "expectedresults-1.2.csv").write_text(
        "# test name,category,real vulnerability,cwe\n"
        "BenchmarkTest00001,sqli,1,CWE-89\n"
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "BenchmarkTest00001.java").write_text("class A {}")
    samples = extract_owasp_benchmark(tmp_path, language="Java")
    assert samples[0].cwes == ["CWE-89"]


def test_cwe_normalisation_returns_empty_for_unparseable(tmp_path):
    # cwe column has gibberish -> normalise_cwe returns "";
    # category fallback ("xss" partial match) recovers CWE-79.
    (tmp_path / "expectedresults-1.2.csv").write_text(
        "# test name,category,real vulnerability,cwe\n"
        "BenchmarkTest00001,xss,1,gibberish\n"
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "BenchmarkTest00001.java").write_text("class A {}")
    samples = extract_owasp_benchmark(tmp_path, language="Java")
    # normalise_cwe returns "" -> falsy -> falls through to category fallback at runtime
    # (note: extractor only calls _cwe_from_category when cwe column is blank,
    # so a gibberish value here will result in cwe=""; vulnerable=1 with empty
    # cwe is filtered out).
    assert samples == []


def test_category_fuzzy_match_when_not_exact(tmp_path):
    # Category 'sqlinjection_compound' isn't an exact key, but 'sqlinjection'
    # is a substring -> _cwe_from_category resolves to CWE-89.
    (tmp_path / "expectedresults-1.2.csv").write_text(
        "# test name,category,real vulnerability,cwe\n"
        "BenchmarkTest00001,sqlinjection_compound,1,\n"
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "BenchmarkTest00001.java").write_text("class A {}")
    samples = extract_owasp_benchmark(tmp_path, language="Java")
    assert samples and samples[0].cwes == ["CWE-89"]
