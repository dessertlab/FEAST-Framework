import io
import zipfile
import pytest
from ingestion.juliet import extract_juliet
from ingestion.schema import FunctionSample


def _make_zip(tmp_path, files: dict[str, str]) -> "Path":
    """Create a ZIP at tmp_path/juliet.zip with the given filename->content mapping."""
    from pathlib import Path
    zpath = tmp_path / "juliet.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return zpath


@pytest.fixture
def juliet_c_path(tmp_path):
    return _make_zip(tmp_path, {
        "CWE121_Stack_Based_Buffer_Overflow/s01/CWE121_bad.c":    "void bad(){overflow();}",
        "CWE121_Stack_Based_Buffer_Overflow/s01/CWE121_good01.c": "void good(){no_overflow();}",
        "CWE89_SQL_Injection/s01/CWE89_bad.c":                    "void bad(){sql();}",
        "no_cwe_dir/CWE_free.c":                                  "void x(){}",
    })


@pytest.fixture
def juliet_java_path(tmp_path):
    return _make_zip(tmp_path, {
        "CWE89_SQL_Injection/s01/CWE89_bad.java":    "public class CWE89_bad { }",
        "CWE89_SQL_Injection/s01/CWE89_good01.java": "public class CWE89_good01 { }",
    })


def test_c_extracts_bad_as_positive(juliet_c_path):
    samples = extract_juliet(juliet_c_path, language="C/C++")
    positives = [s for s in samples if s.label == 1]
    assert len(positives) == 2
    assert all(s.cwes for s in positives)


def test_c_extracts_good_as_negative(juliet_c_path):
    samples = extract_juliet(juliet_c_path, language="C/C++")
    negatives = [s for s in samples if s.label == 0]
    assert len(negatives) == 1
    assert all(s.cwes == [] for s in negatives)


def test_c_cwe_from_dir(juliet_c_path):
    samples = extract_juliet(juliet_c_path, language="C/C++")
    positives = [s for s in samples if s.label == 1 and s.code == "void bad(){overflow();}"]
    assert positives[0].cwes == ["CWE-121"]


def test_java_branch_and_language(juliet_java_path):
    samples = extract_juliet(juliet_java_path, language="Java")
    assert all(s.branch == "synth" for s in samples)
    assert all(s.language == "Java" for s in samples)


def test_no_cwe_dir_skipped(juliet_c_path):
    samples = extract_juliet(juliet_c_path, language="C/C++")
    codes = [s.code for s in samples]
    assert "void x(){}" not in codes


def test_missing_archive_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Juliet archive not found"):
        extract_juliet(tmp_path / "missing.zip", language="C/C++")


def test_skips_directories_and_unrelated_files_c(tmp_path):
    zpath = _make_zip(tmp_path, {
        "CWE121_Stack_Based_Buffer_Overflow/s01/":             "",   # directory entry
        "CWE121_Stack_Based_Buffer_Overflow/s01/notes.txt":    "txt",
        "CWE121_Stack_Based_Buffer_Overflow/s01/CWE121_bad.c": "void b(){}",
        "CWE121_Stack_Based_Buffer_Overflow/s01/empty_bad.c":  "",   # empty payload
        "CWE121_Stack_Based_Buffer_Overflow/s01/no_keyword.c": "void n(){}",
    })
    samples = extract_juliet(zpath, language="C/C++")
    codes = [s.code for s in samples]
    assert codes == ["void b(){}"]


def test_skips_directories_and_unrelated_files_java(tmp_path):
    zpath = _make_zip(tmp_path, {
        "CWE89_SQL_Injection/notes.txt":          "info",
        "CWE89_SQL_Injection/s01/":               "",
        "CWE89_SQL_Injection/s01/CWE89_bad.java": "class X {}",
        "CWE89_SQL_Injection/s01/empty_bad.java": "",
        "CWE89_SQL_Injection/s01/floating.java":  "class F {}",   # neither _bad nor _good
        # No CWE token anywhere in the path -> _cwe_from_path returns None.
        "non_juliet/util/some_bad.java":          "class Y {}",
    })
    samples = extract_juliet(zpath, language="Java")
    codes = [s.code for s in samples]
    assert codes == ["class X {}"]
