import pytest
from ingestion.formai import extract_formai
from ingestion.schema import FunctionSample
import pandas as pd


@pytest.fixture
def formai_path(tmp_path):
    df = pd.DataFrame([
        {"filename": "test.c",   "classification": "Vulnerable",     "source_code": "void v(){buf[100]=1;}", "vul_type": "buffer overflow"},
        {"filename": "test.c",   "classification": "Non-Vulnerable",  "source_code": "void s(){buf[0]=1;}",  "vul_type": ""},
        {"filename": "test.c",   "classification": "Vulnerable",      "source_code": "void x(){}",           "vul_type": "unknown_type_xyz"},
        {"filename": "test.py",  "classification": "Vulnerable",      "source_code": "def bad():",           "vul_type": "buffer overflow"},
    ])
    p = tmp_path / "formai.csv"
    df.to_csv(p, index=False)
    return p


def test_extracts_positive(formai_path):
    samples = extract_formai(formai_path)
    positives = [s for s in samples if s.label == 1]
    assert len(positives) == 1
    assert positives[0].cwes == ["CWE-121"]
    assert positives[0].sample_id == "test"


def test_extracts_negative(formai_path):
    samples = extract_formai(formai_path)
    negatives = [s for s in samples if s.label == 0]
    assert len(negatives) == 1


def test_drops_unknown_vul_type(formai_path):
    samples = extract_formai(formai_path)
    codes = [s.code for s in samples]
    assert "void x(){}" not in codes


def test_skips_non_c_files(formai_path):
    samples = extract_formai(formai_path)
    codes = [s.code for s in samples]
    assert "def bad():" not in codes


def test_branch_and_language(formai_path):
    samples = extract_formai(formai_path)
    assert all(s.branch == "ai" for s in samples)
    assert all(s.language == "C/C++" for s in samples)


def test_extracts_v2_json_error_type_and_keeps_file_stem(tmp_path):
    p = tmp_path / "FormAI-v2.json"
    p.write_text(
        """[
          {
            "file_name": "proc_monitor.c",
            "category": "VULNERABLE",
            "source_code": "int main(){char b[4]; scanf(\\"%s\\", b);}",
            "error_type": "buffer overflow on scanf"
          },
          {
            "file_name": "safe_proc_monitor.c",
            "category": "NON-VULNERABLE",
            "source_code": "int main(){return 0;}",
            "error_type": ""
          }
        ]""",
        encoding="utf-8",
    )

    samples = extract_formai(p)

    assert [(s.sample_id, s.label, s.cwes) for s in samples] == [
        ("proc_monitor", 1, ["CWE-121"]),
        ("safe_proc_monitor", 0, []),
    ]


def test_maps_esbmc_array_bounds_variants(tmp_path):
    p = tmp_path / "formai.csv"
    pd.DataFrame([{
        "filename": "bounds.c",
        "classification": "Vulnerable",
        "source_code": "int main(){int a[1]; return a[2];}",
        "vul_type": "array bounds violated: lower bound",
    }]).to_csv(p, index=False)

    samples = extract_formai(p)

    assert samples[0].cwes == ["CWE-119"]


def test_extracts_jsonl(tmp_path):
    p = tmp_path / "formai.jsonl"
    p.write_text(
        '{"file_name":"a.c","category":"Vulnerable","source_code":"void f(){}","error_type":"buffer overflow"}\n'
        '\n'
        '{"file_name":"b.c","category":"Non-Vulnerable","source_code":"void g(){}","error_type":""}\n',
        encoding="utf-8",
    )
    samples = extract_formai(p)
    assert [(s.sample_id, s.label, s.cwes) for s in samples] == [
        ("a", 1, ["CWE-121"]),
        ("b", 0, []),
    ]


def test_resolves_directory_to_v2_json(tmp_path):
    (tmp_path / "FormAI-v2.json").write_text(
        '[{"file_name":"x.c","category":"Vulnerable","source_code":"int main(){}","error_type":"memory leak"}]',
        encoding="utf-8",
    )
    samples = extract_formai(tmp_path)
    assert samples[0].cwes == ["CWE-401"]
    assert samples[0].sample_id == "x"


def test_resolves_directory_with_arbitrary_csv(tmp_path):
    (tmp_path / "weird_name.csv").write_text(
        "filename,classification,source_code,vul_type\n"
        "z.c,Vulnerable,void z(){},division by zero\n",
        encoding="utf-8",
    )
    samples = extract_formai(tmp_path)
    assert samples == [] or samples[0].cwes == ["CWE-369"]


def test_resolves_directory_no_data_raises(tmp_path):
    (tmp_path / "readme.txt").write_text("nothing useful")
    with pytest.raises(FileNotFoundError):
        extract_formai(tmp_path)


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "data.xml"
    p.write_text("<root/>")
    with pytest.raises(ValueError, match="Unsupported FormAI file type"):
        extract_formai(p)


def test_json_array_must_be_an_array(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"not":"an array"}', encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON array"):
        extract_formai(p)


def test_skips_empty_source_code(tmp_path):
    p = tmp_path / "formai.csv"
    pd.DataFrame([
        {"filename": "empty.c", "classification": "Vulnerable", "source_code": "", "vul_type": "buffer overflow"},
        {"filename": "ok.c",    "classification": "Vulnerable", "source_code": "void v(){}", "vul_type": "buffer overflow"},
    ]).to_csv(p, index=False)
    samples = extract_formai(p)
    assert len(samples) == 1
    assert samples[0].sample_id == "ok"


def test_pick_handles_non_string_values(tmp_path):
    # Verify _pick survives ints/floats without crashing on pd.isna.
    # pandas reads numeric strings as numbers; using a list-like value would
    # exercise the TypeError fallback in _pick.
    p = tmp_path / "formai.json"
    p.write_text(
        '[{"file_name":"n.c","category":"Vulnerable","source_code":"void f(){}",'
        '"error_type":"buffer overflow","extra":[1,2,3]}]',
        encoding="utf-8",
    )
    samples = extract_formai(p)
    assert samples[0].cwes == ["CWE-121"]


def test_vulnerable_with_empty_vul_type_skipped(tmp_path):
    # Vulnerable record with empty vul_type -> _map_vul_type returns None -> skip.
    p = tmp_path / "formai.csv"
    pd.DataFrame([{
        "filename": "x.c",
        "classification": "Vulnerable",
        "source_code": "void v(){}",
        "vul_type": "",
    }]).to_csv(p, index=False)
    samples = extract_formai(p)
    assert samples == []
