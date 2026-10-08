import io
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from ingestion.schema import FunctionSample
import main as feast_cli


def _sample(code="code", cwes=None, label=0, branch="real", sample_id=""):
    return FunctionSample(
        code=code,
        cwes=[] if cwes is None else cwes,
        label=label,
        branch=branch,
        sample_id=sample_id,
    )


def _present_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("present", encoding="utf-8")


def _present_dir(path):
    _present_file(path / ".present")


def test_parse_cwe_whitelist_normalises_ids():
    assert feast_cli._parse_cwe_whitelist("cwe-079,89,CWE89") == {"CWE-79", "CWE-89"}


def test_parse_cwe_whitelist_accepts_none_and_rejects_empty_csv():
    assert feast_cli._parse_cwe_whitelist(None) is None
    with pytest.raises(ValueError):
        feast_cli._parse_cwe_whitelist(" , ")


def test_parse_cwe_whitelist_rejects_invalid_ids():
    with pytest.raises(ValueError):
        feast_cli._parse_cwe_whitelist("CWE-79,foo")


def test_parse_cwe_types_all_and_empty_values():
    assert feast_cli._parse_cwe_types(" ALL ") == feast_cli._VALID_CWE_TYPES
    with pytest.raises(ValueError):
        feast_cli._parse_cwe_types(" , ")


def test_parse_cwe_types_rejects_unknown_values():
    with pytest.raises(ValueError):
        feast_cli._parse_cwe_types("leaf,leef")


def test_parse_branches_normalises_and_rejects_unknown_values():
    assert feast_cli._parse_branches(None) is None
    assert feast_cli._parse_branches(" all ") is None
    assert feast_cli._parse_branches("Real,AI") == {"real", "ai"}
    with pytest.raises(ValueError):
        feast_cli._parse_branches("real,manual")
    with pytest.raises(ValueError):
        feast_cli._parse_branches(" , ")


def test_path_component_handles_reserved_weird_and_long_names():
    reserved = feast_cli._path_component("CON", fallback="dataset")
    assert reserved != "CON"
    assert len(reserved) <= feast_cli._MAX_PATH_COMPONENT

    weird = feast_cli._path_component('..//bad:name*with?chars <>|"', fallback="dataset")
    assert weird
    assert ".." not in weird
    assert not any(ch in weird for ch in '<>:"/\\|?*')

    long_name = feast_cli._path_component("x" * 300, fallback="dataset", max_len=64)
    assert len(long_name) <= 64
    assert long_name.endswith(feast_cli._path_hash("x" * 300))


def test_safe_extract_zip_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../outside.txt", "nope")

    with zipfile.ZipFile(archive) as zf, pytest.raises(ValueError):
        feast_cli._safe_extract_zip(zf, tmp_path / "out")


def test_safe_extract_zip_rejects_windows_traversal_and_drive_paths(tmp_path):
    archive = tmp_path / "bad_windows.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(r"..\outside.txt", "nope")
        zf.writestr(r"C:\temp\outside.txt", "nope")

    with zipfile.ZipFile(archive) as zf, pytest.raises(ValueError):
        feast_cli._safe_extract_zip(zf, tmp_path / "out")


def test_safe_extract_zip_normalises_backslash_members(tmp_path):
    archive = tmp_path / "ok_windows.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(r"nested\file.txt", "ok")

    out = tmp_path / "out"
    with zipfile.ZipFile(archive) as zf:
        feast_cli._safe_extract_zip(zf, out)

    assert (out / "nested" / "file.txt").read_text() == "ok"


def test_safe_extract_zip_creates_directories_and_rejects_symlinks(tmp_path):
    ok_archive = tmp_path / "dirs.zip"
    with zipfile.ZipFile(ok_archive, "w") as zf:
        zf.writestr("pkg/", "")
        zf.writestr("pkg/file.txt", "ok")

    out = tmp_path / "ok"
    with zipfile.ZipFile(ok_archive) as zf:
        feast_cli._safe_extract_zip(zf, out)

    assert (out / "pkg").is_dir()
    assert (out / "pkg" / "file.txt").read_text() == "ok"

    bad_archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(bad_archive, "w") as zf:
        zf.writestr(info, "target")

    with zipfile.ZipFile(bad_archive) as zf, pytest.raises(ValueError):
        feast_cli._safe_extract_zip(zf, tmp_path / "bad")


def test_safe_extract_tar_rejects_path_traversal(tmp_path):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tf:
        payload = b"nope"
        info = tarfile.TarInfo("../outside.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    data.seek(0)
    with tarfile.open(fileobj=data, mode="r") as tf, pytest.raises(ValueError):
        feast_cli._safe_extract_tar(tf, tmp_path / "out")


def test_safe_extract_tar_extracts_dirs_files_and_rejects_special_members(tmp_path):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tf:
        directory = tarfile.TarInfo("pkg")
        directory.type = tarfile.DIRTYPE
        tf.addfile(directory)
        payload = b"ok"
        info = tarfile.TarInfo("pkg/file.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    data.seek(0)
    out = tmp_path / "out"
    with tarfile.open(fileobj=data, mode="r") as tf:
        feast_cli._safe_extract_tar(tf, out)

    assert (out / "pkg").is_dir()
    assert (out / "pkg" / "file.txt").read_text() == "ok"

    bad = io.BytesIO()
    with tarfile.open(fileobj=bad, mode="w") as tf:
        link = tarfile.TarInfo("link")
        link.type = tarfile.SYMTYPE
        link.linkname = "target"
        tf.addfile(link)

    bad.seek(0)
    with tarfile.open(fileobj=bad, mode="r") as tf, pytest.raises(ValueError):
        feast_cli._safe_extract_tar(tf, tmp_path / "bad")


def test_archive_target_rejects_empty_nul_and_absolute_paths(tmp_path):
    for name in ["", "bad\x00name", "/abs.txt", r"\abs.txt", r"C:\abs.txt"]:
        with pytest.raises(ValueError):
            feast_cli._archive_target(tmp_path, name)

    target = feast_cli._archive_target(tmp_path, "./nested/./file.txt")
    assert target == (tmp_path / "nested" / "file.txt").resolve()


def test_resolve_source_matches_exact_case_insensitive_slug_and_missing():
    registry = {"CVEfixes(Python)": object(), "PyVul": object()}
    assert feast_cli._resolve_source("PyVul", registry) == "PyVul"
    assert feast_cli._resolve_source("pyvul", registry) == "PyVul"
    assert feast_cli._resolve_source("cvefixes python", registry) == "CVEfixes(Python)"
    assert feast_cli._resolve_source("missing", registry) is None


def test_with_ids_fills_missing_ids_but_keeps_existing_ids():
    wrapped = feast_cli._with_ids(lambda: [
        _sample(code="def a(): pass", sample_id=""),
        _sample(code="def b(): pass", sample_id="kept"),
    ])

    samples = wrapped()

    assert samples[0].sample_id == feast_cli._hash("def a(): pass")[:16]
    assert samples[1].sample_id == "kept"


def test_to_df_filters_vulnerable_cwes_and_keeps_safe_rows(monkeypatch):
    cwe_types = {"CWE-79": "leaf", "CWE-999": "unknown"}
    monkeypatch.setattr(feast_cli, "_cwe_type", lambda cwe: cwe_types[cwe])

    df = feast_cli._to_df(
        "Dataset",
        [
            _sample("def vuln(): pass", ["CWE-79", "CWE-999"], 1, sample_id="v"),
            _sample("def filtered(): pass", ["CWE-999"], 1, sample_id="drop"),
            _sample("def safe(): pass", ["CWE-79"], 0, sample_id="s"),
        ],
        "Python",
        allowed_types={"leaf"},
        allowed_cwes={"CWE-79"},
    )

    assert list(df["sample_id"]) == ["v", "s"]
    assert df.loc[df["sample_id"] == "v", "cwes"].item() == ["CWE-79"]
    assert df.loc[df["sample_id"] == "s", "cwes"].item() == []


def test_to_df_returns_schema_for_empty_result(monkeypatch):
    monkeypatch.setattr(feast_cli, "_cwe_type", lambda _cwe: "unknown")

    df = feast_cli._to_df(
        "Dataset",
        [_sample("def filtered(): pass", ["CWE-999"], 1)],
        "Python",
        allowed_types={"leaf"},
        allowed_cwes=None,
    )

    assert df.empty
    assert list(df.columns) == [
        "code",
        "language",
        "label",
        "cwes",
        "branch",
        "source",
        "code_hash",
        "sample_id",
    ]


def test_load_collections_resolves_filters_skips_unknown_and_missing():
    registry = {
        "RealSet": lambda: [
            _sample("real", branch="real"),
            _sample("ai", branch="ai"),
        ],
        "MissingSet": lambda: (_ for _ in ()).throw(FileNotFoundError("missing")),
    }

    collections = feast_cli._load_collections(
        registry,
        sources=["realset", "MissingSet", "unknown"],
        branches={"real"},
    )

    assert list(collections) == ["RealSet"]
    assert [s.code for s in collections["RealSet"]] == ["real"]


def test_process_lang_writes_per_dataset_parquets(tmp_path, monkeypatch):
    monkeypatch.setattr(feast_cli, "PROC_DIR", tmp_path)
    monkeypatch.setattr(feast_cli, "_cwe_type", lambda _cwe: "leaf")

    dfs = feast_cli._process_lang(
        {"PyVul": [_sample("def vuln(): pass", ["CWE-79"], 1, sample_id="v")]},
        "Python",
        allowed_types={"leaf"},
        allowed_cwes=None,
    )

    assert list(dfs) == ["PyVul"]
    assert (tmp_path / "python" / "pyvul.parquet").exists()


def test_merge_lang_returns_empty_when_all_inputs_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(feast_cli, "MERGED_DIR", tmp_path)
    empty = pd.DataFrame(columns=[
        "code",
        "language",
        "label",
        "cwes",
        "branch",
        "source",
        "code_hash",
        "sample_id",
    ])

    merged = feast_cli._merge_lang({"Empty": empty}, "Python", min_cwe_count=1)

    assert merged.empty
    assert not (tmp_path / "python_merged.parquet").exists()


def test_merge_lang_keeps_priority_and_filters_sparse_cwes(tmp_path, monkeypatch):
    monkeypatch.setattr(feast_cli, "MERGED_DIR", tmp_path)

    vuln_real = pd.DataFrame([
        {
            "code": "def vuln_one(): pass",
            "language": "Python",
            "label": 1,
            "cwes": ["CWE-79"],
            "branch": "real",
            "source": "RealSet",
            "code_hash": "same",
            "sample_id": "same",
        },
        {
            "code": "def vuln_two(): pass",
            "language": "Python",
            "label": 1,
            "cwes": ["CWE-79"],
            "branch": "real",
            "source": "RealSet",
            "code_hash": "keep",
            "sample_id": "keep",
        },
        {
            "code": "def sparse(): pass",
            "language": "Python",
            "label": 1,
            "cwes": ["CWE-999"],
            "branch": "real",
            "source": "RealSet",
            "code_hash": "drop",
            "sample_id": "drop",
        },
    ])
    safe_ai_conflict = pd.DataFrame([
        {
            "code": "def vuln_one(): pass",
            "language": "Python",
            "label": 0,
            "cwes": [],
            "branch": "ai",
            "source": "AISet",
            "code_hash": "same",
            "sample_id": "same",
        }
    ])

    merged = feast_cli._merge_lang(
        {"RealSet": vuln_real, "AISet": safe_ai_conflict},
        "Python",
        min_cwe_count=2,
    )

    assert set(merged["code_hash"]) == {"same", "keep"}
    assert merged.loc[merged["code_hash"] == "same", "label"].item() == 1
    assert (tmp_path / "python_merged.parquet").exists()


def test_cmd_synthesize_runs_all_languages_with_synthetic_registry(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    merged_dir = tmp_path / "merged"
    monkeypatch.setattr(feast_cli, "PROC_DIR", processed_dir)
    monkeypatch.setattr(feast_cli, "MERGED_DIR", merged_dir)
    monkeypatch.setattr(feast_cli, "_ensure_nav", lambda: None)
    monkeypatch.setattr(feast_cli, "_cwe_type", lambda _cwe: "leaf")
    monkeypatch.setattr(
        feast_cli,
        "_build_registry",
        lambda _raw: {
            "C/C++": {"CSet": lambda: [_sample("int main() {}", ["CWE-79"], 1, sample_id="c")]},
            "Java": {"JSet": lambda: [_sample("class A {}", [], 0, sample_id="j")]},
            "Python": {"PSet": lambda: [_sample("def p(): pass", [], 0, sample_id="p")]},
        },
    )

    feast_cli.cmd_synthesize(SimpleNamespace(
        lang="all",
        cwe_types="leaf",
        cwes=None,
        branches="all",
        min_cwe_count=1,
        sources=None,
    ))

    assert (processed_dir / "c_cpp" / "cset.parquet").exists()
    assert (merged_dir / "c_cpp_merged.parquet").exists()
    assert (merged_dir / "java_merged.parquet").exists()
    assert (merged_dir / "python_merged.parquet").exists()


def test_cmd_synthesize_continues_when_language_has_no_collections(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(feast_cli, "PROC_DIR", tmp_path / "processed")
    monkeypatch.setattr(feast_cli, "MERGED_DIR", tmp_path / "merged")
    monkeypatch.setattr(feast_cli, "_ensure_nav", lambda: None)
    monkeypatch.setattr(
        feast_cli,
        "_build_registry",
        lambda _raw: {"Python": {"EmptySet": lambda: []}},
    )

    feast_cli.cmd_synthesize(SimpleNamespace(
        lang="python",
        cwe_types="leaf",
        cwes="CWE-79",
        branches="real",
        min_cwe_count=2,
        sources="missing",
    ))

    assert "No data for Python" in capsys.readouterr().out
    assert not (tmp_path / "merged" / "python_merged.parquet").exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lang": "ruby"}, "Unknown language"),
        ({"cwe_types": "leef"}, "Invalid --cwe-types"),
        ({"cwes": "CWE-79,nope"}, "Invalid --cwes"),
        ({"branches": "manual"}, "Invalid --branches"),
        ({"min_cwe_count": 0}, "Invalid --min-cwe-count"),
    ],
)
def test_cmd_synthesize_exits_for_invalid_args(kwargs, message, capsys):
    args = {
        "lang": "python",
        "cwe_types": "leaf",
        "cwes": None,
        "branches": "all",
        "min_cwe_count": 1,
        "sources": None,
    }
    args.update(kwargs)

    with pytest.raises(SystemExit):
        feast_cli.cmd_synthesize(SimpleNamespace(**args))

    assert message in capsys.readouterr().out


def test_materialize_sanitises_weird_and_long_paths(tmp_path, monkeypatch):
    merged_dir = tmp_path / "merged"
    materialized_dir = tmp_path / "materialized"
    merged_dir.mkdir()
    monkeypatch.setattr(feast_cli, "MERGED_DIR", merged_dir)
    monkeypatch.setattr(feast_cli, "MAT_DIR", materialized_dir)

    long_sample_id = "CON" + ("x" * 300)
    weird_source = '..//CON:very weird dataset name with spaces and <>:"|?*' + ("z" * 200)
    code_hash = "a" * 64
    pd.DataFrame([
        {
            "code": "def safe():\n    return 1\n",
            "language": "Python",
            "label": 0,
            "cwes": [],
            "branch": "real",
            "source": weird_source,
            "code_hash": code_hash,
            "sample_id": long_sample_id,
        }
    ]).to_parquet(merged_dir / "python_merged.parquet", index=False)

    feast_cli.cmd_materialize(SimpleNamespace(lang="python", overwrite=True))

    index = pd.read_parquet(materialized_dir / "python" / "index.parquet")
    rel_path = index["materialized_path"].item()
    assert ".." not in rel_path.split("/")
    assert not any(ch in rel_path for ch in '<>:"\\|?*')
    assert all(len(part) <= feast_cli._MAX_PATH_COMPONENT + 3 for part in rel_path.split("/"))
    assert (materialized_dir / "python" / rel_path).exists()


def test_materialize_skips_existing_files_and_backfills_sample_id(tmp_path, monkeypatch):
    merged_dir = tmp_path / "merged"
    materialized_dir = tmp_path / "materialized"
    merged_dir.mkdir()
    monkeypatch.setattr(feast_cli, "MERGED_DIR", merged_dir)
    monkeypatch.setattr(feast_cli, "MAT_DIR", materialized_dir)

    code_hash = "b" * 64
    pd.DataFrame([{
        "code": "print('new')\n",
        "language": "Python",
        "label": 0,
        "cwes": [],
        "branch": "real",
        "source": "PyVul",
        "code_hash": code_hash,
    }]).to_parquet(merged_dir / "python_merged.parquet", index=False)

    existing = materialized_dir / "python" / "pyvul" / f"{code_hash[:16]}.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("print('old')\n")

    feast_cli.cmd_materialize(SimpleNamespace(lang="python", overwrite=False))

    assert existing.read_text() == "print('old')\n"
    index = pd.read_parquet(materialized_dir / "python" / "index.parquet")
    assert index["sample_id"].item() == code_hash[:16]


def test_materialize_handles_missing_inputs_source_column_and_bad_language(tmp_path, monkeypatch, capsys):
    merged_dir = tmp_path / "merged"
    materialized_dir = tmp_path / "materialized"
    merged_dir.mkdir()
    monkeypatch.setattr(feast_cli, "MERGED_DIR", merged_dir)
    monkeypatch.setattr(feast_cli, "MAT_DIR", materialized_dir)

    feast_cli.cmd_materialize(SimpleNamespace(lang="java", overwrite=False))
    assert "not found" in capsys.readouterr().out

    pd.DataFrame([{
        "code": "def p(): pass",
        "label": 0,
        "cwes": [],
        "branch": "real",
        "code_hash": "c" * 64,
        "sample_id": "sample",
    }]).to_parquet(merged_dir / "python_merged.parquet", index=False)
    feast_cli.cmd_materialize(SimpleNamespace(lang="python", overwrite=False))
    assert 'missing "source" column' in capsys.readouterr().out

    with pytest.raises(SystemExit):
        feast_cli.cmd_materialize(SimpleNamespace(lang="ruby", overwrite=False))


def test_materialize_all_languages_writes_c_java_python(tmp_path, monkeypatch):
    merged_dir = tmp_path / "merged"
    materialized_dir = tmp_path / "materialized"
    merged_dir.mkdir()
    monkeypatch.setattr(feast_cli, "MERGED_DIR", merged_dir)
    monkeypatch.setattr(feast_cli, "MAT_DIR", materialized_dir)

    for slug, language, code in [
        ("c_cpp", "C/C++", "int main(void) { return 0; }\n"),
        ("java", "Java", "class A {}\n"),
        ("python", "Python", "def p():\n    return 1\n"),
    ]:
        pd.DataFrame([{
            "code": code,
            "language": language,
            "label": 0,
            "cwes": [],
            "branch": "real",
            "source": "Dataset",
            "code_hash": slug * 16,
            "sample_id": f"{slug}-sample",
        }]).to_parquet(merged_dir / f"{slug}_merged.parquet", index=False)

    feast_cli.cmd_materialize(SimpleNamespace(lang="all", overwrite=True))

    assert (materialized_dir / "c_cpp" / "dataset" / "c_cpp-sample.c").exists()
    assert (materialized_dir / "java" / "dataset" / "java-sample.java").exists()
    assert (materialized_dir / "python" / "dataset" / "python-sample.py").exists()




def test_enrich_language_with_sat_reports_uses_runs_and_preserves_empty_tool_columns():
    merged = pd.DataFrame([
        {
            "code": "def a(): pass",
            "language": "Python",
            "label": 1,
            "cwes": ["CWE-79"],
            "branch": "real",
            "source": "PyVul",
            "code_hash": "a" * 64,
            "sample_id": "sample-a",
        },
        {
            "code": "def b(): pass",
            "language": "Python",
            "label": 0,
            "cwes": [],
            "branch": "ai",
            "source": "SecurityEval",
            "code_hash": "b" * 64,
            "sample_id": "sample-b",
        },
    ])
    payload = {
        "tools": [
            {"name": "bandit", "status": "completed", "hit_count": 1},
            {"name": "semgrep", "status": "completed", "hit_count": 0},
            {"name": "pysa", "status": "skipped", "hit_count": 0},
        ],
        "runs": [
            {
                "file": "pyvul/sample-a.py",
                "tools": {"bandit": ["CWE-79", "79"], "semgrep": [], "pysa": None},
            },
            {
                "file": "old_path/sample-b.py",
                "tools": {"bandit": [], "semgrep": ["CWE-89"]},
            },
            {
                "file": "missing.py",
                "tools": {"bandit": ["CWE-20"]},
            },
        ],
        "findings": [
            {"file": "pyvul/sample-a.py", "source_tool": "bandit", "cwe_ids": ["CWE-120"]},
            {"file": "pyvul/sample-a.py", "source_tool": "semgrep", "cwe_ids": []},
        ],
    }

    enriched, stats = feast_cli._enrich_language_with_sat_reports(merged, [(Path("python.json"), payload)])

    assert stats["runs"] == 3
    assert stats["runs_matched"] == 2
    assert stats["findings"] == 2
    assert stats["findings_matched"] == 1
    assert set(stats["tools"]) == {"bandit", "semgrep"}
    assert "pysa" not in enriched.columns
    assert enriched.loc[0, "bandit"] == ["CWE-79", "CWE-120"]
    assert enriched.loc[0, "semgrep"] == []
    assert enriched.loc[1, "bandit"] == []
    assert enriched.loc[1, "semgrep"] == ["CWE-89"]


def test_cmd_enrich_writes_one_output_parquet_per_language(tmp_path):
    merged_dir = tmp_path / "merged"
    reports_dir = tmp_path / "SAT-reports"
    out_dir = tmp_path / "enriched"
    merged_dir.mkdir()
    reports_dir.mkdir()

    pd.DataFrame([{
        "code": "def a(): pass",
        "language": "Python",
        "label": 1,
        "cwes": ["CWE-79"],
        "branch": "real",
        "source": "PyVul",
        "code_hash": "a" * 64,
        "sample_id": "sample-a",
    }]).to_parquet(merged_dir / "python_merged.parquet", index=False)
    pd.DataFrame([{
        "code": "class A {}",
        "language": "Java",
        "label": 0,
        "cwes": [],
        "branch": "synth",
        "source": "CAPEC_LLM(Java)",
        "code_hash": "b" * 64,
        "sample_id": "sample-b",
    }]).to_parquet(merged_dir / "java_merged.parquet", index=False)

    (reports_dir / "python.json").write_text(
        """{"tools": [{"name": "semgrep", "status": "completed"}], "runs": [
          {"file": "pyvul/sample-a.py", "tools": {"semgrep": ["CWE-79"]}}
        ], "findings": []}""",
        encoding="utf-8",
    )
    (reports_dir / "java.json").write_text(
        """{"tools": [{"name": "codeql", "status": "completed"}], "runs": [
          {"file": "capec_llm_java/sample-b.java", "tools": {"codeql": ["CWE-22"]}}
        ], "findings": []}""",
        encoding="utf-8",
    )

    feast_cli.cmd_enrich(SimpleNamespace(merged_dir=merged_dir, reports_dir=reports_dir, out_dir=out_dir))

    python_enriched = pd.read_parquet(out_dir / "python.parquet")
    java_enriched = pd.read_parquet(out_dir / "java.parquet")
    assert list(python_enriched["semgrep"])[0] == ["CWE-79"]
    assert "codeql" not in python_enriched.columns
    assert list(java_enriched["codeql"])[0] == ["CWE-22"]
    assert "semgrep" not in java_enriched.columns


def test_build_registry_contains_expected_sources():
    registry = feast_cli._build_registry(feast_cli.ROOT / "data" / "raw")

    assert set(registry) == {"C/C++", "Java", "Python"}
    assert {"PrimeVul", "ICVul", "MegaVul", "CASTLE", "FormAI"} <= set(registry["C/C++"])
    assert {"CVEfixes(Java)", "OWASP(Java)", "CAPEC_LLM(Java)"} <= set(registry["Java"])
    assert {"PyVul", "SecurityEval", "PatchEval", "LLMSecEval"} <= set(registry["Python"])


def test_cmd_list_prints_registry(monkeypatch, capsys):
    monkeypatch.setattr(
        feast_cli,
        "_build_registry",
        lambda _raw: {
            "Python": {"PyVul": lambda: []},
            "Java": {"OWASP(Java)": lambda: []},
        },
    )

    feast_cli.cmd_list(SimpleNamespace())

    output = capsys.readouterr().out
    assert "FEAST source datasets" in output
    assert "PyVul" in output
    assert "--sources" in output


def test_cmd_download_skips_when_all_datasets_are_present(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "raw"
    monkeypatch.setattr(feast_cli, "RAW_DIR", raw)

    for file_name in [
        "primevul_train.jsonl",
        "primevul_test.jsonl",
        "secvuleval.csv",
        "crossvul.zip",
        "juliet_c.zip",
        "juliet_java.zip",
    ]:
        _present_file(raw / file_name)

    for dir_name in [
        "icvul",
        "cvefixes",
        "megavul",
        "sven",
        "castle",
        "formai",
        "owasp_benchmark",
        "owasp_benchmark_python",
        "capec_llm",
        "patcheval",
        "pyvul",
        "security_eval",
    ]:
        _present_file(raw / dir_name / ".present")

    _present_file(raw / "llmseceval" / "zenodo" / "case" / "gen_scenario" / "vuln.py")
    _present_file(raw / "llmseceval" / "CWE-79" / "Secure" / "safe.py")

    feast_cli.cmd_download(SimpleNamespace())

    output = capsys.readouterr().out
    assert "PrimeVul" in output
    assert "already present" in output
    assert "All datasets present" in output


def test_cmd_download_uses_faked_downloaders_without_network(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "raw"
    monkeypatch.setattr(feast_cli, "RAW_DIR", raw)

    hf_module = ModuleType("huggingface_hub")

    def fake_hf_hub_download(repo_id, filename, repo_type):
        del repo_id, repo_type
        local = tmp_path / "hf" / filename.replace("/", "_")
        _present_file(local)
        return str(local)

    def fake_snapshot_download(repo_id, repo_type):
        del repo_type
        local = tmp_path / "snapshot" / feast_cli._slug(repo_id)
        _present_file(local / "sample.txt")
        return str(local)

    hf_module.hf_hub_download = fake_hf_hub_download
    hf_module.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_module)

    gdown_module = ModuleType("gdown")

    def fake_gdown_download(id, output):
        del id
        with zipfile.ZipFile(output, "w") as zf:
            zf.writestr("function_info.csv", "id,code\n1,int main(){}\n")
        return output

    gdown_module.download = fake_gdown_download
    monkeypatch.setitem(sys.modules, "gdown", gdown_module)

    datasets_module = ModuleType("datasets")

    def fake_load_dataset(name, trust_remote_code):
        del name, trust_remote_code
        frame = pd.DataFrame([{"code": "int main() {}", "target": 0}])
        return {"train": SimpleNamespace(to_pandas=lambda: frame)}

    datasets_module.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    def fake_urlretrieve(url, dest):
        del url
        _present_file(dest)
        return str(dest), None

    def fake_subprocess_run(args, capture_output, text):
        del capture_output, text
        url = args[3]
        dest = raw / args[-1] if not str(args[-1]).startswith(str(raw)) else args[-1]
        dest = feast_cli.Path(dest)
        if "LLMSecEval" in url:
            _present_file(dest / "Dataset" / "Secure Code Samples" / "CWE-79" / "safe.py")
            _present_file(dest / "Dataset" / "Secure Code Samples" / "misc" / "ignored.py")
        else:
            _present_file(dest / "README.md")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    feast_cli.cmd_download(SimpleNamespace())

    output = capsys.readouterr().out
    assert "PrimeVul" in output
    assert "manual download required" in output
    assert "Manual downloads still required" in output
    assert (raw / "primevul_train.jsonl").exists()
    assert (raw / "icvul" / "function_info.csv").exists()
    assert (raw / "cvefixes" / "train-00000-of-00003.parquet").exists()
    assert (raw / "secvuleval.csv").exists()
    assert (raw / "formai" / "FormAI_dataset_human_readable-V1.csv").exists()
    assert (raw / "juliet_c.zip").exists()
    assert (raw / "llmseceval" / "CWE-79" / "Secure" / "safe.py").exists()


def test_build_parser_supports_aliases_and_data_dir(tmp_path):
    parser = feast_cli._build_parser()

    args = parser.parse_args(["s", "--lang", "py", "--data-dir", str(tmp_path)])
    assert args.command == "s"
    assert args.lang == "py"
    assert args.data_dir == tmp_path

    fuse_args = parser.parse_args([
        "fusion", "--calibration", "sensitivity,specificity",
        "--taumin", "0.1", "--taumax", "0.8",
    ])
    assert fuse_args.calibration == "sensitivity,specificity"
    assert fuse_args.tau_min == 0.1
    assert fuse_args.tau_max == 0.8

    assert parser.parse_args(["dl"]).command == "dl"
    assert parser.parse_args(["m"]).command == "m"
    assert parser.parse_args(["ls"]).command == "ls"


def test_main_dispatches_commands_and_data_dir(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(feast_cli, "RAW_DIR", feast_cli.ROOT / "data" / "raw")
    monkeypatch.setattr(feast_cli.sys, "argv", ["main.py", "s", "--data-dir", str(tmp_path)])
    monkeypatch.setattr(feast_cli, "cmd_synthesize", lambda args: calls.append(("synth", args.lang)))

    feast_cli.main()

    assert calls == [("synth", "all")]
    assert feast_cli.RAW_DIR == tmp_path


@pytest.mark.parametrize(
    ("argv", "command_name"),
    [
        (["main.py", "download"], "download"),
        (["main.py", "materialize"], "materialize"),
        (["main.py", "list"], "list"),
    ],
)
def test_main_dispatches_remaining_commands(monkeypatch, argv, command_name):
    calls = []
    monkeypatch.setattr(feast_cli.sys, "argv", argv)
    monkeypatch.setattr(feast_cli, "cmd_download", lambda _args: calls.append("download"))
    monkeypatch.setattr(feast_cli, "cmd_materialize", lambda _args: calls.append("materialize"))
    monkeypatch.setattr(feast_cli, "cmd_list", lambda _args: calls.append("list"))

    feast_cli.main()

    assert calls == [command_name]
