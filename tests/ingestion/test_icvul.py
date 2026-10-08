import pytest
from pathlib import Path
from ingestion.icvul import extract_icvul
from ingestion.schema import FunctionSample


@pytest.fixture
def icvul_path(tmp_path):
    # function_info.csv -- hash = fc_hash, before_change filters vuln version
    (tmp_path / "function_info.csv").write_text(
        "hash,code,before_change\n"
        'abc123,"void vuln_fc() { return buf[i]; }",True\n'
        'def456,"void not_mapped() { return 0; }",True\n'
        'ghi789,"void vuln_fc2() { strcpy(dst, src); }",True\n'
        'abc123,"void vuln_fc() { patched; }",False\n'  # after_change -- must be excluded
    )
    # cve_fc_vcc_mapping.csv -- fc_hash is a plain string (NOT a list)
    (tmp_path / "cve_fc_vcc_mapping.csv").write_text(
        "cve_id,cwe_id,fc_hash,vcc_hash\n"
        "CVE-2020-0001,CWE-119,abc123,[]\n"
        "CVE-2020-0002,CWE-120,ghi789,[]\n"
        # def456 has no mapping -> not a FC
    )
    return tmp_path


def test_extracts_only_mapped_functions(icvul_path):
    samples = extract_icvul(icvul_path)
    codes = [s.code for s in samples]
    assert "void vuln_fc() { return buf[i]; }" in codes
    assert "void vuln_fc2() { strcpy(dst, src); }" in codes
    assert "void not_mapped() { return 0; }" not in codes


def test_excludes_after_change_version(icvul_path):
    """before_change=False rows must be excluded even if hash maps to a CVE."""
    samples = extract_icvul(icvul_path)
    codes = [s.code for s in samples]
    assert "void vuln_fc() { patched; }" not in codes


def test_no_negatives_produced(icvul_path):
    samples = extract_icvul(icvul_path)
    assert all(s.label == 1 for s in samples)


def test_cwe_assigned_to_function(icvul_path):
    samples = extract_icvul(icvul_path)
    f1 = next(s for s in samples if s.code == "void vuln_fc() { return buf[i]; }")
    assert f1.cwes == ["CWE-119"]


def test_returns_function_samples(icvul_path):
    samples = extract_icvul(icvul_path)
    assert all(isinstance(s, FunctionSample) for s in samples)


def test_nvd_placeholder_cwes_dropped(tmp_path):
    (tmp_path / "function_info.csv").write_text(
        "hash,code,before_change\n"
        'abc123,"void f() {}",True\n'
    )
    (tmp_path / "cve_fc_vcc_mapping.csv").write_text(
        "cve_id,cwe_id,fc_hash,vcc_hash\n"
        "CVE-2020-0001,NVD-CWE-Other,abc123,[]\n"
        "CVE-2020-0002,NVD-CWE-noinfo,abc123,[]\n"
    )
    samples = extract_icvul(tmp_path)
    assert samples == []
