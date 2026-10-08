from ingestion.utils import split_cwe, _NVD_PLACEHOLDERS


def test_split_cwe_simple():
    assert split_cwe("CWE-119") == ["CWE-119"]


def test_split_cwe_concatenated():
    assert split_cwe("CWE-20CWE-190") == ["CWE-20", "CWE-190"]


def test_split_cwe_nvd_placeholder():
    assert split_cwe("NVD-CWE-noinfo") == []
    assert split_cwe("NVD-CWE-Other") == []


def test_split_cwe_empty():
    assert split_cwe("") == []
    assert split_cwe(None) == []


def test_split_cwe_no_numeric():
    assert split_cwe("CWE-Other") == []


def test_split_cwe_multiple_tokens():
    result = split_cwe("CWE-79 and CWE-89")
    assert result == ["CWE-79", "CWE-89"]
