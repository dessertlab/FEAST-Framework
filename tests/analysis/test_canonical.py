import pandas as pd
import pytest

from analysis.canonical import CANONICAL_LEVEL, LEVELS, CweCanonicalizer
from ingestion.cwe_navigator import CWENavigator

XML = "data/cwec_latest.xml"


@pytest.fixture(scope="module")
def navigator():
    return CWENavigator(XML)


def test_primary_parent_follows_view_1000_primary_edge(navigator):
    # CWE-120 has parents in views 700/1003/1340 too; only the 1000 primary (787) must win.
    assert navigator.primary_parent("120") == "787"
    assert navigator.primary_parent("CWE-798") == "1391"
    assert navigator.primary_parent("73") == "642"


def test_primary_path_reaches_pillar(navigator):
    assert navigator.primary_path("120") == ("120", "787", "119", "118", "664")
    assert navigator.abstraction("664") == "Pillar"


def test_only_pillar_child_level_is_supported(navigator):
    assert LEVELS == ("pillar_child",)
    assert CANONICAL_LEVEL == "pillar_child"
    with pytest.raises(ValueError):
        CweCanonicalizer(navigator, level="class")


@pytest.mark.parametrize("cwe,family", [
    ("79", "CWE-74"),     # 79 -> 74 -> 707
    ("89", "CWE-74"),     # 89 -> 943 -> 74 -> 707
    ("78", "CWE-74"),     # 78 -> 77 -> 74 -> 707
    ("120", "CWE-118"),   # 120 -> 787 -> 119 -> 118 -> 664
    ("125", "CWE-118"),   # 125 -> 119 -> 118 -> 664
    ("1024", "CWE-1024"), # 1024 -> 697, already a direct pillar child
])
def test_family_mapping_to_direct_child_of_pillar(navigator, cwe, family):
    canon = CweCanonicalizer(navigator)
    assert canon.family(cwe) == family


def test_pillars_are_not_canonical_families(navigator):
    canon = CweCanonicalizer(navigator)
    assert navigator.abstraction("707") == "Pillar"
    assert canon.family("707") is None
    assert canon.family("664") is None


def test_unmapped_tokens_return_none(navigator):
    canon = CweCanonicalizer(navigator)
    assert canon.family("CWE-0") is None
    assert canon.family("not-a-cwe") is None


def test_canonicalize_frame_rewrites_gt_and_tools(navigator):
    canon = CweCanonicalizer(navigator)
    df = pd.DataFrame([
        {"cwes": ["CWE-89"], "toolA": ["CWE-79"], "toolB": []},
        {"cwes": ["CWE-120", "CWE-125"], "toolA": ["CWE-787"], "toolB": ["CWE-707", "CWE-0"]},
    ])
    out = canon.canonicalize_frame(df, ["toolA", "toolB"])
    assert out.at[0, "cwes"] == ["CWE-74"]
    assert out.at[0, "toolA"] == ["CWE-74"]
    assert out.at[1, "cwes"] == ["CWE-118"]
    assert out.at[1, "toolA"] == ["CWE-118"]
    assert out.at[1, "toolB"] == []


def test_canonical_map_reports_single_level(navigator):
    canon = CweCanonicalizer(navigator)
    mapping = canon.canonical_map(["CWE-79", "CWE-707"])
    assert mapping.loc[0, ["cwe", "family", "level"]].to_dict() == {
        "cwe": "CWE-79", "family": "CWE-74", "level": "pillar_child"
    }
    assert mapping.loc[1, "cwe"] == "CWE-707"
    assert pd.isna(mapping.loc[1, "family"])
    assert mapping.loc[1, "level"] == "pillar_child"
