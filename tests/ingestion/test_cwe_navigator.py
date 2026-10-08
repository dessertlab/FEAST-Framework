import pytest

from ingestion.cwe_navigator import CWENavigator


@pytest.fixture
def cwe_xml(tmp_path):
    path = tmp_path / "cwe.xml"
    path.write_text(
        """<?xml version="1.0"?>
<Weakness_Catalog xmlns="http://cwe.mitre.org/cwe-7">
  <Weaknesses>
    <Weakness ID="20" Name="Input Validation"/>
    <Weakness ID="77" Name="Command Injection">
      <Related_Weaknesses>
        <Related_Weakness Nature="PeerOf" CWE_ID="20"/>
      </Related_Weaknesses>
    </Weakness>
    <Weakness ID="79" Name="XSS">
      <Related_Weaknesses>
        <Related_Weakness Nature="ChildOf" CWE_ID="20"/>
      </Related_Weaknesses>
    </Weakness>
    <Weakness ID="89" Name="SQL Injection">
      <Related_Weaknesses>
        <Related_Weakness Nature="ChildOf" CWE_ID="79"/>
      </Related_Weaknesses>
    </Weakness>
    <Weakness ID="999" Name="Orphan"/>
  </Weaknesses>
  <Categories>
    <Category ID="1000" Name="Root Category">
      <Relationships>
        <Has_Member CWE_ID="20"/>
        <Has_Member/>
        <Has_Member CWE_ID="2000"/>
      </Relationships>
    </Category>
    <Category ID="2000" Name="Nested Category">
      <Relationships>
        <Has_Member CWE_ID="79"/>
      </Relationships>
    </Category>
    <Category ID="3000" Name="Detached Category">
      <Relationships>
        <Has_Member CWE_ID="999"/>
      </Relationships>
    </Category>
    <Category ID="4000" Name="Child Category">
      <Relationships>
        <Has_Member Nature="ChildOf" CWE_ID="1000"/>
      </Relationships>
    </Category>
  </Categories>
  <Views>
    <View ID="VIEW" Name="Research View">
      <Members>
        <Has_Member CWE_ID="1000"/>
        <Has_Member/>
      </Members>
    </View>
    <View ID="EMPTY" Name="Empty View">
      <Members/>
    </View>
  </Views>
</Weakness_Catalog>
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def navigator(cwe_xml):
    return CWENavigator(str(cwe_xml))


def test_navigator_parses_entities_and_child_relations(navigator):
    assert set(navigator.weaknesses) == {"20", "77", "79", "89", "999"}
    assert set(navigator.categories) == {"1000", "2000", "3000", "4000"}
    assert set(navigator.views) == {"VIEW", "EMPTY"}
    assert navigator.child_of["79"] == ["20"]
    assert navigator.child_of["4000"] == ["1000"]
    assert "77" not in navigator.child_of


def test_get_element_name(navigator):
    assert navigator.get_element_name("79") == "XSS"
    assert navigator.get_element_name("1000") == "Root Category"
    assert navigator.get_element_name("missing") == "Unknown"
