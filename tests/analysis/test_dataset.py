
import pandas as pd

from analysis.dataset import FeastDataset, normalise_cwe


def _df():
    return pd.DataFrame([
        {
            "sample_id": "a",
            "language": "Python",
            "branch": "real",
            "source": "PyVul",
            "cwes": ["CWE-79"],
            "bandit": ["CWE-79"],
            "semgrep": [],
        },
        {
            "sample_id": "b",
            "language": "Java",
            "branch": "synth",
            "source": "CAPEC_LLM(Java)",
            "cwes": ["CWE-89"],
            "bandit": [],
            "semgrep": ["CWE-89"],
        },
    ])


def test_normalise_cwe_accepts_common_forms():
    assert normalise_cwe("cwe-079") == "CWE-79"
    assert normalise_cwe("CWE89") == "CWE-89"
    assert normalise_cwe("20") == "CWE-20"
    assert normalise_cwe("not-a-cwe") is None


def test_feast_dataset_chainable_filters():
    dataset = FeastDataset(_df())

    subset = dataset.filter_by_branch("real").filter_by_cwe("CWE-79").filter_by_tool("bandit")

    assert len(subset) == 1
    assert subset.to_pandas()["sample_id"].item() == "a"
    assert len(dataset) == 2
    assert len(subset.history) == 3


def test_feast_dataset_tool_filter_can_target_specific_cwe():
    dataset = FeastDataset(_df())

    subset = dataset.filter_by_tool("semgrep", cwes="CWE-89")

    assert len(subset) == 1
    assert subset.to_pandas()["sample_id"].item() == "b"
