import pandas as pd

from analysis.family_performance_plots import (
    MAXIMIZE_METRICS,
    family_metric_matrix,
    save_family_performance_plots,
    top_families,
)


def _metric_row(strategy, family, support, f1, precision=None, fpr=0.1):
    return {
        "strategy": strategy,
        "family": family,
        "positive_support": support,
        "f1": f1,
        "precision": f1 if precision is None else precision,
        "fpr": fpr,
    }


def _per_family():
    families = [("CWE-1", 50), ("CWE-2", 40), ("CWE-3", 30), ("CWE-4", 20)]
    rows = []
    for family, support in families:
        rows.extend([
            _metric_row("tool:a", family, support, 0.40),
            _metric_row("tool:b", family, support, 0.45),
            _metric_row("or_1_of_2", family, support, 0.50),
            _metric_row("traditional_2_of_2", family, support, 0.55),
            _metric_row("weighted_fire_ppv_silence_npv_tau_0_5", family, support, 0.70),
            _metric_row("weighted_fire_fpr_silence_fnr_tau_0_7", family, support, 0.60),
            _metric_row("bks_tau_0_4", family, support, 0.65),
        ])
    return pd.DataFrame(rows)


def _overall():
    return pd.DataFrame([
        {"strategy": "tool:a", "f1_weighted": 0.40, "precision_weighted": 0.40},
        {"strategy": "tool:b", "f1_weighted": 0.45, "precision_weighted": 0.45},
        {"strategy": "or_1_of_2", "f1_weighted": 0.50, "precision_weighted": 0.50},
        {"strategy": "traditional_2_of_2", "f1_weighted": 0.55, "precision_weighted": 0.55},
        {"strategy": "weighted_fire_ppv_silence_npv_tau_0_5", "f1_weighted": 0.70, "precision_weighted": 0.70},
        {"strategy": "weighted_fire_fpr_silence_fnr_tau_0_7", "f1_weighted": 0.60, "precision_weighted": 0.60},
        {"strategy": "bks_tau_0_4", "f1_weighted": 0.65, "precision_weighted": 0.65},
    ])


def test_top_families_uses_ground_truth_support():
    assert top_families(_per_family(), n=3) == ["CWE-1", "CWE-2", "CWE-3"]


def test_family_metric_matrix_selects_representative_strategy_rows():
    matrix, support = family_metric_matrix(_per_family(), _overall(), "f1", tools=["a", "b"], top_n=3)

    assert list(matrix.columns) == ["CWE-1", "CWE-2", "CWE-3"]
    assert list(support.astype(int)) == [50, 40, 30]
    assert "tool:a" in matrix.index
    assert "or_1_of_2" in matrix.index
    assert "traditional_2_of_2" in matrix.index
    assert "weighted (tau=0.5, cal=ppv/npv)" in matrix.index
    assert "weighted (tau=0.7, cal=fpr/fnr)" not in matrix.index
    assert "bks (tau=0.4)" in matrix.index


def test_family_performance_plots_write_heatmap_and_dot_heatmap(tmp_path):
    written = save_family_performance_plots(
        _per_family(),
        _overall(),
        tmp_path,
        metrics=("f1", "fpr"),
        tools=["a", "b"],
        top_n=3,
    )
    names = {path.name for path in written}

    assert "fpr" not in MAXIMIZE_METRICS
    assert names == {
        "heatmap_f1.svg", "heatmap_f1.png",
        "dot_heatmap_f1.svg", "dot_heatmap_f1.png",
    }
    assert (tmp_path / "heatmap_f1.svg").is_file()
    assert (tmp_path / "heatmap_f1.png").is_file()
    assert (tmp_path / "dot_heatmap_f1.svg").is_file()
    assert (tmp_path / "dot_heatmap_f1.png").is_file()
