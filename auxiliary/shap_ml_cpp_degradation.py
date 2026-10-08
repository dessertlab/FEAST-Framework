"""SHAP analysis of the ML fusion strategies on C/C++ (Reviewer #2 Minor Issue #3).

The paper attributes the ML strategies' (Decision Tree, Random Forest, Gradient
Boosting, XGBoost) degradation on C/C++ loosely to "training data quality." The
reviewer asks for a SHAP analysis to identify which specific tool alerts are actually
misleading the models. This is cheap here: every model's only input is the binary
tool-fire vector, at most 6 features for C/C++, so shap.TreeExplainer runs in closed
form and near-instantly.

Reuses the EXACT hyperparameters and feature construction from analysis/fusion/ml.py
(imported directly, not reimplemented) so the explained models are the same ones the
paper actually reports on, not a parallel approximation.

For each C/C++ family in the "full" tier's restriction, for each classifier, the model
is fit on each fold's calibration split (5-fold CV, matching the main experiment) and
SHAP values are computed on that fold's held-out validation split; values are pooled
across folds. Reported per (family, classifier, tool):
  - mean(|SHAP|): overall importance of that tool's fire signal to the model's output.
  - mean(SHAP | true label = safe): the tool's average PUSH TOWARD "vulnerable" on rows
    that are actually safe -- a large positive value here identifies a tool whose firing
    is specifically misleading the model into false positives on C/C++.
  - mean(SHAP | true label = vulnerable): same, on genuinely vulnerable rows (a tool
    correctly pushing toward "vulnerable" here is doing its job; comparing the two rows
    tells you whether a tool is broadly unreliable or specifically noisy on safe code).

Requires data/enriched/c_cpp.parquet and data/cwec_latest.xml (neither ships in the git
repo), plus `shap` (add via `uv add shap` if not already synced).

    uv run python auxiliary/shap_ml_cpp_degradation.py
    uv run python auxiliary/shap_ml_cpp_degradation.py --classifiers random_forest,xgboost
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.experiment import prepare_canonical
from analysis.fusion.common import build_fire_index, precompute_labels
from analysis.fusion.ml import _balanced_sample_weight, _build_feature_matrix
from analysis.folds import split_train_validation

ALL_CLASSIFIERS = ("decision_tree", "random_forest", "gradient_boosting", "xgboost")


def _make_classifier(name: str, seed: int):
    if name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(
            criterion="gini", max_depth=2, min_samples_leaf=3, min_samples_split=5,
            class_weight="balanced", random_state=seed,
        ), False
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=100, max_features=2, max_depth=3, min_samples_leaf=3,
            min_samples_split=5, bootstrap=True, class_weight="balanced", random_state=seed,
        ), False
    if name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.1, max_depth=2, max_features=2,
            min_samples_leaf=3, min_samples_split=5, subsample=0.8, random_state=seed,
        ), True
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=50, learning_rate=0.1, max_depth=2, subsample=0.8,
            colsample_bytree=1.0, reg_lambda=1.0, min_child_weight=3,
            objective="binary:logistic", eval_metric="logloss", random_state=seed, n_jobs=1,
        ), True
    raise ValueError(f"unknown classifier {name!r}")


def _shap_values_for_positive_class(explainer, x: np.ndarray) -> np.ndarray:
    """Normalise shap's inconsistent return shape across sklearn/xgboost + shap versions
    to a single (n_samples, n_features) array of SHAP values for the positive class."""
    raw = explainer.shap_values(x)
    if isinstance(raw, list):
        return np.asarray(raw[1] if len(raw) > 1 else raw[0])
    raw = np.asarray(raw)
    if raw.ndim == 3:  # (n_samples, n_features, n_classes)
        return raw[:, :, -1]
    return raw


def run(lang: str, classifiers: list[str], tier: str, seed: int) -> pd.DataFrame:
    import shap

    from analysis.fusion import TIER_MIN_COUNT
    min_cwe_count = TIER_MIN_COUNT.get(tier, 100)
    prep = prepare_canonical(lang, "pillar_child", n_splits=5, min_cwe_count=min_cwe_count, seed=seed)
    cdf, tools, families, folds = prep.cdf, prep.tools, prep.families, prep.folds
    if not families:
        print(f"[{lang}] no families survive the '{tier}' tier restriction.")
        return pd.DataFrame()

    rows = []
    for clf_name in classifiers:
        use_sample_weight = _make_classifier(clf_name, seed)[1]
        for family in families:
            pooled_shap: list[np.ndarray] = []
            pooled_label: list[np.ndarray] = []
            for fold in sorted(folds["fold"].unique()):
                cal_df, val_df = split_train_validation(cdf, folds, validation_fold=fold)
                cal_fire, _ = build_fire_index(cal_df, tools)
                val_fire, _ = build_fire_index(val_df, tools)
                cal_labels = precompute_labels(cal_df, families)
                val_labels = precompute_labels(val_df, families)

                x_cal = _build_feature_matrix(cal_fire, len(cal_df), tools, family)
                y_cal = np.array([int(cal_labels[(i, family)]) for i in range(len(cal_df))])
                x_val = _build_feature_matrix(val_fire, len(val_df), tools, family)
                y_val = np.array([int(val_labels[(i, family)]) for i in range(len(val_df))])
                if len(np.unique(y_cal)) < 2 or len(x_val) == 0:
                    continue

                clf, _ = _make_classifier(clf_name, seed)
                if use_sample_weight:
                    clf.fit(x_cal, y_cal, sample_weight=_balanced_sample_weight(y_cal))
                else:
                    clf.fit(x_cal, y_cal)

                explainer = shap.TreeExplainer(clf)
                sv = _shap_values_for_positive_class(explainer, x_val)
                pooled_shap.append(sv)
                pooled_label.append(y_val)

            if not pooled_shap:
                continue
            sv = np.concatenate(pooled_shap, axis=0)
            yv = np.concatenate(pooled_label, axis=0)
            for j, tool in enumerate(tools):
                col = sv[:, j]
                rows.append({
                    "language": lang, "classifier": clf_name, "family": family, "tool": tool,
                    "mean_abs_shap": float(np.mean(np.abs(col))),
                    "mean_shap_on_safe": float(col[yv == 0].mean()) if (yv == 0).any() else None,
                    "mean_shap_on_vuln": float(col[yv == 1].mean()) if (yv == 1).any() else None,
                    "n_val_rows": len(col),
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="c_cpp")
    ap.add_argument("--classifiers", default=",".join(ALL_CLASSIFIERS))
    ap.add_argument("--tier", default="full", choices=["base", "medium", "full"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    classifiers = [c.strip() for c in args.classifiers.split(",") if c.strip()]
    result = run(args.lang, classifiers, args.tier, args.seed)
    if result.empty:
        print("\nNo data available (or shap not installed) -- nothing to report.")
        return

    print("\n=== Mean |SHAP| per (classifier, tool), aggregated across families (support-weighted) ===")
    weights = result.groupby("family")["n_val_rows"].transform("first")
    result["_w"] = weights
    agg = (
        result.groupby(["classifier", "tool"])
        .apply(lambda g: pd.Series({
            "mean_abs_shap_weighted": np.average(g["mean_abs_shap"], weights=g["_w"]),
            "mean_shap_on_safe_weighted": np.average(g["mean_shap_on_safe"].fillna(0), weights=g["_w"]),
        }))
        .reset_index()
    )
    print(agg.sort_values(["classifier", "mean_abs_shap_weighted"], ascending=[True, False]).to_string(index=False))

    out_path = ROOT / "data" / "results" / "_cross_language" / f"shap_{args.lang}_per_family.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.drop(columns="_w").to_csv(out_path, index=False)
    print(f"\nSaved per-family breakdown to {out_path}")
    print(
        "\nmean_shap_on_safe_weighted is the quantity of interest for Reviewer #2's "
        "Minor Issue #3: the tool(s) with the largest positive value are pushing the "
        "model toward false positives on genuinely safe C/C++ code."
    )


if __name__ == "__main__":
    main()
