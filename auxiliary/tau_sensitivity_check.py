"""Tau sensitivity check: does the F1-vs-baseline conclusion (Figure 4's status column)
survive perturbing the nested-selected tau by one grid step (+/- 0.1)?

Motivation (reviewer objection #2): the 95% CIs in Figure 4 are computed treating the
nested-selected tau as fixed and known; they do not propagate the uncertainty of the
tau-selection step itself. Re-deriving that uncertainty formally would require repeating
the whole nested procedure under resampling -- expensive and not attempted here. This
script instead answers a narrower, cheap question: if tau had landed one grid step away
(the finest resolution used anywhere in the tau sweep, tau in {0.1, ..., 0.9}), would the
reported significance status (improvement / degradation / inconclusive) have changed?
Every tau variant's F1 is already materialised in fusion_metrics_per_family.csv (see
analysis.fusion.predictions.expand_tau_variants), so this reuses mean_difference_ci with a
different fixed preselected strategy -- no nested selection, no re-fitting, is re-run.

A strategy at the edge of the grid (tau=0.1 or tau=0.9) only has one neighbour to check.

Writes, per language, data/results/<lang>/pillar_child/full/plots/tau_sensitivity_f1.csv
and a combined data/results/_cross_language/tau_sensitivity_f1.csv across all requested
languages, flagging any strategy whose status is not identical at tau-0.1, tau, tau+0.1.

Requires each language's full-tier fusion_metrics_per_family.csv, config.json, and
mean_difference_ci_f1.csv to already exist.

    uv run python auxiliary/tau_sensitivity_check.py
    uv run python auxiliary/tau_sensitivity_check.py --lang java
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.fusion.common import DEFAULT_TAUS, split_tau_strategy, tau_suffix
from analysis.mean_difference_ci import mean_difference_ci, strategy_label

TAU_GRID = set(round(t, 1) for t in DEFAULT_TAUS)
STEP = 0.1


def neighbor_taus(tau: float) -> list[float]:
    """Grid-adjacent taus that actually exist (edges of the grid have only one)."""
    candidates = [round(tau - STEP, 1), round(tau + STEP, 1)]
    return [t for t in candidates if t in TAU_GRID]


def run_lang(language: str, results_root: Path) -> pd.DataFrame | None:
    results_dir = results_root / language / "pillar_child" / "full"
    config_path = results_dir / "config.json"
    per_family_path = results_dir / "fusion_metrics_per_family.csv"
    f1_ci_path = results_dir / "plots" / "mean_difference_ci_f1.csv"

    missing = [p for p in (config_path, per_family_path, f1_ci_path) if not p.exists()]
    if missing:
        print(f"[{language}] missing files -- run `main.py fusion --tier full "
              f"--calibration sensitivity,specificity` first:")
        for p in missing:
            print(f"  {p}")
        return None

    tools = json.loads(config_path.read_text(encoding="utf-8")).get("tools", [])
    per_family = pd.read_csv(per_family_path)
    selected = pd.read_csv(f1_ci_path)

    plan = []
    neighbor_names: list[str] = []
    for _, r in selected.iterrows():
        strategy = str(r["strategy"])
        base, tau = split_tau_strategy(strategy)
        if tau is None:
            continue  # baselines / strategies with no tau sweep -- nothing to perturb
        taus = neighbor_taus(tau)
        names = [f"{base}_{tau_suffix(t)}" for t in taus]
        neighbor_names.extend(names)
        plan.append({
            "base": base, "tau_selected": tau, "status_selected": r["status"],
            "mean_difference_selected": r["mean_difference"], "p_value_selected": r["p_value"],
            "neighbors": list(zip(taus, names)),
        })

    if not plan:
        print(f"[{language}] no tau-swept strategies found in {f1_ci_path}")
        return None

    neighbor_ci = mean_difference_ci(
        per_family, metric="f1", tools=tools, preselected_strategies=neighbor_names,
    )
    ci_by_strategy = {str(row["strategy"]): row for _, row in neighbor_ci.iterrows()}

    rows = []
    for item in plan:
        record = {
            "language": language,
            "strategy": strategy_label(item["base"]),
            "base_strategy": item["base"],
            "tau_minus": None, "status_minus": None, "mean_difference_minus": None,
            "tau_selected": item["tau_selected"],
            "status_selected": item["status_selected"],
            "mean_difference_selected": item["mean_difference_selected"],
            "p_value_selected": item["p_value_selected"],
            "tau_plus": None, "status_plus": None, "mean_difference_plus": None,
        }
        for t, name in item["neighbors"]:
            row = ci_by_strategy.get(name)
            if row is None:
                continue
            side = "minus" if t < item["tau_selected"] else "plus"
            record[f"tau_{side}"] = t
            record[f"status_{side}"] = row["status"]
            record[f"mean_difference_{side}"] = row["mean_difference"]
        statuses = {record["status_selected"], record["status_minus"], record["status_plus"]}
        statuses.discard(None)
        record["stable"] = len(statuses) == 1
        rows.append(record)

    out = pd.DataFrame(rows)
    out_path = results_dir / "plots" / "tau_sensitivity_f1.csv"
    out.to_csv(out_path, index=False)
    print(f"[{language}] wrote {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="all", choices=["c_cpp", "java", "python", "all"])
    ap.add_argument("--results-dir", type=str, default="data/results")
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    langs = ["c_cpp", "java", "python"] if args.lang == "all" else [args.lang]

    frames = [r for lang in langs if (r := run_lang(lang, results_root)) is not None]
    if not frames:
        print("\nNo data available -- see the messages above.")
        return

    all_rows = pd.concat(frames, ignore_index=True)
    cols = ["language", "strategy", "tau_minus", "status_minus", "tau_selected",
            "status_selected", "tau_plus", "status_plus", "stable"]
    print("\n=== Tau sensitivity: status at tau-0.1 / tau / tau+0.1 ===")
    print(all_rows[cols].to_string(index=False))

    unstable = all_rows[~all_rows["stable"]]
    print(f"\n{len(unstable)}/{len(all_rows)} (strategy, language) pairs change status "
          f"when tau is perturbed by one grid step.")
    if not unstable.empty:
        print(unstable[["language", "strategy", "status_minus", "status_selected", "status_plus"]].to_string(index=False))

    out_dir = ROOT / "data" / "results" / "_cross_language"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(out_dir / "tau_sensitivity_f1.csv", index=False)
    print(f"\nSaved combined table to {out_dir / 'tau_sensitivity_f1.csv'}")


if __name__ == "__main__":
    main()
