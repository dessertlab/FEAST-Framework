"""PR-AUC counterpart of the F1 mean-difference-CI report (Figure 4), addressing the
reviewer's request for a threshold-independent comparison alongside F1.

PR-AUC is computed from the continuous fusion score, so unlike F1 it does not depend on
tau at all: every tau-suffixed variant of a given base strategy (e.g. bks_tau_0_1 ...
bks_tau_0_9) has the identical PR-AUC (see analysis.fusion.predictions.binary_metrics --
pr_auc is a function of (labels, scores) only, never of the thresholded prediction). This
script therefore reuses, unchanged, the exact strategy set already nested-selected for F1
(the "strategy" column of each language's committed mean_difference_ci_f1.csv) rather than
re-running tau selection -- the PR-AUC value for e.g. "bks_tau_0_7" is exactly the PR-AUC
of plain "bks", whichever tau happens to be in the selected name.

Writes, per language, data/results/<lang>/pillar_child/full/plots/mean_difference_ci_pr_auc.csv
(+ .svg/.png forest plot), using the same Wilcoxon/Hodges-Lehmann machinery and per-family
pairing as the F1 report. Then regenerates the cross-language combined plot
(data/results/combined_ci_pr_auc_{sans,serif}.{png,svg}).

Requires each language's full-tier fusion_metrics_per_family.csv, config.json, and
mean_difference_ci_f1.csv to already exist (i.e. `main.py fusion --tier full
--calibration sensitivity,specificity` must have been run first).

    uv run python auxiliary/pr_auc_ci_report.py
    uv run python auxiliary/pr_auc_ci_report.py --lang java
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

from analysis.combined_plots import make_combined_plot
from analysis.mean_difference_ci import save_mean_difference_ci_report

METRIC = "pr_auc"


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
    # Reuse the exact strategy set (base + selected tau) already nested-selected for F1.
    selected_strategies = pd.read_csv(f1_ci_path)["strategy"].astype(str).tolist()

    intervals, paths = save_mean_difference_ci_report(
        per_family, results_dir / "plots", metric=METRIC, tools=tools,
        preselected_strategies=selected_strategies,
    )
    print(f"[{language}] wrote {[p.name for p in paths]}")
    intervals.insert(0, "language", language)
    return intervals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="all", choices=["c_cpp", "java", "python", "all"])
    ap.add_argument("--results-dir", type=str, default="data/results")
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    langs = ["c_cpp", "java", "python"] if args.lang == "all" else [args.lang]

    rows = [r for lang in langs if (r := run_lang(lang, results_root)) is not None]
    if not rows:
        print("\nNo data available -- see the messages above.")
        return

    all_intervals = pd.concat(rows, ignore_index=True)
    cols = ["language", "baseline", "strategy", "n_pairs", "n_nonzero_pairs", "mean_difference",
            "weighted_difference", "n_effective",
            "hodges_lehmann", "p_value", "ci_low", "ci_high", "status"]
    print("\n=== PR-AUC vs. 2ooN baseline (same test + pairing behind Figure 4, threshold-free) ===")
    print(all_intervals[[c for c in cols if c in all_intervals.columns]].to_string(index=False))

    if args.lang == "all":
        make_combined_plot(use_sans=False, results_root=str(results_root), metric=METRIC)
        make_combined_plot(use_sans=True, results_root=str(results_root), metric=METRIC)


if __name__ == "__main__":
    main()
