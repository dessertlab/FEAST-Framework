"""Summarises how much the per-fold tau choice (before aggregation) varies across the 5
outer folds, per fusion strategy and language -- transparency requested for reviewer
objection #2 ("the high variance of the threshold ... needs to be accounted for").

analysis.experiment._select_tau_nested records, for each of the 5 outer folds, the tau
that would have been chosen from that fold's own inner calibration-internal split, in
fusion_tau_selection_by_fold.csv -- before they get aggregated (mean over folds, then
argmax) into the single tau actually used for Figure 4. That raw per-fold record is never
otherwise summarised. This script turns it into one row per (language, strategy): the
final (aggregated) tau, the min/max/mode/spread of the 5 independent per-fold choices, and
how many of the 5 folds agree with the final tau -- so a reader can see at a glance
whether tau selection is stable or noisy, without re-deriving anything.

"Spread" is max(fold tau) - min(fold tau) on the tau grid (0.1 steps); a spread of 0.0
means all 5 folds independently picked the same tau.

Requires each language's full-tier fusion_tau_selection_by_fold.csv and
mean_difference_ci_f1.csv to already exist.

    uv run python auxiliary/tau_fold_variability.py
    uv run python auxiliary/tau_fold_variability.py --lang java
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.fusion.common import split_tau_strategy
from analysis.mean_difference_ci import strategy_label


def _mode(values: list[float]) -> float:
    """Most frequent value; ties broken by the smallest tau for determinism."""
    counts = Counter(values)
    best = max(counts.values())
    return min(v for v, n in counts.items() if n == best)


def run_lang(language: str, results_root: Path) -> pd.DataFrame | None:
    results_dir = results_root / language / "pillar_child" / "full"
    by_fold_path = results_dir / "fusion_tau_selection_by_fold.csv"
    f1_ci_path = results_dir / "plots" / "mean_difference_ci_f1.csv"

    missing = [p for p in (by_fold_path, f1_ci_path) if not p.exists()]
    if missing:
        print(f"[{language}] missing files -- run `main.py fusion --tier full "
              f"--calibration sensitivity,specificity` first:")
        for p in missing:
            print(f"  {p}")
        return None

    by_fold = pd.read_csv(by_fold_path)
    final_tau_by_base = {
        split_tau_strategy(str(s))[0]: split_tau_strategy(str(s))[1]
        for s in pd.read_csv(f1_ci_path)["strategy"]
        if split_tau_strategy(str(s))[1] is not None
    }

    rows = []
    for base, group in by_fold.groupby("base_strategy"):
        taus = group["tau"].astype(float).tolist()
        final_tau = final_tau_by_base.get(base)
        n_folds = len(taus)
        n_agree = sum(1 for t in taus if final_tau is not None and abs(t - final_tau) < 1e-9)
        rows.append({
            "language": language,
            "strategy": strategy_label(base),
            "base_strategy": base,
            "final_tau": final_tau,
            "n_folds": n_folds,
            "n_folds_agreeing_with_final": n_agree,
            "tau_min": min(taus),
            "tau_max": max(taus),
            "tau_mode": _mode(taus),
            "spread": round(max(taus) - min(taus), 2),
            "per_fold_taus": ",".join(f"{t:.1f}" for t in sorted(taus)),
        })

    out = pd.DataFrame(rows).sort_values("spread", ascending=False).reset_index(drop=True)
    out_path = results_dir / "plots" / "tau_fold_variability.csv"
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
    cols = ["language", "strategy", "final_tau", "tau_min", "tau_max", "tau_mode",
            "spread", "n_folds_agreeing_with_final", "per_fold_taus"]
    print("\n=== Per-fold tau variability (before aggregation into the final Figure-4 tau) ===")
    print(all_rows[cols].to_string(index=False))

    wide = all_rows[all_rows["spread"] >= 0.2]
    print(f"\n{len(wide)}/{len(all_rows)} (strategy, language) pairs have a spread >= 0.2 "
          f"(more than one grid step) across the 5 folds.")

    out_dir = ROOT / "data" / "results" / "_cross_language"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(out_dir / "tau_fold_variability.csv", index=False)
    print(f"\nSaved combined table to {out_dir / 'tau_fold_variability.csv'}")


if __name__ == "__main__":
    main()
