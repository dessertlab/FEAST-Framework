"""Compares three CWE matching schemes through the K-of-N voting baseline only.

Motivation: Reviewer #2's Major Concern #4 argues that pillar_child normalisation
(collapsing raw CWEs to a common family before matching) lowers the detection bar by
crediting a tool for a family-level "hit" even when its specific raw CWE is unrelated to
the ground truth's (a "cousin" match -- see auxiliary/cwe_family_conflation_ablation.py).
This script measures the practical impact of that choice through the simplest, least
adaptive fusion strategy available (K-of-N voting), which has no learned weights that
could otherwise absorb or obscure the effect of the matching-rule choice -- so it isolates
the variable of interest more cleanly than a calibrated strategy would. This answers "does
matching-rule choice matter for voting", not "for every fusion strategy" -- a scope that
should be stated explicitly wherever these numbers are used.

Three matching schemes, nested by construction (exact subset-of exact_or_direct_child
subset-of family):
  - "exact"                 : a tool's fire counts toward ground-truth family c only if
                               its specific raw CWE literally equals one of the row's
                               ground-truth raw CWEs.
  - "exact_or_direct_child" : the reviewer's own suggested control condition, verbatim --
                               "exactly matches the ground-truth CWE (or a direct child),
                               without parent absorption." Directional and distance-limited
                               (see _is_direct_child): only a tool CWE that is MORE SPECIFIC
                               than the ground truth by exactly one primary-path level
                               counts in addition to exact; a tool reporting the ground
                               truth's parent does not count.
  - "family"                : the paper's current method -- any raw CWE that canonicalises
                               to family c counts, regardless of its relation to the
                               specific ground-truth CWE.
For safe (label=0) rows there is no ground-truth CWE to match against, so all three
schemes behave identically there (a fire is a fire, full stop); they can only differ on
vulnerable rows.

(An earlier version of this script also carried a "vertical" scheme -- symmetric,
unbounded-distance ancestor/descendant matching -- dropped per the authors' decision.
"exact_or_direct_child" is narrower and directional by design, matching the reviewer's
literal wording rather than the broader vertical-closure notion.)

Study population: restricted to "tool-disagreement" rows -- rows where the union of raw
CWEs reported across all tool columns has at least 2 distinct values. This is the only
subpopulation where the three schemes can plausibly disagree; including agreement rows
would just dilute the comparison with cases where the answer is identical regardless of
scheme.

Requires data/enriched/<lang>.parquet and data/cwec_latest.xml (neither ships in the
git repo). Uses the same restricted family list as the main analysis for comparability,
and the paper's own K=2 voting baseline (2ooN for every language).

    uv run python auxiliary/normalization_scheme_voting_ablation.py --lang c_cpp
    uv run python auxiliary/normalization_scheme_voting_ablation.py --lang all --threshold 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.aggregation import REPORT_METRICS, support_weighted_over_families
from analysis.canonical import CweCanonicalizer
from analysis.dataset import as_list, detect_tool_columns
from analysis.fusion.predictions import binary_metrics

SCHEMES = ("exact", "exact_or_direct_child", "family")


def _is_direct_child(tool_cwe: str, gt_raw: set[str], canon: CweCanonicalizer) -> bool:
    """True iff tool_cwe is a direct (one-level) child of some ground-truth CWE.

    This is the reviewer's own suggested control condition, verbatim: "exactly matches
    the ground-truth CWE (or a direct child), without parent absorption." It is
    directional and distance-limited, unlike the (removed) "vertical" scheme: only a
    tool that is MORE SPECIFIC than the ground truth by exactly one primary-path level
    counts. A tool reporting the ground truth's own parent (i.e. a more generic CWE)
    does NOT count -- that is exactly the "parent absorption" the reviewer rules out.
    """
    parent = canon.nav.primary_parent(tool_cwe)
    if parent is None:
        return False
    norm = canon.nav._num
    parent = norm(parent)
    return any(norm(g) == parent for g in gt_raw)


def _is_vertical(tool_cwe: str, gt_raw: set[str], canon: CweCanonicalizer) -> bool:
    # CWENavigator normalises everything internally to bare numeric strings ("120", no
    # "CWE-" prefix -- see CWENavigator._num), while every raw CWE elsewhere in this
    # script (the "cwes"/tool columns) uses the "CWE-120" form. primary_path() returns
    # the bare form, so comparisons must go through the same normaliser on both sides,
    # or every "in" check below silently and permanently fails.
    norm = canon.nav._num
    t_num = norm(tool_cwe)
    t_path = {norm(x) for x in canon.nav.primary_path(tool_cwe)}
    for g in gt_raw:
        if norm(g) in t_path or t_num in {norm(x) for x in canon.nav.primary_path(g)}:
            return True
    return False


def _fired_families(
    raw_cwes: list[str],
    gt_raw: set[str],
    label: int,
    scheme: str,
    canon: CweCanonicalizer,
    allowed_families: set[str],
) -> set[str]:
    """Families this tool's raw output fires under one matching scheme, for one row."""
    fired: set[str] = set()
    for c in raw_cwes:
        fam = canon.family(c)
        if fam is None or fam not in allowed_families:
            continue
        if label == 0 or scheme == "family":
            fired.add(fam)
        elif scheme == "exact":
            if c in gt_raw:
                fired.add(fam)
        elif scheme == "exact_or_direct_child":
            if c in gt_raw or _is_direct_child(c, gt_raw, canon):
                fired.add(fam)
        elif scheme == "vertical":
            if c in gt_raw or _is_vertical(c, gt_raw, canon):
                fired.add(fam)
    return fired


def run_lang(lang: str, threshold: int, xml_path: Path) -> dict[str, pd.DataFrame]:
    enriched_path = ROOT / "data" / "enriched" / f"{lang}.parquet"
    if not enriched_path.exists():
        print(f"[{lang}] missing {enriched_path} -- run `main.py enrich` first. Skipping.")
        return {}

    df = pd.read_parquet(enriched_path)
    tool_columns = detect_tool_columns(df)
    n_tools = len(tool_columns)
    canon = CweCanonicalizer.from_xml(level="pillar_child", cwe_xml_path=str(xml_path))

    results_family_path = ROOT / "data" / "results" / lang / "pillar_child" / "full" / "fusion_metrics_per_family.csv"
    if not results_family_path.exists():
        results_family_path = ROOT / "data" / "results" / lang / "pillar_child" / "base" / "fusion_metrics_per_family.csv"
    allowed_families = set(pd.read_csv(results_family_path)["family"].unique()) if results_family_path.exists() else None

    # Restrict to tool-disagreement rows: >=2 distinct raw CWEs across all tool columns.
    disagreement_mask = []
    per_row_tool_raw: list[dict[str, list[str]]] = []
    per_row_gt_raw: list[set[str]] = []
    for _, r in df.iterrows():
        tool_raw = {t: as_list(r[t]) for t in tool_columns}
        per_row_tool_raw.append(tool_raw)
        per_row_gt_raw.append(set(as_list(r["cwes"])))
        distinct = {c for lst in tool_raw.values() for c in lst}
        disagreement_mask.append(len(distinct) >= 2)

    df = df.reset_index(drop=True)
    keep_idx = [i for i, keep in enumerate(disagreement_mask) if keep]
    print(f"[{lang}] {len(keep_idx)}/{len(df)} rows kept (tool-disagreement subpopulation)")
    if not keep_idx:
        return {}

    # GT family set per kept row, restricted to the same family population as the main analysis.
    gt_family_sets = []
    for i in keep_idx:
        fams = {canon.family(c) for c in per_row_gt_raw[i]}
        fams.discard(None)
        if allowed_families is not None:
            fams &= allowed_families
        gt_family_sets.append(fams)
    families = sorted(allowed_families) if allowed_families is not None else sorted({f for s in gt_family_sets for f in s})

    per_scheme_rows: dict[str, list[dict]] = {s: [] for s in SCHEMES}
    for scheme in SCHEMES:
        for pos, i in enumerate(keep_idx):
            label_i = int(df.at[i, "label"])
            gt_raw_i = per_row_gt_raw[i]
            fired_by_tool = {
                t: _fired_families(per_row_tool_raw[i][t], gt_raw_i, label_i, scheme, canon, set(families))
                for t in tool_columns
            }
            gt_fams = gt_family_sets[pos]
            for fam in families:
                fire_count = sum(1 for t in tool_columns if fam in fired_by_tool[t])
                per_scheme_rows[scheme].append({
                    "family": fam,
                    "label": fam in gt_fams,
                    "prediction": fire_count >= threshold,
                    "score": fire_count / n_tools if n_tools else 0.0,
                })

    per_family_by_scheme = {}
    for scheme in SCHEMES:
        rows = pd.DataFrame(per_scheme_rows[scheme])
        recs = []
        for fam, g in rows.groupby("family"):
            m = binary_metrics(g["label"].tolist(), g["prediction"].tolist(), g["score"].tolist())
            m["family"] = fam
            recs.append(m)
        per_family_by_scheme[scheme] = pd.DataFrame(recs)
    return per_family_by_scheme


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="all", choices=["c_cpp", "java", "python", "all"])
    ap.add_argument("--threshold", type=int, default=2, help="K for K-of-N voting (paper uses 2 for every language)")
    ap.add_argument("--cwe-xml", default=str(ROOT / "data" / "cwec_latest.xml"))
    args = ap.parse_args()

    langs = ["c_cpp", "java", "python"] if args.lang == "all" else [args.lang]
    headline_rows = []
    for lang in langs:
        per_family_by_scheme = run_lang(lang, args.threshold, Path(args.cwe_xml))
        for scheme, per_family in per_family_by_scheme.items():
            if per_family.empty:
                continue
            per_family = per_family.assign(strategy=scheme)
            overall = support_weighted_over_families(per_family, metric_columns=REPORT_METRICS)
            row = {"language": lang, "scheme": scheme}
            for metric in ("f1", "pr_auc"):
                row[metric] = overall.iloc[0].get(f"{metric}_weighted")
            headline_rows.append(row)

            out_path = ROOT / "data" / "results" / "_cross_language" / f"normalization_scheme_{lang}_{scheme}_per_family.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            per_family.to_csv(out_path, index=False)

    if not headline_rows:
        print("\nNo data available -- see the missing-file messages above.")
        return

    headline = pd.DataFrame(headline_rows)
    print("\n=== Voting (K-of-N) headline metrics per language and matching scheme ===")
    print(headline.to_string(index=False))
    out_path = ROOT / "data" / "results" / "_cross_language" / "normalization_scheme_voting_headline.csv"
    headline.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
