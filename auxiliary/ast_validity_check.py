"""Syntactic-validity check on materialised samples: function-level vs. snippet-level.

Reviewer #2's Major Concern #1 challenges whether FEAST's samples are analysable
fragments or complete units. Separately, we verified by reading every ingestion/*.py
module that FEAST itself never slices code by a fixed line/character window or
diff-hunk context -- every extractor either reads a whole upstream file or passes a
single upstream column through verbatim (see the per-module audit already done for
this rebuttal). What that audit could NOT settle is whether the upstream dataset's own
extraction actually produced a syntactically complete function for every sample that
survives into data/enriched/.

This script settles that empirically, on the exact samples used by the paper (the
"code" column of data/enriched/<lang>.parquet, i.e. after Stage 2/3/4 filtering,
dedup, and enrichment -- not a fresh read of the raw upstream files), by parsing each
sample and checking for a syntax error:

  - Python:      ast.parse() (stdlib, no extra dependency). A sample is syntactically
                 valid iff it parses at all; top-level FunctionDef/AsyncFunctionDef
                 count characterises "is it (approximately) one function".
  - C/C++:       tree-sitter-cpp. tree-sitter is error-tolerant by design (it targets
                 editors that must parse code while it's being typed), so it can parse
                 an incomplete file without crashing -- but it marks the malformed
                 region with an ERROR node. root_node.has_error is therefore a strong,
                 conservative validity signal: a genuinely truncated snippet (cut off
                 mid-statement or mid-block) will almost always trip it, while a
                 complete function with unresolved #include-d types will NOT (missing
                 symbols are a semantic/linkage concern, not a syntax error -- syntax
                 validity is exactly the property a SAT's parser front-end needs).
  - Java:        FEAST's own README notes Java samples are "function bodies... not
                 compilable top-level classes" -- bare methods are not valid top-level
                 Java syntax on their own, so each sample is wrapped in a throwaway
                 `class __Wrapper__ { ... }` before parsing with tree-sitter-java. The
                 same has_error check applies to the wrapped tree.

Reports, per (language, source) -- "source" is the dataset column already in
data/enriched (PrimeVul, CVEfixes, Juliet, ...), matching the paper's own Table 5
breakdown -- the percentage of syntactically valid samples and the distribution of
top-level function counts, so a dataset that is NOT function-level (multi-function
files, or genuine truncation) shows up as a low validity rate or a top-level-function
count far from 1, not just an aggregate language-wide number.

Requires data/enriched/<lang>.parquet (already used by the other rebuttal scripts) and
tree-sitter + tree-sitter-c/-cpp/-java for the non-Python languages (add via `uv sync`
if not already installed; Python needs no extra dependency).

    uv run python auxiliary/ast_validity_check.py --lang c_cpp
    uv run python auxiliary/ast_validity_check.py --lang all
"""

from __future__ import annotations

import argparse
import ast as pyast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check_python(code: str) -> tuple[bool, int | None, str | None, str]:
    try:
        tree = pyast.parse(code)
    except SyntaxError as exc:
        return False, None, str(exc), "invalid"
    n_funcs = sum(1 for node in tree.body if isinstance(node, (pyast.FunctionDef, pyast.AsyncFunctionDef)))
    return True, n_funcs, None, "module"


_CPP_LANG = None
_JAVA_LANG = None


def _cpp_parser():
    global _CPP_LANG
    from tree_sitter import Language, Parser
    import tree_sitter_cpp as tscpp
    if _CPP_LANG is None:
        _CPP_LANG = Language(tscpp.language())
    return Parser(_CPP_LANG)


def _java_parser():
    global _JAVA_LANG
    from tree_sitter import Language, Parser
    import tree_sitter_java as tsjava
    if _JAVA_LANG is None:
        _JAVA_LANG = Language(tsjava.language())
    return Parser(_JAVA_LANG)


def _count_named(node, type_name: str) -> int:
    return sum(1 for child in node.children if child.type == type_name)


def _check_cpp(code: str, parser) -> tuple[bool, int | None, str | None, str]:
    tree = parser.parse(code.encode("utf-8", errors="replace"))
    root = tree.root_node
    valid = not root.has_error
    n_funcs = _count_named(root, "function_definition")
    return valid, n_funcs, (None if valid else "tree-sitter ERROR node present"), "compilation_unit"


def _count_method_declarations(node) -> int:
    """Recursive count of method_declaration nodes anywhere under node."""
    total = 1 if node.type == "method_declaration" else 0
    for child in node.children:
        total += _count_method_declarations(child)
    return total


def _check_java(code: str, parser) -> tuple[bool, int | None, str | None, str]:
    """Try the sample as a standalone compilation unit first (whole file with its own
    package/import/class declaration -- Juliet/OWASP/CrossVul(Java) all extract whole
    files, per the ingestion audit), and only fall back to wrapping it as a bare class
    member if that fails. Wrapping a whole file's package/import statements inside
    another class is always a syntax error regardless of the sample's own validity, so
    trying unwrapped first is required, not optional -- an all-wrapped strategy silently
    misclassifies every whole-file source as 100% invalid.
    """
    tree = parser.parse(code.encode("utf-8", errors="replace"))
    root = tree.root_node
    if not root.has_error:
        return True, _count_method_declarations(root), None, "compilation_unit"

    wrapped = f"class __Wrapper__ {{\n{code}\n}}"
    tree2 = parser.parse(wrapped.encode("utf-8", errors="replace"))
    root2 = tree2.root_node
    if not root2.has_error:
        return True, _count_method_declarations(root2), None, "wrapped_member"

    return False, None, "tree-sitter ERROR node present (both as compilation unit and wrapped)", "invalid"


def _apply_support_filter(df: pd.DataFrame, min_cwe_count: int, xml_path: Path) -> pd.DataFrame:
    """Reproduce the paper's Sec. 4.2 "Support" filter: drop vulnerable rows whose CWEs
    all belong to canonical families with fewer than `min_cwe_count` GT occurrences.
    Safe rows are never dropped. Identical logic to auxiliary/check_experiment_counts.py,
    which is what established that this filter is exactly what turns the raw
    data/enriched/merged counts into the paper's published Table 5 counts.
    """
    from collections import Counter

    from analysis.canonical import CweCanonicalizer
    from analysis.dataset import as_list

    canon = CweCanonicalizer.from_xml(level="pillar_child", cwe_xml_path=str(xml_path))
    families = [canon.families(as_list(cwes)) for cwes in df["cwes"]]
    df = df.assign(_families=families)

    family_counts = Counter()
    for fams in df.loc[df["label"] == 1, "_families"]:
        family_counts.update(fams)
    frequent = {fam for fam, count in family_counts.items() if count >= min_cwe_count}

    keep = (df["label"] == 0) | df["_families"].map(lambda fams: bool(set(fams) & frequent))
    return df[keep].drop(columns="_families").reset_index(drop=True)


def run_lang(lang: str, min_cwe_count: int | None, xml_path: Path) -> pd.DataFrame:
    enriched_path = ROOT / "data" / "enriched" / f"{lang}.parquet"
    if not enriched_path.exists():
        print(f"[{lang}] missing {enriched_path} -- run `main.py enrich` first. Skipping.")
        return pd.DataFrame()

    cols = ["code", "source", "label", "cwes"] if min_cwe_count is not None else ["code", "source"]
    df = pd.read_parquet(enriched_path)
    df = df[[c for c in cols if c in df.columns]]
    if "source" not in df.columns:
        df = df.assign(source="unknown")

    if min_cwe_count is not None:
        n_before = len(df)
        df = _apply_support_filter(df, min_cwe_count, xml_path)
        print(f"[{lang}] Support filter (>= {min_cwe_count} GT occurrences): {n_before} -> {len(df)} rows")
    df = df[["code", "source"]]

    if lang == "python":
        checker = lambda code: _check_python(code)
    elif lang in ("c_cpp", "c", "cpp"):
        parser = _cpp_parser()
        checker = lambda code: _check_cpp(code, parser)
    elif lang == "java":
        parser = _java_parser()
        checker = lambda code: _check_java(code, parser)
    else:
        raise ValueError(f"unsupported lang {lang!r}")

    rows = []
    for _, r in df.iterrows():
        valid, n_funcs, err, mode = checker(str(r["code"]))
        rows.append({
            "language": lang, "source": r["source"], "valid": valid,
            "n_top_level_functions": n_funcs, "parse_mode": mode,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="all", help="comma-separated: c_cpp,java,python or 'all'")
    ap.add_argument(
        "--min-cwe-count", type=int, default=100,
        help="Sec. 4.2 Support filter threshold (paper default: 100); pass 0 to disable and use the raw enriched population",
    )
    ap.add_argument("--cwe-xml", default=str(ROOT / "data" / "cwec_latest.xml"))
    args = ap.parse_args()
    langs = ["c_cpp", "java", "python"] if args.lang == "all" else [s.strip() for s in args.lang.split(",")]
    min_cwe_count = None if args.min_cwe_count <= 0 else args.min_cwe_count

    frames = []
    for lang in langs:
        try:
            frames.append(run_lang(lang, min_cwe_count, Path(args.cwe_xml)))
        except ImportError as exc:
            print(f"[{lang}] missing dependency -- {exc}. Run `uv sync` to install tree-sitter grammars.")

    result = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if result.empty:
        print("\nNo data available -- see the messages above.")
        return

    print("\n=== Syntactic validity per (language, source) ===")
    summary = (
        result.groupby(["language", "source"])
        .agg(
            n_samples=("valid", "size"),
            pct_valid=("valid", "mean"),
            mean_top_level_functions=("n_top_level_functions", "mean"),
            pct_exactly_one_function=("n_top_level_functions", lambda s: (s == 1).mean()),
            pct_compilation_unit=("parse_mode", lambda s: (s == "compilation_unit").mean()),
            pct_wrapped_member=("parse_mode", lambda s: (s == "wrapped_member").mean()),
        )
        .reset_index()
    )
    summary["pct_valid"] = (summary["pct_valid"] * 100).round(2)
    summary["pct_exactly_one_function"] = (summary["pct_exactly_one_function"] * 100).round(2)
    summary["pct_compilation_unit"] = (summary["pct_compilation_unit"] * 100).round(2)
    summary["pct_wrapped_member"] = (summary["pct_wrapped_member"] * 100).round(2)
    print(summary.sort_values(["language", "pct_valid"]).to_string(index=False))

    print("\n=== Overall per language ===")
    overall = (
        result.groupby("language")
        .agg(n_samples=("valid", "size"), pct_valid=("valid", "mean"))
        .reset_index()
    )
    overall["pct_valid"] = (overall["pct_valid"] * 100).round(2)
    print(overall.to_string(index=False))

    out_dir = ROOT / "data" / "results" / "_cross_language"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "ast_validity_by_source.csv", index=False)
    result.to_csv(out_dir / "ast_validity_per_sample.csv", index=False)
    print(f"\nSaved to {out_dir / 'ast_validity_by_source.csv'} and ast_validity_per_sample.csv")


if __name__ == "__main__":
    main()
