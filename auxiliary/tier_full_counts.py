"""
tier_full_counts.py
-------------------
Compute per-source (dataset) vuln / safe sample counts **after** the tier-full
restriction is applied, replicating exactly what experiment.py does.

Logic (mirrors prepare_canonical() in analysis/experiment.py):
  1. Load data/enriched/<lang>.parquet  (already SAT-enriched, PrimeVul downsampled)
  2. Canonicalise the GT `cwes` column to pillar_child families via CweCanonicalizer
  3. Compute GT family support = #rows with label==1 that contain each family
  4. Keep families with support >= TIER_FULL_MIN  (100)
  5. A vuln row (label==1) is KEPT iff at least one of its canonical families is frequent
  6. A safe row (label==0) is KEPT iff it belongs to a source that has at least one kept
     vuln row  (the source is part of the experiment)
  7. Count kept vuln and safe rows per source, across all languages

Run from the FEAST project root:
    uv run python auxiliary/tier_full_counts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make sure the project root is on sys.path so we can import from analysis/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.canonical import CweCanonicalizer, CANONICAL_LEVEL
from analysis.dataset import load_enriched, as_list

TIER_FULL_MIN = 100
LANGUAGES = ["c_cpp", "java", "python"]
CWE_XML = ROOT / "data" / "cwec_latest.xml"
ENRICHED_DIR = ROOT / "data" / "enriched"
OUT_CSV = ROOT / "auxiliary" / "tier_full_counts.csv"


def canonicalise_cwes(cwes_series: pd.Series, canon: CweCanonicalizer) -> pd.Series:
    """Map each cell (list of raw CWEs) to a list of canonical families (deduped)."""
    def _map(raw_list):
        families = {canon.family(cwe) for cwe in as_list(raw_list)}
        return sorted(f for f in families if f is not None)
    return cwes_series.map(_map)


def process_language(language: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  Language: {language}")
    print(f"{'='*60}")

    # 1. Load enriched dataset
    dataset = load_enriched(language, enriched_dir=ENRICHED_DIR)
    df = dataset.to_pandas().reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows from enriched/{language}.parquet")

    # 2. Canonicalise GT CWEs
    canon = CweCanonicalizer.from_xml(CANONICAL_LEVEL, str(CWE_XML))
    df["canonical_cwes"] = canonicalise_cwes(df["cwes"], canon)

    # 3. Compute GT family support (count distinct rows per family, label==1 only)
    from collections import Counter
    support: Counter = Counter()
    vuln_mask = df["label"] == 1
    for families in df.loc[vuln_mask, "canonical_cwes"]:
        support.update(set(families))  # each family counted once per row

    frequent_families = {f for f, cnt in support.items() if cnt >= TIER_FULL_MIN}
    print(f"  Families with support >= {TIER_FULL_MIN}: {len(frequent_families)}")
    print(f"  (total families with any support: {len(support)})")

    # 4. Mark which rows are kept
    #    Vuln rows: kept if at least one canonical family is frequent
    def vuln_kept(families: list) -> bool:
        return bool(set(families) & frequent_families)

    df["vuln_kept"] = False
    df.loc[vuln_mask, "vuln_kept"] = df.loc[vuln_mask, "canonical_cwes"].map(vuln_kept)

    # 5. Identify which sources have at least one kept vuln row
    #    Safe rows from those sources are included in the experiment
    sources_with_kept_vuln = set(df.loc[df["vuln_kept"], "source"].unique())
    print(f"  Sources with ≥1 kept vuln row: {sorted(sources_with_kept_vuln)}")

    safe_mask = df["label"] == 0
    df["safe_kept"] = safe_mask & df["source"].isin(sources_with_kept_vuln)

    # 6. Count per source
    rows = []
    all_sources = sorted(df["source"].unique())
    for source in all_sources:
        src_df = df[df["source"] == source]
        vuln_count = int(src_df["vuln_kept"].sum())
        safe_count = int(src_df["safe_kept"].sum())
        total = vuln_count + safe_count
        rows.append({
            "language": language,
            "source": source,
            "vuln": vuln_count,
            "safe": safe_count,
            "total": total,
        })
        print(f"    {source:35s}  vuln={vuln_count:6,}  safe={safe_count:7,}  total={total:7,}")

    return pd.DataFrame(rows)


def main():
    all_frames = []
    for lang in LANGUAGES:
        enriched_path = ENRICHED_DIR / f"{lang}.parquet"
        if not enriched_path.exists():
            print(f"[SKIP] {enriched_path} not found")
            continue
        frame = process_language(lang)
        all_frames.append(frame)

    if not all_frames:
        print("No enriched parquets found. Run `uv run python main.py enrich` first.")
        return

    result = pd.concat(all_frames, ignore_index=True)

    # Filter out sources with 0 total (not in experiment for this language)
    result = result[result["total"] > 0].copy()
    result = result.sort_values(["language", "total"], ascending=[True, False]).reset_index(drop=True)

    print(f"\n{'='*60}")
    print("  FINAL COUNTS (tier full, >= 100 occurrences per family)")
    print(f"{'='*60}")
    print(result.to_string(index=False))

    # Grand totals per language
    print(f"\n{'='*60}")
    print("  TOTALS PER LANGUAGE")
    print(f"{'='*60}")
    for lang in LANGUAGES:
        lang_df = result[result["language"] == lang]
        if lang_df.empty:
            continue
        print(f"  {lang}: vuln={lang_df['vuln'].sum():,}  safe={lang_df['safe'].sum():,}  "
              f"total={lang_df['total'].sum():,}")

    # Grand total across all
    print(f"\n  GRAND TOTAL: vuln={result['vuln'].sum():,}  safe={result['safe'].sum():,}  "
          f"total={result['total'].sum():,}")

    # Save CSV
    result.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
