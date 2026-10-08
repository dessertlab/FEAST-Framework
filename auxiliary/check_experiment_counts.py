import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.canonical import CweCanonicalizer
from analysis.dataset import as_list

def check_lang(lang):
    print(f"\n==================== {lang.upper()} ====================")
    enriched_path = ROOT / 'data' / 'enriched' / f"{lang}.parquet"
    folds_path = ROOT / 'data' / 'results' / lang / 'pillar_child' / 'full' / 'folds.csv'
    config_path = ROOT / 'data' / 'results' / lang / 'pillar_child' / 'full' / 'config.json'
    xml_path = ROOT / 'data' / 'cwec_latest.xml'
    
    if not enriched_path.exists() or not folds_path.exists() or not config_path.exists():
        print(f"Files for {lang} are missing.")
        return
        
    df = pd.read_parquet(enriched_path)
    folds = pd.read_csv(folds_path)
    
    print(f"Enriched parquet rows: {len(df)}")
    print(f"Folds CSV rows: {len(folds)}")
    
    # Let's count actual vuln/safe per source in the enriched parquet (all rows)
    all_grouped = df.groupby(['source', 'label']).size().unstack(fill_value=0)
    all_grouped = all_grouped.rename(columns={0: 'safe', 1: 'vuln'})
    all_grouped['total'] = all_grouped['safe'] + all_grouped['vuln']
    print("\n--- Raw counts in enriched dataset (all rows in folds) ---")
    for source, row in all_grouped.sort_values(by='total', ascending=False).iterrows():
        print(f"  {source:<25} | vuln: {row['vuln']:>6,} | safe: {row['safe']:>6,} | total: {row['total']:>6,}")
        
    # Let's see what happens if we filter out rows with no frequent CWE families
    canon = CweCanonicalizer.from_xml(level="pillar_child", cwe_xml_path=str(xml_path))
    mapped_cwes = [canon.families(as_list(cwes)) for cwes in df['cwes']]
    df['families'] = mapped_cwes
    
    from collections import Counter
    family_counts = Counter()
    for families in df.loc[df['label'] == 1, 'families']:
        family_counts.update(families)
        
    frequent_families = {fam for fam, count in family_counts.items() if count >= 100}
    
    def has_frequent(row):
        if row['label'] == 0:
            return True
        return bool(set(row['families']) & frequent_families)
        
    filtered_df = df[df.apply(has_frequent, axis=1)].copy()
    filtered_grouped = filtered_df.groupby(['source', 'label']).size().unstack(fill_value=0)
    filtered_grouped = filtered_grouped.rename(columns={0: 'safe', 1: 'vuln'})
    filtered_grouped['total'] = filtered_grouped['safe'] + filtered_grouped['vuln']
    
    print("\n--- Filtered counts (vulnerable rows must have at least one CWE family >= 100 occurrences) ---")
    for source, row in filtered_grouped.sort_values(by='total', ascending=False).iterrows():
        print(f"  {source:<25} | vuln: {row['vuln']:>6,} | safe: {row['safe']:>6,} | total: {row['total']:>6,}")

def main():
    for lang in ['c_cpp', 'java', 'python']:
        check_lang(lang)

if __name__ == '__main__':
    main()
