import sys
from pathlib import Path
from collections import Counter
import pandas as pd

# Add the root directory to path to allow importing from ingestion and analysis
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.canonical import CweCanonicalizer
from analysis.dataset import as_list

def main():
    merged_dir = ROOT / 'data' / 'merged'
    xml_path = ROOT / 'data' / 'cwec_latest.xml'
    
    if not xml_path.exists():
        print(f"Error: {xml_path} not found.")
        sys.exit(1)
        
    canon = CweCanonicalizer.from_xml(level="pillar_child", cwe_xml_path=str(xml_path))
    
    languages = ['c_cpp', 'java', 'python']
    
    all_results = []
    
    for lang in languages:
        parquet_path = merged_dir / f"{lang}_merged.parquet"
        if not parquet_path.exists():
            print(f"Warning: {parquet_path} not found. Skipping {lang}.")
            continue
            
        print(f"\nProcessing {lang} from {parquet_path.name}...")
        df = pd.read_parquet(parquet_path)
        
        # 1. Map raw CWEs to canonical families for all rows
        # We only map ground truth cwes
        mapped_cwes = []
        for cwes_val in df['cwes']:
            families = canon.families(as_list(cwes_val))
            mapped_cwes.append(families)
        df['families'] = mapped_cwes
        
        # 2. Count ground-truth occurrences of each canonical family
        family_counts = Counter()
        # Only count for vulnerable samples (label == 1)
        for families in df.loc[df['label'] == 1, 'families']:
            family_counts.update(families)
            
        # 3. Filter families with >= 100 occurrences
        frequent_families = {fam for fam, count in family_counts.items() if count >= 100}
        print(f"Total canonical families: {len(family_counts)}")
        print(f"Frequent canonical families (>= 100 occurrences): {len(frequent_families)}")
        print(f"Frequent families: {sorted(frequent_families, key=lambda x: int(x.removeprefix('CWE-')))}")
        
        # 4. Filter the rows:
        # - Safe samples (label == 0) are kept.
        # - Vulnerable samples (label == 1) are kept only if they have at least one frequent family.
        def keep_row(row):
            if row['label'] == 0:
                return True
            # For label == 1, check if any of its families is frequent
            row_fams = set(row['families'])
            return bool(row_fams & frequent_families)
            
        keep_mask = df.apply(keep_row, axis=1)
        filtered_df = df[keep_mask].copy()
        
        print(f"Original rows: {len(df)}")
        print(f"Filtered rows: {len(filtered_df)}")
        
        # Group by source (dataset) and count
        grouped = filtered_df.groupby(['source', 'label']).size().unstack(fill_value=0)
        if 0 not in grouped.columns:
            grouped[0] = 0
        if 1 not in grouped.columns:
            grouped[1] = 0
            
        grouped = grouped.rename(columns={0: 'safe', 1: 'vuln'})
        grouped['total'] = grouped['safe'] + grouped['vuln']
        grouped = grouped.sort_values(by='total', ascending=False)
        
        for source, row in grouped.iterrows():
            all_results.append({
                'language': lang,
                'dataset': source,
                'vuln': row['vuln'],
                'safe': row['safe'],
                'total': row['total']
            })
            print(f"  {source:<25} | vuln: {row['vuln']:>5} | safe: {row['safe']:>5} | total: {row['total']:>5}")
            
    # Print combined results
    print("\n" + "="*50 + "\nCOMBINED RESULTS:\n" + "="*50)
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    # Save to CSV for easy loading
    output_csv = ROOT / 'auxiliary' / 'filtered_counts_tier_full.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

if __name__ == '__main__':
    main()
