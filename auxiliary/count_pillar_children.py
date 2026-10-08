import sys
from pathlib import Path
from collections import defaultdict

# Add the root directory to path to allow importing from ingestion and analysis
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.cwe_navigator import CWENavigator

def main():
    xml_path = ROOT / 'data' / 'cwec_latest.xml'
    
    if not xml_path.exists():
        print(f"Error: {xml_path} not found.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading CWE Navigator using: {xml_path}")
    nav = CWENavigator(str(xml_path))
    
    # Identify all pillars in the CWE catalog
    pillars = {}
    for cwe_id, weakness in nav.weaknesses.items():
        if nav.abstraction(cwe_id) == 'Pillar':
            pillars[cwe_id] = weakness.get('Name', 'Unknown Name')
            
    print(f"Found {len(pillars)} pillars in the CWE catalog.\n")
    
    # We want to find direct children of these pillars in View 1000.
    # A weakness 'CWE-X' is a direct child of a pillar 'CWE-Y' in View 1000 if:
    # Y is a pillar, and X is a ChildOf Y under View 1000.
    #
    # We will distinguish between:
    # 1. Primary ChildOf relationships in View 1000 (Ordinal='Primary')
    # 2. All ChildOf relationships in View 1000 (Primary and Secondary)
    
    primary_children = defaultdict(list)
    all_children = defaultdict(list)
    
    for cwe_id, weakness in nav.weaknesses.items():
        # Iterate over all Related_Weakness elements
        for rel in weakness.findall('.//cwe:Related_Weakness', nav.ns):
            nature = rel.get('Nature')
            view_id = rel.get('View_ID')
            parent_id = rel.get('CWE_ID')
            ordinal = rel.get('Ordinal')
            
            if nature == 'ChildOf' and view_id == '1000':
                if parent_id in pillars:
                    # This weakness is a direct child of a pillar in View 1000
                    all_children[parent_id].append(cwe_id)
                    if ordinal == 'Primary':
                        primary_children[parent_id].append(cwe_id)
                        
    # Print the results in a clear format
    print("=" * 110)
    print(f"{'Pillar ID':<12} | {'Pillar Name':<50} | {'Primary Children':<18} | {'Total Children':<14}")
    print("=" * 110)
    
    total_primary = 0
    total_all = 0
    
    # Sort pillars by ID numerically
    sorted_pillars = sorted(pillars.keys(), key=lambda x: int(x))
    
    for p_id in sorted_pillars:
        p_name = pillars[p_id]
        p_prim_count = len(primary_children[p_id])
        p_all_count = len(all_children[p_id])
        
        total_primary += p_prim_count
        total_all += p_all_count
        
        print(f"CWE-{p_id:<8} | {p_name:<50} | {p_prim_count:>16} | {p_all_count:>12}")
        
    print("=" * 110)
    print(f"{'TOTAL':<12} | {'':<50} | {total_primary:>16} | {total_all:>12}")
    print("=" * 110)
    
    # Print the list of children for each pillar for transparency
    print("\nDetailed list of children per pillar:")
    for p_id in sorted_pillars:
        prim_list = sorted(primary_children[p_id], key=lambda x: int(x))
        all_list = sorted(all_children[p_id], key=lambda x: int(x))
        
        prim_str = ", ".join(f"CWE-{c}" for c in prim_list) if prim_list else "None"
        all_str = ", ".join(f"CWE-{c}" for c in all_list) if all_list else "None"
        
        print(f"\nCWE-{p_id} ({pillars[p_id]}):")
        print(f"  - Primary children ({len(prim_list)}): {prim_str}")
        print(f"  - All children ({len(all_list)}): {all_str}")

if __name__ == '__main__':
    main()
