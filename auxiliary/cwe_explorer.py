"""Ad-hoc exploration CLI for the MITRE CWE hierarchy.

Not used by the FEAST pipeline (see ingestion/cwe_navigator.py for the
canonicalisation-facing subset); this is a standalone tool for manually
inspecting arbitrary-view paths, descendants and ChildOf chains, e.g.:

    python -m auxiliary.cwe_explorer data/cwec_latest.xml 120 --hierarchy --top-parent
"""
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from ingestion.cwe_navigator import CWENavigator


class CWEExplorer(CWENavigator):
    """CWENavigator plus arbitrary-view path/descendant lookups and print helpers."""

    def __init__(self, xml_file_path: str):
        super().__init__(xml_file_path)

        # View structure: view_id -> {parent_id -> [children_ids]}
        self.view_structure: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        # Parent->child map for reverse navigation
        self.view_parent_map: Dict[str, Dict[str, str]] = defaultdict(dict)

        self._build_view_hierarchies()

    def _build_view_hierarchies(self):
        """
        Builds the complete hierarchy of each View.
        """
        for view_id, view in self.views.items():
            # Collect all direct members of the view
            for member in view.findall('.//cwe:Members/cwe:Has_Member', self.ns):
                member_id = member.get('CWE_ID')
                if member_id:
                    # Add to the structure as a child of the view
                    self.view_structure[view_id][view_id].append(member_id)
                    self.view_parent_map[view_id][member_id] = view_id

                    # If it is a category, process recursively
                    if member_id in self.categories:
                        self._process_category_members(view_id, member_id)

    def _process_category_members(self, view_id: str, category_id: str):
        """
        Recursively processes all members of a category.
        """
        if category_id not in self.categories:
            return

        category = self.categories[category_id]

        # Find all members of this category
        for member in category.findall('.//cwe:Relationships/cwe:Has_Member', self.ns):
            member_id = member.get('CWE_ID')

            if not member_id:
                continue

            # Add to the structure
            self.view_structure[view_id][category_id].append(member_id)
            self.view_parent_map[view_id][member_id] = category_id

            # If it is a category, recurse
            if member_id in self.categories:
                self._process_category_members(view_id, member_id)

    def _find_path_in_view(self, view_id: str, cwe_id: str) -> Optional[Tuple[str, ...]]:
        """
        Finds the complete path from the view root up to the specified CWE.
        """
        if view_id not in self.view_parent_map:
            return None

        if cwe_id not in self.view_parent_map[view_id]:
            visited = set()
            current = cwe_id

            while current and current not in visited:
                visited.add(current)
                if current in self.view_parent_map[view_id]:
                    path = self._build_path(view_id, current)
                    if path:
                        if current != cwe_id:
                            return path + (cwe_id,)
                        return path
                    return None

                parents = self.child_of.get(current, [])
                if not parents:
                    break
                current = parents[0]

            return None

        return self._build_path(view_id, cwe_id)

    def _build_path(self, view_id: str, cwe_id: str) -> Optional[Tuple[str, ...]]:
        """
        Builds the path from the view root by climbing the parent map.
        """
        path = []
        current = cwe_id
        visited = set()

        while current and current not in visited:
            visited.add(current)
            path.append(current)

            if current == view_id:
                break

            parent = self.view_parent_map[view_id].get(current)
            if not parent:
                break

            current = parent

        if path and path[-1] == view_id:
            path.pop()

        path.reverse()

        return tuple(path) if path else None

    def get_top_parent(self, cwe_id: str) -> Optional[str]:
        """Finds the highest-level parent by climbing all ChildOf relations."""
        if cwe_id not in self.child_of and cwe_id not in self.weaknesses and cwe_id not in self.categories:
            return None

        visited: set = set()
        current = cwe_id

        while current and current not in visited:
            visited.add(current)
            parents = self.child_of.get(current, [])

            if not parents:
                return current if current != cwe_id else None

            current = parents[0]

        return None

    def get_paths_in_views(self, cwe_id: str, view_ids: List[str]) -> Dict[str, Optional[Tuple[str, ...]]]:
        """Finds the paths of the CWE in multiple views."""
        results = {}

        for view_id in view_ids:
            if view_id not in self.views:
                results[view_id] = None
                continue

            path = self._find_path_in_view(view_id, cwe_id)
            results[view_id] = path

        return results

    def get_element_type(self, element_id: str) -> str:
        """Gets the type of an element."""
        if element_id in self.weaknesses:
            return 'Weakness'
        elif element_id in self.categories:
            return 'Category'
        else:
            return 'Unknown'

    def print_paths(self, cwe_id: str, paths: Dict[str, Optional[Tuple[str, ...]]]):
        """Prints the found paths in a readable format."""
        cwe_name = self.get_element_name(cwe_id)
        cwe_type = self.get_element_type(cwe_id)

        print(f"\n{'='*70}")
        print(f"CWE-{cwe_id}: {cwe_name} ({cwe_type})")
        print(f"{'='*70}")

        for view_id, path in paths.items():
            view_name = self.views[view_id].get('Name', 'Unknown') if view_id in self.views else 'Unknown'
            print(f"\nView {view_id}: {view_name}")

            if path is None:
                print(f"   CWE not found in this view")
            elif len(path) == 0:
                print(f"   Empty path")
            else:
                print(f"   Path found ({len(path)} elements):")
                for i, element_id in enumerate(path):
                    element_name = self.get_element_name(element_id)
                    element_type = self.get_element_type(element_id)
                    indent = "   " + "  " * i
                    arrow = "\\->" if i == len(path) - 1 else "+->"
                    highlight = " *" if element_id == cwe_id else ""
                    print(f"{indent}{arrow} CWE-{element_id}: {element_name} ({element_type}){highlight}")

    def print_hierarchy(self, cwe_id: str):
        """Prints the complete ChildOf hierarchy of a CWE."""
        print(f"\n{'='*70}")
        print(f"ChildOf Hierarchy for CWE-{cwe_id}")
        print(f"{'='*70}")

        visited: set = set()
        current = cwe_id
        path = []

        while current and current not in visited:
            visited.add(current)
            name = self.get_element_name(current)
            type_str = self.get_element_type(current)
            path.append((current, name, type_str))
            parents = self.child_of.get(current, [])
            if not parents:
                break
            current = parents[0]

        if len(path) == 1:
            print("   No parent (already at highest level)")
        else:
            for i, (id_, name, type_str) in enumerate(path):
                indent = "  " * i
                arrow = "\\->" if i == len(path) - 1 else "+->"
                highlight = " (TOP)" if i == len(path) - 1 else ""
                print(f"{indent}{arrow} CWE-{id_}: {name}{highlight}")

    def get_descendants(self, element_id: str, max_depth: Optional[int] = None) -> Dict[str, Dict[int, List[str]]]:
        """Return all descendants of a View, Category, or Weakness."""
        result = {"type": None, "by_view": {}}

        if element_id in self.views:
            result["type"] = "View"
            result["by_view"][element_id] = self._get_descendants_in_view(element_id, element_id, max_depth)
            return result

        if element_id in self.categories:
            result["type"] = "Category"
            for view_id, parents in self.view_parent_map.items():
                if element_id in parents:
                    result["by_view"][view_id] = self._get_descendants_in_view(view_id, element_id, max_depth)
            if not result["by_view"]:
                result["by_view"]["no_view"] = self._get_descendants_in_categories(element_id, max_depth)
            return result

        if element_id in self.weaknesses:
            result["type"] = "Weakness"
            return result

        result["type"] = "Unknown"
        return result

    def _get_descendants_in_view(self, view_id: str, start_id: str, max_depth: Optional[int]) -> Dict[int, List[str]]:
        results = defaultdict(list)
        queue = deque([(start_id, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth != 0:
                results[depth].append(current)
            if max_depth is not None and depth >= max_depth:
                continue
            for child in self.view_structure[view_id].get(current, []):
                queue.append((child, depth + 1))

        return results

    def _get_descendants_in_categories(self, category_id: str, max_depth: Optional[int]) -> Dict[int, List[str]]:
        results = defaultdict(list)
        queue = deque([(category_id, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth != 0:
                results[depth].append(current)
            if max_depth is not None and depth >= max_depth:
                continue
            category = self.categories.get(current)
            if category is None:
                continue
            for member in category.findall('.//cwe:Relationships/cwe:Has_Member', self.ns):
                child_id = member.get('CWE_ID')
                if child_id:
                    queue.append((child_id, depth + 1))

        return results


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='CWE Explorer')
    parser.add_argument('xml_file', help='Path to MITRE cwec_latest.xml file')
    parser.add_argument('cwe_id', help='CWE ID to analyze (e.g., 120)')
    parser.add_argument('--views', nargs='+', default=['700', '888', '1000', '1154'])
    parser.add_argument('--hierarchy', action='store_true')
    parser.add_argument('--top-parent', action='store_true')
    args = parser.parse_args()

    try:
        explorer = CWEExplorer(args.xml_file)
        cwe_id = args.cwe_id.removeprefix('CWE-')

        if args.hierarchy:
            explorer.print_hierarchy(cwe_id)
        if args.top_parent:
            top = explorer.get_top_parent(cwe_id)
            if top:
                print(f"\nTop Parent: CWE-{top} - {explorer.get_element_name(top)}")
            else:
                print(f"\nNo top parent found for CWE-{cwe_id}")

        paths = explorer.get_paths_in_views(cwe_id, args.views)
        explorer.print_paths(cwe_id, paths)

    except FileNotFoundError:
        print(f"Error: File '{args.xml_file}' not found", file=sys.stderr)
        sys.exit(1)
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
