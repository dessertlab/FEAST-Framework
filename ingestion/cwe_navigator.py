import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class CWENavigator:
    """Loads the MITRE CWE XML catalogue and answers canonicalisation queries.

    Only exposes what the fusion pipeline needs (weaknesses/categories/views lookup,
    the primary-path climb used by ``analysis/canonical.py``). For ad-hoc exploration
    of the raw hierarchy (arbitrary-view paths, descendants, a CLI), see
    ``auxiliary/cwe_explorer.py``.
    """

    def __init__(self, xml_file_path: str):
        """
        Initializes the navigator with the CWE XML file.

        Args:
            xml_file_path: Path to the cwec_latest.xml file downloaded from MITRE
        """
        self.tree = ET.parse(xml_file_path)
        self.root = self.tree.getroot()

        # Namespace used in the MITRE XML file
        self.ns = {'cwe': 'http://cwe.mitre.org/cwe-7'}

        # Dictionaries to store entities
        self.weaknesses: Dict[str, ET.Element] = {}
        self.categories: Dict[str, ET.Element] = {}
        self.views: Dict[str, ET.Element] = {}

        # Parent relations (ChildOf)
        self.child_of: Dict[str, List[str]] = defaultdict(list)

        # MITRE abstraction level of each weakness: "Pillar"|"Class"|"Base"|"Variant"|"Compound".
        self.abstraction_of: Dict[str, str] = {}

        # Primary ChildOf parent scoped per view: view_id -> {cwe -> parent}.
        # A weakness can sit under several parents and several views; within a view MITRE
        # marks exactly one ChildOf edge as Ordinal="Primary" — the canonical navigation
        # parent for that view. We index it so the canonicalizer can climb a deterministic
        # path (no reliance on XML ordering of secondary edges).
        self.primary_parent_in_view: Dict[str, Dict[str, str]] = defaultdict(dict)

        self._parse_xml()

    def _parse_xml(self):
        """Parses the XML file and builds the basic data structures."""

        # Parse Weaknesses
        for weakness in self.root.findall('.//cwe:Weakness', self.ns):
            cwe_id = weakness.get('ID')
            self.weaknesses[cwe_id] = weakness
            self.abstraction_of[cwe_id] = weakness.get('Abstraction') or ''

            # Extract ChildOf relations, keeping View_ID + Ordinal so we can later climb
            # the *primary* path of a chosen view (e.g. CWE-1000 Research Concepts).
            for rel in weakness.findall('.//cwe:Related_Weakness', self.ns):
                if rel.get('Nature') != 'ChildOf':
                    continue
                parent_id = rel.get('CWE_ID')
                self.child_of[cwe_id].append(parent_id)
                view_id = rel.get('View_ID')
                if view_id and rel.get('Ordinal') == 'Primary':
                    # First primary edge for this (view, cwe) wins; duplicates across the
                    # same view in the XML are ignored to stay deterministic.
                    self.primary_parent_in_view[view_id].setdefault(cwe_id, parent_id)

        # Parse Categories
        for category in self.root.findall('.//cwe:Category', self.ns):
            cat_id = category.get('ID')
            self.categories[cat_id] = category

            # Extract ChildOf relations for categories
            for rel in category.findall('.//cwe:Relationships/cwe:Has_Member', self.ns):
                nature = rel.get('Nature')
                if nature == 'ChildOf':
                    parent_id = rel.get('CWE_ID')
                    self.child_of[cat_id].append(parent_id)

        # Parse Views
        for view in self.root.findall('.//cwe:View', self.ns):
            view_id = view.get('ID')
            self.views[view_id] = view

    @staticmethod
    def _num(cwe) -> str:
        """Normalise 'CWE-120' / 'cwe-120' / 120 -> '120'."""
        text = str(cwe).strip().upper()
        return text.removeprefix('CWE-') if text.startswith('CWE-') else text

    def abstraction(self, cwe_id) -> Optional[str]:
        """MITRE abstraction level of a weakness, or None if unknown/not a weakness."""
        return self.abstraction_of.get(self._num(cwe_id)) or None

    def primary_parent(self, cwe_id, view: str = "1000") -> Optional[str]:
        """Primary ChildOf parent of ``cwe_id`` within ``view`` (Ordinal=Primary).

        Returns None when the weakness has no primary edge in that view — i.e. it is a
        root/pillar of the view, or it is not placed under that view at all.
        """
        return self.primary_parent_in_view.get(str(view), {}).get(self._num(cwe_id))

    def primary_path(self, cwe_id, view: str = "1000") -> Tuple[str, ...]:
        """Path from ``cwe_id`` up to its view root, following only primary edges.

        Returned leaf-first: ``(cwe, parent, ..., pillar)``. Climbing stops at the first
        node without a primary parent in the view (a pillar/root). Cycle-safe.
        """
        num = self._num(cwe_id)
        path: List[str] = [num]
        seen = {num}
        current = num
        while True:
            parent = self.primary_parent(current, view)
            if not parent or parent in seen:
                break
            path.append(parent)
            seen.add(parent)
            current = parent
        return tuple(path)

    def get_element_name(self, element_id: str) -> str:
        """Gets the name of an element (weakness or category)."""
        if element_id in self.weaknesses:
            return self.weaknesses[element_id].get('Name', 'Unknown')
        elif element_id in self.categories:
            return self.categories[element_id].get('Name', 'Unknown')
        else:
            return 'Unknown'
