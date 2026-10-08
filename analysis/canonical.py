"""Canonicalisation of CWE IDs to direct children of CWE-1000 pillars.

The fusion methodology evaluates tools and ground truth at the granularity of a
*CWE family* rather than the raw leaf CWE. The family set is now fixed to the
weaknesses that are direct children of a Pillar in the Research Concepts view
(CWE-1000): along the deterministic primary ``ChildOf`` path, a raw CWE maps to
``direct_child -> pillar``'s ``direct_child``.

Example paths in CWE-1000::

    CWE-89  -> CWE-943 -> CWE-74  -> CWE-707  maps to CWE-74
    CWE-120 -> CWE-787 -> CWE-119 -> CWE-118 -> CWE-664 maps to CWE-118
    CWE-1024 -> CWE-697 maps to CWE-1024

Pillars themselves are not families in this analysis level, because the target
families are only their first-order children. Downstream matching remains exact
set membership; the CWE hierarchy is consulted only here.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from analysis.dataset import as_list, normalise_cwe, normalise_cwes
from ingestion.cwe_navigator import CWENavigator

CANONICAL_LEVEL = "pillar_child"
LEVELS = (CANONICAL_LEVEL,)
RESEARCH_VIEW = "1000"


class CweCanonicalizer:
    """Map raw CWE IDs to the direct child under their CWE-1000 pillar."""

    def __init__(self, navigator: CWENavigator, level: str = CANONICAL_LEVEL, view: str = RESEARCH_VIEW):
        if level not in LEVELS:
            raise ValueError(f"level must be {CANONICAL_LEVEL!r}, got {level!r}")
        self.nav = navigator
        self.level = level
        self.view = view
        self._cache: dict[str, str | None] = {}

    @classmethod
    def from_xml(cls, level: str = CANONICAL_LEVEL, cwe_xml_path: str = "data/cwec_latest.xml") -> "CweCanonicalizer":
        return cls(CWENavigator(cwe_xml_path), level=level)

    # -- core mapping ---------------------------------------------------------
    def family(self, cwe) -> str | None:
        """Return the direct-child-under-pillar family for ``cwe``.

        Returns ``None`` for tokens that are not weaknesses in the catalogue,
        weaknesses outside the CWE-1000 primary pillar hierarchy, and pillars
        themselves. These values are reported as unmapped and excluded from the
        analysis.
        """
        normalised = normalise_cwe(cwe)
        if normalised is None:
            return None
        num = normalised.removeprefix("CWE-")
        if num in self._cache:
            return self._cache[num]

        family = self._resolve(num)
        self._cache[num] = family
        return family

    def _resolve(self, num: str) -> str | None:
        if num not in self.nav.weaknesses:
            return None

        # primary_path is leaf-first: (cwe, parent, ..., direct_child, pillar).
        path = self.nav.primary_path(num, self.view)
        if len(path) < 2:
            return None

        pillar = path[-1]
        if self.nav.abstraction(pillar) != "Pillar":
            return None

        target = path[-2]
        return f"CWE-{target}"

    def families(self, cwes: Iterable) -> list[str]:
        """Map raw CWEs to the sorted set of their defined canonical families."""
        out = {self.family(cwe) for cwe in normalise_cwes(cwes)}
        out.discard(None)
        return sorted(out, key=lambda c: int(c.removeprefix("CWE-")))

    # -- dataframe-level transform ------------------------------------------
    def canonicalize_frame(self, df: pd.DataFrame, tool_columns: Iterable[str]) -> pd.DataFrame:
        """Return a copy of ``df`` with GT ``cwes`` and tool columns rewritten.

        Each list of raw CWE IDs becomes a sorted list of direct children of CWE-1000
        pillars. Computed once up front and reused across folds, so cached hierarchy
        lookups happen a single time per distinct CWE.
        """
        out = df.copy()
        columns = ["cwes", *[c for c in tool_columns if c in out.columns]]
        for column in columns:
            if column in out.columns:
                out[column] = out[column].map(lambda values: self.families(as_list(values)))
        return out

    def canonical_map(self, cwes: Iterable) -> pd.DataFrame:
        """Audit table mapping each distinct input CWE to its canonical family."""
        rows = []
        for cwe in sorted(normalise_cwes(cwes), key=lambda c: int(c.removeprefix("CWE-"))):
            rows.append({"cwe": cwe, "family": self.family(cwe), "level": self.level})
        return pd.DataFrame(rows, columns=["cwe", "family", "level"])


def all_raw_cwes(df: pd.DataFrame, tool_columns: Iterable[str]) -> set[str]:
    """Collect every raw CWE appearing in ground truth or any tool column."""
    cwes: set[str] = set()
    for column in ["cwes", *[c for c in tool_columns if c in df.columns]]:
        if column in df.columns:
            for values in df[column]:
                cwes |= normalise_cwes(as_list(values))
    return cwes
