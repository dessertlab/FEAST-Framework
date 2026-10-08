import hashlib
import json
import re
from pathlib import Path

import pandas as pd

_NVD_PLACEHOLDERS: frozenset[str] = frozenset({
    "NVD-CWE-Other",
    "NVD-CWE-noinfo",
    "NVD-CWE-Other ",
})

_CWE_RE: re.Pattern = re.compile(r"CWE-\d+")
_CWE_RE_LOOSE: re.Pattern = re.compile(r"CWE[-_]?(\d+)", re.IGNORECASE)


def _norm_code(code: str) -> str:
    """Normalise code for content hashing (mirrors ``main._norm``)."""
    code = str(code).replace("\r\n", "\n").replace("\r", "\n")
    lines = [l.rstrip() for l in code.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def code_sample_id(code: str) -> str:
    """Stable content-derived sample id (matches ``main._hash(code)[:16]``)."""
    return hashlib.sha256(_norm_code(code).encode()).hexdigest()[:16]


# Samples deliberately dropped at ingestion time. Keyed by content-derived
# sample_id so the exclusion survives any path/source bookkeeping.
#   6792fa11e2e1c154 -- CVEfixes(C) RELIC crypto benchmark function (CWE-190)
#       that hangs the SAT static-analysis tools (joern/codeql run unbounded and
#       never terminate), so it can never be materialized/analysed.
EXCLUDED_SAMPLE_IDS: frozenset[str] = frozenset({
    "6792fa11e2e1c154",
})


def split_cwe(raw: str) -> list[str]:
    """Extract all CWE-NNN tokens from raw string.

    Handles concatenated IDs (CWE-20CWE-190 -> [CWE-20, CWE-190]),
    NVD placeholders (dropped), and empty/None input.
    """
    if not raw or not isinstance(raw, str):
        return []
    raw = raw.strip()
    if raw in _NVD_PLACEHOLDERS:
        return []
    return _CWE_RE.findall(raw)


def normalise_cwe(raw) -> str:
    """Normalise a single CWE reference to canonical 'CWE-N' form.

    Accepts 'CWE-089', 'cwe-79', 'CWE79' (no dash), 'CWE-79-1' (extra suffix,
    first number wins), and bare digits ('79'). Returns '' if no CWE number
    can be extracted.
    """
    if raw is None:
        return ""
    raw = str(raw).strip()
    if not raw:
        return ""
    if raw.isdigit():
        return f"CWE-{int(raw)}"
    m = _CWE_RE_LOOSE.search(raw)
    if not m:
        return ""
    return f"CWE-{int(m.group(1))}"


def load_parquet_dir_or_file(path: Path) -> pd.DataFrame:
    """Load a single .parquet file, or concatenate every .parquet file under a directory."""
    if path.is_dir():
        parts = sorted(path.rglob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No .parquet files found in {path}")
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return pd.read_parquet(path)


def load_json_records(path: Path) -> list[dict]:
    """Load JSON records from a single file, or every .json file under a directory.

    Each file may hold a list of records or a single record dict.
    """
    if path.is_file():
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return raw if isinstance(raw, list) else [raw]

    records: list[dict] = []
    for jf in sorted(path.rglob("*.json")):
        with open(jf, encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, list):
            records.extend(raw)
        elif isinstance(raw, dict):
            records.append(raw)
    return records
