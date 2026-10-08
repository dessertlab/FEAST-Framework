import ast
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ingestion.schema import FunctionSample
from ingestion.utils import _NVD_PLACEHOLDERS, _CWE_RE


@dataclass
class RawPrimeVulEntry:
    code: str
    cwes: list[str]
    label: int
    commit_id: str
    project: str


def _parse_cwes(raw) -> list[str]:
    """Parse CWE field from PrimeVul row."""
    if raw is None or isinstance(raw, float):
        return []
    raw = str(raw).strip()
    if not raw:
        return []

    candidates: list[str] = []
    if raw.startswith("["):
        try:
            items = ast.literal_eval(raw)
            candidates = [c.strip() for c in items if isinstance(c, str) and c.strip()]
        except (ValueError, SyntaxError):
            return []
    else:
        candidates = [c.strip() for c in raw.split() if c.strip()]

    result: list[str] = []
    for c in candidates:
        if c in _NVD_PLACEHOLDERS:
            continue
        parts = _CWE_RE.findall(c)
        result.extend(parts)
    return list(dict.fromkeys(result))


def _load_raw(path: Path) -> list[RawPrimeVulEntry]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd
        rows = pd.read_parquet(path).to_dict(orient="records")
    elif suffix == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    else:
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)

    entries = []
    for row in rows:
        entries.append(RawPrimeVulEntry(
            code=str(row["func"]),
            cwes=_parse_cwes(row.get("cwe")),
            label=int(row["target"]),
            commit_id=str(row.get("commit_id", "")),
            project=str(row.get("project", "")),
        ))
    return entries


def _single_function_commit_ids(entries: list[RawPrimeVulEntry]) -> set[str]:
    counts = Counter(e.commit_id for e in entries if e.label == 1)
    return {cid for cid, n in counts.items() if n == 1}


def extract_primevul(path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from a PrimeVul JSON or Parquet file.

    Positives: target=1, CWE non-empty, single-function commit.
    Negatives: all target=0.
    """
    entries = _load_raw(path)
    single_commits = _single_function_commit_ids(entries)

    samples = []
    for e in entries:
        if e.label == 1:
            if e.cwes and e.commit_id in single_commits:
                samples.append(FunctionSample(
                    code=e.code, cwes=e.cwes, label=1,
                    branch="real", language="C/C++",
                ))
        else:
            samples.append(FunctionSample(
                code=e.code, cwes=[], label=0,
                branch="real", language="C/C++",
            ))
    return samples
