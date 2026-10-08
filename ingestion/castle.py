import json
from pathlib import Path

from ingestion.schema import FunctionSample
from ingestion.utils import _CWE_RE, _NVD_PLACEHOLDERS


def extract_castle(data_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from CASTLE JSON file.

    Source: GitHub `CASTLE-Benchmark/CASTLE-Benchmark`.
    Expected file: CASTLE-C250.json (or any .json in the directory).

    Schema: top-level dict with 'tests' list; each entry has:
      - code        : str  -- function body
      - cwe         : int  -- CWE number (e.g. 22, not "CWE-22")
      - vulnerable  : bool -- True = vulnerable

    Positives: vulnerable=True, CWE non-empty.
    Negatives: vulnerable=False, cwes=[].
    """
    if data_path.is_dir():
        candidates = list(data_path.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No .json files found in {data_path}")
        data_path = candidates[0]

    with open(data_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    # Actual schema: top-level dict with 'tests' list; fallback to plain list
    records = raw.get("tests", []) if isinstance(raw, dict) else raw

    samples: list[FunctionSample] = []
    for rec in records:
        code = str(rec.get("code", "") or "").strip()
        if not code:
            continue
        is_vuln = bool(rec.get("vulnerable", False))
        cwe_val = rec.get("cwe", "")
        # cwe is an int in this dataset (e.g. 22); normalise to "CWE-22"
        if isinstance(cwe_val, int):
            cwe_raw = f"CWE-{cwe_val}"
        else:
            cwe_raw = str(cwe_val or "").strip()

        if is_vuln:
            if not cwe_raw or cwe_raw in _NVD_PLACEHOLDERS:
                continue
            cwes = _CWE_RE.findall(cwe_raw)
            if not cwes:
                continue
            samples.append(FunctionSample(
                code=code, cwes=cwes, label=1,
                branch="synth", language="C/C++",
            ))
        else:
            samples.append(FunctionSample(
                code=code, cwes=[], label=0,
                branch="synth", language="C/C++",
            ))

    return samples
