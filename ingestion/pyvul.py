import json
from pathlib import Path

from ingestion.schema import FunctionSample
from ingestion.utils import normalise_cwe


def extract_pyvul(data_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from PyVul.

    Source: GitHub `billquan/PyVul`.

    Uses `dataset/function_level_dataset.out` (JSONL) joined with
    `dataset/commits_cwe_map.json` (commit URL -> CWE string).

    Filters to Python only. code_before -> label=1, code_after -> label=0.
    """
    dataset_dir = data_path / "dataset"
    fld_path = dataset_dir / "function_level_dataset.out"
    cwe_map_path = dataset_dir / "commits_cwe_map.json"

    if not fld_path.exists():
        raise FileNotFoundError(f"function_level_dataset.out not found in {dataset_dir}")
    if not cwe_map_path.exists():
        raise FileNotFoundError(f"commits_cwe_map.json not found in {dataset_dir}")

    cwe_map: dict[str, str] = json.loads(cwe_map_path.read_text(encoding="utf-8"))

    samples: list[FunctionSample] = []
    with open(fld_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("programming_language", "").lower() != "python":
                continue
            cwe = normalise_cwe(cwe_map.get(rec.get("commit", ""), ""))
            if not cwe:
                continue
            vuln = str(rec.get("code_before", "") or "").strip()
            fixed = str(rec.get("code_after", "") or "").strip()
            if vuln:
                samples.append(FunctionSample(
                    code=vuln, cwes=[cwe], label=1,
                    branch="real", language="Python",
                ))
            if fixed:
                samples.append(FunctionSample(
                    code=fixed, cwes=[], label=0,
                    branch="real", language="Python",
                ))
    return samples
