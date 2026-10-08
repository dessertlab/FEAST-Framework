from pathlib import Path

from ingestion.schema import FunctionSample
from ingestion.utils import _CWE_RE, load_json_records, normalise_cwe


def _extract_cwes_from_cwe_info(cwe_info) -> list[str]:
    """Parse cwe_info field (dict or string) into a list of CWE IDs."""
    if not cwe_info:
        return []
    if isinstance(cwe_info, dict):
        cwes = []
        for key in cwe_info:
            cwe = normalise_cwe(key)
            if cwe:
                cwes.append(cwe)
        return list(dict.fromkeys(cwes))
    # fallback: treat as string
    return list(dict.fromkeys(_CWE_RE.findall(str(cwe_info))))


def extract_patcheval(
    data_path: Path,
    docker_verified_only: bool = False,
) -> list[FunctionSample]:
    """Extract FunctionSamples from PatchEval.

    Source: GitHub `bytedance/PatchEval`.

    Expected: a JSON file (or directory of JSON files) with records:
      - cve_id              : str
      - cwe_info            : dict  -- {CWE-NNN: description, ...}
      - vul_func            : str   -- vulnerable function body
      - fix_func            : str   -- fixed function body
      - programming_language: str

    Only Python-language entries are extracted.

    docker_verified_only: if True, keep only entries where docker_verified=True.
    """
    records = load_json_records(data_path)

    samples: list[FunctionSample] = []

    for rec in records:
        lang = str(rec.get("programming_language", "") or "").strip().lower()
        if lang != "python":
            continue

        if docker_verified_only and not rec.get("docker_verified", False):
            continue

        cwes = _extract_cwes_from_cwe_info(rec.get("cwe_info"))
        if not cwes:
            continue

        vul_func = str(rec.get("vul_func", "") or "").strip()
        if vul_func:
            samples.append(FunctionSample(
                code=vul_func, cwes=cwes, label=1,
                branch="real", language="Python",
            ))

        fix_func = str(rec.get("fix_func", "") or "").strip()
        if fix_func:
            samples.append(FunctionSample(
                code=fix_func, cwes=[], label=0,
                branch="real", language="Python",
            ))

    return samples
