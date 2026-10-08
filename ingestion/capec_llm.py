import re
from pathlib import Path

from ingestion.schema import FunctionSample
from ingestion.utils import _CWE_RE, load_json_records

# Language detection heuristics from code content / file metadata
_JAVA_KEYWORDS   = re.compile(r"\b(public\s+class|import\s+java\.|@Override|System\.out)\b")
_PYTHON_KEYWORDS = re.compile(r"\b(def\s+\w+\s*\(|import\s+\w+|print\s*\(|:\s*$)", re.MULTILINE)


def _detect_language(code: str) -> str | None:
    if _JAVA_KEYWORDS.search(code):
        return "Java"
    if _PYTHON_KEYWORDS.search(code):
        return "Python"
    return None


def _extract_cwes_from_description(description: str) -> list[str]:
    """Extract all CWE-NNN tokens from a free-text description."""
    found = _CWE_RE.findall(description)
    return list(dict.fromkeys(found))  # deduplicate, preserve order


def extract_capec_llm(
    data_path: Path,
    language: str = "Java",
) -> list[FunctionSample]:
    """Extract FunctionSamples from CAPEC_LLM dataset.

    Source: GitHub `llmForCapec/CAPECDatasetsLLM`.

    Expected structure: JSON files (one per CAPEC entry or a single dataset.json)
    with records containing:
      - capec_id      : str
      - code_snippet  : str  -- the code sample
      - description   : str  -- free text mentioning CWE IDs

    Samples without a parsable CWE are dropped.
    Language is detected from code content; language parameter filters the output.

    language parameter accepts: "Java", "Python".
    """
    if not data_path.exists():
        raise FileNotFoundError(f"CAPEC_LLM path not found: {data_path}")

    records = load_json_records(data_path)

    samples: list[FunctionSample] = []

    for rec in records:
        code = str(rec.get("code_snippet", "") or "").strip()
        if not code:
            continue

        # Language filter
        detected = _detect_language(code)
        if detected != language:
            continue

        description = str(rec.get("description", "") or "")
        cwes = _extract_cwes_from_description(description)
        if not cwes:
            continue

        samples.append(FunctionSample(
            code=code, cwes=cwes, label=1,
            branch="ai", language=language,
        ))

    return samples
