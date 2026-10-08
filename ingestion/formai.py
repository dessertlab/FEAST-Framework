import json
import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from ingestion.schema import FunctionSample

# Mapping from ESBMC/FormAI vul_type strings to CWE IDs.
# Based on ESBMC property checks used in the FormAI dataset paper.
_VUL_TYPE_TO_CWE: dict[str, str] = {
    "buffer overflow":              "CWE-121",
    "buffer overflow on scanf":     "CWE-121",
    "stack-based buffer overflow":  "CWE-121",
    "heap buffer overflow":         "CWE-122",
    "array bounds violated":        "CWE-119",
    "array bounds violated lower bound": "CWE-119",
    "array bounds violated upper bound": "CWE-119",
    "out-of-bounds":                "CWE-787",
    "out-of-bounds read":           "CWE-125",
    "out-of-bounds write":          "CWE-787",
    "dereference failure":          "CWE-476",
    "memory leak":                  "CWE-401",
    "memory leak detected":         "CWE-401",
    "use after free":               "CWE-416",
    "use-after-free":               "CWE-416",
    "double free":                  "CWE-415",
    "dangling pointer":             "CWE-416",
    "null pointer dereference":     "CWE-476",
    "null pointer":                 "CWE-476",
    "integer overflow":             "CWE-190",
    "integer underflow":            "CWE-191",
    "division by zero":             "CWE-369",
    "array bounds":                 "CWE-119",
    "array out of bounds":          "CWE-119",
    "uninitialized variable":       "CWE-457",
    "format string":                "CWE-134",
    "race condition":               "CWE-362",
    "deadlock":                     "CWE-833",
    "arithmetic overflow":          "CWE-190",
    "signed integer overflow":      "CWE-190",
    "unsigned integer overflow":    "CWE-190",
}


def _normalise_vul_type(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", raw.strip().lower()).strip()


def _map_vul_type(raw: str) -> str | None:
    if not raw:
        return None
    key = _normalise_vul_type(raw)
    for known, cwe in _VUL_TYPE_TO_CWE.items():
        known_key = _normalise_vul_type(known)
        if key == known_key or known_key in key:
            return cwe
    return None


def _pick(row: dict, *names: str) -> object:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if not missing and (not isinstance(value, str) or value.strip()):
            return value
    return ""


def _iter_json_array(path: Path) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buffer = ""
        in_array = False
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk and not buffer.strip():
                break
            buffer += chunk
            while True:
                stripped = buffer.lstrip()
                if not in_array:
                    if not stripped:
                        buffer = stripped
                        break
                    if stripped[0] == "[":
                        in_array = True
                        stripped = stripped[1:].lstrip()
                    else:
                        raise ValueError(f"{path} must contain a JSON array")
                if stripped.startswith("]"):
                    return
                if stripped.startswith(","):
                    stripped = stripped[1:].lstrip()
                try:
                    item, idx = decoder.raw_decode(stripped)
                except json.JSONDecodeError:
                    buffer = stripped
                    break
                if isinstance(item, dict):
                    yield item
                buffer = stripped[idx:]


def _iter_records(data_path: Path) -> Iterator[dict]:
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(data_path)
        yield from df.to_dict("records")
        return
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            with data_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            yield record
            return
        yield from _iter_json_array(data_path)
        return
    raise ValueError(f"Unsupported FormAI file type: {data_path.suffix}")


def _resolve_data_file(data_path: Path) -> Path:
    if not data_path.is_dir():
        return data_path
    candidates = [
        data_path / "FormAI-v2.json",
        data_path / "FormAI_dataset_human_readable-V1.csv",
    ]
    candidates.extend(sorted(data_path.glob("*.json")))
    candidates.extend(sorted(data_path.glob("*.jsonl")))
    candidates.extend(sorted(data_path.glob("*.csv")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No FormAI .json, .jsonl, or .csv files found in {data_path}")


def extract_formai(data_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from FormAI CSV/JSON files.

    Source: GitHub / HuggingFace `NTUYG/FormAI-dataset`.
    Expected v1 CSV columns: filename, classification, source_code, vul_type.
    Expected v2 JSON fields: file_name, category, source_code, error_type.

    Positives: classification/category contains 'Vulnerable', ESBMC finding maps to a known CWE.
    Negatives: classification contains 'Non-Vulnerable' or 'Safe'.

    Only C/C++ files (extension filter on filename column).
    """
    data_path = _resolve_data_file(data_path)

    samples: list[FunctionSample] = []
    for i, row in enumerate(_iter_records(data_path)):
        filename = str(_pick(row, "filename", "file_name") or "")
        ext = Path(filename).suffix.lower()
        if ext not in (".c", ".cpp", ".cc"):
            continue

        code = str(_pick(row, "source_code", "code", "code_snippet") or "").strip()
        if not code:
            continue

        classification = str(_pick(row, "classification", "category") or "").strip().lower()
        vul_type = str(_pick(row, "vul_type", "error_type", "violated_property", "stack_trace") or "").strip()
        sample_id = Path(filename).stem or f"formai_{i}"

        if "vulnerable" in classification and "non" not in classification:
            cwe = _map_vul_type(vul_type)
            if not cwe:
                continue
            samples.append(FunctionSample(
                code=code, cwes=[cwe], label=1,
                branch="ai", language="C/C++", sample_id=sample_id,
            ))
        elif "non-vulnerable" in classification or "safe" in classification:
            samples.append(FunctionSample(
                code=code, cwes=[], label=0,
                branch="ai", language="C/C++", sample_id=sample_id,
            ))

    return samples
