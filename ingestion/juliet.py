import re
import zipfile
from pathlib import Path

from ingestion.schema import FunctionSample

_CWE_DIR_RE = re.compile(r"CWE(\d+)", re.IGNORECASE)

# Extensions per language
_C_EXTS   = {".c", ".cpp", ".cc"}
_JAVA_EXT = ".java"


def _cwe_from_path(path: str) -> str | None:
    """Extract CWE number from first matching directory component."""
    for part in Path(path).parts:
        m = _CWE_DIR_RE.search(part)
        if m:
            return f"CWE-{int(m.group(1))}"
    return None


def _is_bad(name: str) -> bool:
    stem = Path(name).stem.lower()
    return "_bad" in stem or stem.endswith("bad")


def _is_good(name: str) -> bool:
    stem = Path(name).stem.lower()
    return "_good" in stem or "_safe" in stem


def _extract_c(zf: zipfile.ZipFile, language: str) -> list[FunctionSample]:
    exts = _C_EXTS
    samples: list[FunctionSample] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        ext = Path(info.filename).suffix.lower()
        if ext not in exts:
            continue
        cwe = _cwe_from_path(info.filename)
        if not cwe:
            continue
        try:
            code = zf.read(info.filename).decode("utf-8", errors="replace").strip()
        except Exception:
            continue
        if not code:
            continue
        name = info.filename
        if _is_bad(name):
            samples.append(FunctionSample(
                code=code, cwes=[cwe], label=1,
                branch="synth", language=language,
            ))
        elif _is_good(name):
            samples.append(FunctionSample(
                code=code, cwes=[], label=0,
                branch="synth", language=language,
            ))
    return samples


def _extract_java(zf: zipfile.ZipFile) -> list[FunctionSample]:
    samples: list[FunctionSample] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        if Path(info.filename).suffix.lower() != _JAVA_EXT:
            continue
        cwe = _cwe_from_path(info.filename)
        if not cwe:
            continue
        try:
            code = zf.read(info.filename).decode("utf-8", errors="replace").strip()
        except Exception:
            continue
        if not code:
            continue
        name = info.filename
        if _is_bad(name):
            samples.append(FunctionSample(
                code=code, cwes=[cwe], label=1,
                branch="synth", language="Java",
            ))
        elif _is_good(name):
            samples.append(FunctionSample(
                code=code, cwes=[], label=0,
                branch="synth", language="Java",
            ))
    return samples


def extract_juliet(
    data_path: Path,
    language: str = "C/C++",
) -> list[FunctionSample]:
    """Extract FunctionSamples from a Juliet Test Suite ZIP.

    Source: NIST SARD (https://samate.nist.gov/SARD/).

    For C/C++: files named *_bad.* are label=1, *_good*.* are label=0.
    For Java:  same naming convention, whole .java file treated as sample.

    CWE is extracted from the directory name (e.g. CWE121_... -> CWE-121).

    language parameter accepts: "C/C++", "Java".
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Juliet archive not found: {data_path}")

    with zipfile.ZipFile(data_path) as zf:
        if language == "Java":
            return _extract_java(zf)
        else:
            return _extract_c(zf, language)
