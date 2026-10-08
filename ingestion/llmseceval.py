import re
from pathlib import Path

from ingestion.schema import FunctionSample

_CWE_DIR_RE = re.compile(r"cwe[-_]?(\d+)", re.IGNORECASE)
_LANG_EXTS = {'Python': '.py', 'C/C++': '.c'}


def _cwe_from_path(path: Path) -> str | None:
    for part in path.parts:
        m = _CWE_DIR_RE.search(part)
        if m:
            return f"CWE-{int(m.group(1))}"
    return None


def extract_llmseceval(
    data_path: Path,
    language: str = 'Python',
) -> list[FunctionSample]:
    """Extract FunctionSamples from LLMSecEval.

    Sources:
      - Vulnerable (C/C++ + Python): Zenodo 5225651 copilot-cwe-scenarios-dataset,
        extracted to data_path/zenodo/. gen_scenario/*.{c,py} files only;
        .reject suffixes excluded by glob.
      - Safe (Python only): GitHub tuhh-softsec/LLMSecEval,
        copied to data_path/CWE-NNN/Secure/*.py.

    data_path layout:
      data_path/
        zenodo/
          copilot-cwe-scenarios-dataset/
            experiments_dow/cwe-NNN/scenario-name/gen_scenario/
              *_copilot_N.{py,c}      -- label=1
        CWE-NNN/
          Secure/
            *.py                      -- label=0 (Python only)
    """
    if not data_path.exists():
        raise FileNotFoundError(f"LLMSecEval directory not found: {data_path}")

    ext = _LANG_EXTS.get(language)
    if ext is None:
        raise ValueError(f"Unsupported language: {language!r}")

    lang_key = 'Python' if language == 'Python' else 'C/C++'
    samples: list[FunctionSample] = []

    # --- Vulnerable samples from Zenodo ---
    zenodo_root = data_path / 'zenodo'
    if zenodo_root.exists():
        for gen_file in zenodo_root.rglob(f'gen_scenario/*{ext}'):
            code = gen_file.read_text(encoding='utf-8', errors='replace').strip()
            if not code:
                continue
            cwe = _cwe_from_path(gen_file)
            if not cwe:
                continue
            samples.append(FunctionSample(
                code=code, cwes=[cwe], label=1,
                branch='ai', language=lang_key,
            ))

    # --- Safe samples from GitHub (Python only) ---
    if language == 'Python':
        for cwe_dir in data_path.iterdir():
            if not cwe_dir.is_dir() or cwe_dir.name == 'zenodo':
                continue
            secure_dir = cwe_dir / 'Secure'
            if not secure_dir.exists():
                continue
            for fpath in secure_dir.iterdir():
                if fpath.suffix.lower() != '.py':
                    continue
                code = fpath.read_text(encoding='utf-8', errors='replace').strip()
                if not code:
                    continue
                samples.append(FunctionSample(
                    code=code, cwes=[], label=0,
                    branch='ai', language='Python',
                ))

    return samples
