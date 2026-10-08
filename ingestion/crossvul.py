import re
import zipfile
from pathlib import Path

from ingestion.schema import FunctionSample

_LANG_FOLDERS: dict[str, str] = {
    'c':    'C/C++',
    'cc':   'C/C++',
    'cpp':  'C/C++',
    'cxx':  'C/C++',
    'java': 'Java',
    'py':   'Python',
}

_CWE_DIR_RE = re.compile(r'^CWE-(\d+)$', re.IGNORECASE)


def extract_crossvul(
    data_path: Path,
    language: str = 'C/C++',
) -> list[FunctionSample]:
    """Extract FunctionSamples from CrossVul ZIP.

    Source: Zenodo crossvul.zip.

    Structure inside ZIP:
      dataset_final_sorted/
        CWE-NNN/
          <lang>/          -- c, cc, cpp, cxx -> C/C++; java -> Java; py -> Python
            bad_NNNN_N     -- label=1 (vulnerable)
            good_NNNN_N    -- label=0 (safe)

    Files have no extension; language is inferred from the folder name.
    language parameter accepts: "C/C++", "Java", "Python".
    """
    if not data_path.exists():
        raise FileNotFoundError(f"CrossVul ZIP not found: {data_path}")

    samples: list[FunctionSample] = []

    with zipfile.ZipFile(data_path) as zf:
        for entry in zf.infolist():
            if entry.is_dir():
                continue
            parts = entry.filename.split('/')
            # Expect: dataset_final_sorted / CWE-NNN / lang / filename
            if len(parts) < 4:
                continue

            cwe_part  = parts[1]
            lang_part = parts[2].lower()
            fname     = parts[3]

            mapped_lang = _LANG_FOLDERS.get(lang_part)
            if mapped_lang != language:
                continue

            m_cwe = _CWE_DIR_RE.match(cwe_part)
            if not m_cwe:
                continue
            cwe = f"CWE-{int(m_cwe.group(1))}"

            if fname.startswith('bad_'):
                label = 1
            elif fname.startswith('good_'):
                label = 0
            else:
                continue

            code = zf.read(entry).decode('utf-8', errors='replace').strip()
            if not code:
                continue

            samples.append(FunctionSample(
                code=code,
                cwes=[cwe] if label == 1 else [],
                label=label,
                branch='real',
                language=language,
            ))

    return samples
