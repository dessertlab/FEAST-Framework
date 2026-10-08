from collections import Counter
from pathlib import Path

from ingestion.schema import FunctionSample
from ingestion.utils import (
    _NVD_PLACEHOLDERS,
    _CWE_RE,
    EXCLUDED_SAMPLE_IDS,
    code_sample_id,
    load_parquet_dir_or_file,
)

_SUPPORTED_LANGUAGES = {"C", "C++", "Java", "Python"}


def extract_cvefixes(data_path: Path, language: str = "C") -> list[FunctionSample]:
    """Extract FunctionSamples from CVEfixes Parquet file(s).

    Source: Hugging Face `hitoshura25/cvefixes` (3 Parquet shards).
    Pass either a directory containing the `.parquet` files or a single file.

    Columns used: vulnerable_code, cwe_id, hash, language.

    Positives: rows where
      - language matches the `language` parameter
      - vulnerable_code is non-empty
      - cwe_id is non-empty and not an NVD placeholder
      - commit hash appears in exactly one row (single-function commit filter)

    Negatives: none.
    """
    df = load_parquet_dir_or_file(data_path)

    # Map language param to dataset language values
    if language in ("C", "C++"):
        lang_filter = {"C", "C++"}
    else:
        lang_filter = {language}

    df = df[df["language"].isin(lang_filter)].copy()
    df = df[df["vulnerable_code"].notna() & (df["vulnerable_code"].str.strip() != "")].copy()

    df = df[df["cwe_id"].notna()].copy()
    df = df[df["cwe_id"].str.strip() != ""].copy()
    df = df[~df["cwe_id"].str.strip().isin(_NVD_PLACEHOLDERS)].copy()

    df["cwes_parsed"] = df["cwe_id"].str.strip().apply(_CWE_RE.findall)
    df = df[df["cwes_parsed"].map(len) > 0].copy()

    hash_counts = Counter(df["hash"])
    single_hashes = {h for h, n in hash_counts.items() if n == 1}
    df = df[df["hash"].isin(single_hashes)]

    # Drop samples on the global exclusion list (e.g. code that hangs SAT tools).
    if EXCLUDED_SAMPLE_IDS:
        excluded = df["vulnerable_code"].map(code_sample_id).isin(EXCLUDED_SAMPLE_IDS)
        df = df[~excluded].copy()

    # Normalise language label
    if language in ("C", "C++"):
        lang_label = "C/C++"
    else:
        lang_label = language

    return [
        FunctionSample(
            code=str(row["vulnerable_code"]),
            cwes=list(row["cwes_parsed"]),
            label=1,
            branch="real",
            language=lang_label,
        )
        for _, row in df.iterrows()
    ]
