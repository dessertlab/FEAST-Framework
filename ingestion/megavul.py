from collections import Counter
from pathlib import Path

from ingestion.schema import FunctionSample
from ingestion.utils import _NVD_PLACEHOLDERS, _CWE_RE, load_parquet_dir_or_file


def extract_megavul(
    data_path: Path,
    cvss_threshold: float | None = None,
) -> list[FunctionSample]:
    """Extract FunctionSamples from MegaVul Parquet file(s).

    Source: Hugging Face `hitoshura25/megavul` (2 Parquet shards).
    Pass either a directory containing the `.parquet` files or a single file.

    Columns used: vulnerable_code, cwe_id, hash, cvss3_base_score.

    Positives: rows where
      - vulnerable_code is non-empty
      - cwe_id is non-empty and not an NVD placeholder
      - commit hash appears in exactly one row (single-function commit filter)
      - if cvss_threshold is set: cvss3_base_score >= threshold (null rows dropped)

    Negatives: none.
    """
    df = load_parquet_dir_or_file(data_path)

    df = df[df["vulnerable_code"].notna() & (df["vulnerable_code"].str.strip() != "")].copy()

    df = df[df["cwe_id"].notna()].copy()
    df = df[df["cwe_id"].str.strip() != ""].copy()
    df = df[~df["cwe_id"].str.strip().isin(_NVD_PLACEHOLDERS)].copy()

    df["cwes_parsed"] = df["cwe_id"].str.strip().apply(_CWE_RE.findall)
    df = df[df["cwes_parsed"].map(len) > 0].copy()

    if cvss_threshold is not None:
        df = df[df["cvss3_base_score"].notna()].copy()
        df = df[df["cvss3_base_score"] >= cvss_threshold].copy()

    hash_counts = Counter(df["hash"])
    single_hashes = {h for h, n in hash_counts.items() if n == 1}
    df = df[df["hash"].isin(single_hashes)]

    return [
        FunctionSample(
            code=str(row["vulnerable_code"]),
            cwes=list(row["cwes_parsed"]),
            label=1,
            branch="real",
            language="C/C++",
        )
        for _, row in df.iterrows()
    ]
