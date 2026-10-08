from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ingestion.schema import FunctionSample
from ingestion.utils import _NVD_PLACEHOLDERS


@dataclass
class RawICVulEntry:
    code: str
    cwes: list[str]
    label: int
    commit_hash: str


def extract_icvul(base_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from an ICVul dataset directory.

    Join logic:
      1. Filter function_info to rows where before_change=True.
      2. Join function_info.hash -> cve_fc_vcc_mapping.fc_hash.
      3. Assign the CVE's CWE to each matched function.
      4. Drop entries whose CWE resolves to an NVD placeholder.

    Positives: before_change=True functions with a real CWE.
    Negatives: none.
    """
    functions = pd.read_csv(base_path / "function_info.csv")
    mapping   = pd.read_csv(base_path / "cve_fc_vcc_mapping.csv")

    functions = functions[functions["before_change"] == True].copy()

    mapping = mapping[mapping["cwe_id"].notna()].copy()
    mapping = mapping[~mapping["cwe_id"].str.strip().isin(_NVD_PLACEHOLDERS)].copy()
    mapping = mapping[mapping["cwe_id"].str.strip() != ""].copy()

    cwe_by_hash = (
        mapping.groupby("fc_hash")["cwe_id"]
        .apply(lambda s: sorted(set(s.dropna().astype(str).str.strip())))
        .reset_index()
        .rename(columns={"cwe_id": "cwes", "fc_hash": "hash"})
    )

    merged = functions.merge(cwe_by_hash, on="hash", how="inner")
    merged = merged[merged["cwes"].map(len) > 0]

    return [
        FunctionSample(
            code=str(row["code"]), cwes=list(row["cwes"]), label=1,
            branch="real", language="C/C++",
        )
        for _, row in merged.iterrows()
    ]
