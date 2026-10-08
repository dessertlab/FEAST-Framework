
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

BASE_COLUMNS = {
    "code",
    "language",
    "label",
    "cwes",
    "branch",
    "source",
    "code_hash",
    "sample_id",
}
KNOWN_LANGUAGE_SLUGS = ["c_cpp", "java", "python"]
LANGUAGE_ALIASES = {
    "c": "c_cpp",
    "cpp": "c_cpp",
    "c++": "c_cpp",
    "c/c++": "c_cpp",
    "py": "python",
}


def as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [value]


def values_as_list(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return list(values)


def normalise_cwe(cwe: object) -> str | None:
    text = str(cwe).strip().upper()
    if not text:
        return None
    if text.startswith("CWE-"):
        suffix = text.removeprefix("CWE-")
    elif text.startswith("CWE"):
        suffix = text.removeprefix("CWE")
    else:
        suffix = text
    if not suffix.isdigit():
        return None
    return f"CWE-{int(suffix)}"


def normalise_cwes(cwes) -> set[str]:
    out = {normalise_cwe(cwe) for cwe in values_as_list(cwes)}
    return {cwe for cwe in out if cwe}


def detect_tool_columns(df: pd.DataFrame) -> list[str]:
    candidates = [col for col in df.columns if col not in BASE_COLUMNS]
    tool_cols = []
    for col in candidates:
        non_null = df[col].dropna()
        sample = non_null.head(25).tolist()
        if not sample or all(isinstance(as_list(value), list) for value in sample):
            tool_cols.append(col)
    return tool_cols


def _language_slug(language: str) -> str:
    language = language.lower().strip()
    return LANGUAGE_ALIASES.get(language, language)


def load_enriched(language: str = "all", enriched_dir: str | Path = "data/enriched") -> "FeastDataset":
    enriched_dir = Path(enriched_dir)
    if not enriched_dir.exists():
        raise FileNotFoundError(f"{enriched_dir} not found. Run `uv run python main.py enrich` first.")

    language = language.lower().strip()
    if language == "all":
        paths = [enriched_dir / f"{slug}.parquet" for slug in KNOWN_LANGUAGE_SLUGS]
        paths = [path for path in paths if path.exists()]
    else:
        paths = [enriched_dir / f"{_language_slug(language)}.parquet"]

    if not paths:
        raise FileNotFoundError(f"No enriched parquet files found under {enriched_dir}")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing enriched parquet(s): {missing}")

    df = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    return FeastDataset(df, source_paths=tuple(paths))


class FeastDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        tool_columns: Iterable[str] | None = None,
        history: tuple[str, ...] = (),
        source_paths: tuple[Path, ...] = (),
    ):
        self.df = df.reset_index(drop=True)
        self.tool_columns = list(tool_columns) if tool_columns is not None else detect_tool_columns(self.df)
        self.history = history
        self.source_paths = source_paths

    @classmethod
    def from_parquet(cls, path: str | Path) -> "FeastDataset":
        path = Path(path)
        return cls(pd.read_parquet(path), source_paths=(path,))

    def _clone(self, df: pd.DataFrame, step: str) -> "FeastDataset":
        return FeastDataset(
            df,
            tool_columns=self.tool_columns,
            history=(*self.history, step),
            source_paths=self.source_paths,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"FeastDataset(rows={len(self):,}, tools={self.tool_columns}, filters={len(self.history)})"

    def to_pandas(self) -> pd.DataFrame:
        return self.df.copy()

    def preview(self, n: int = 10, columns: list[str] | None = None) -> pd.DataFrame:
        if columns is None:
            columns = ["sample_id", "language", "branch", "source", "label", "cwes", *self.tool_columns]
        return self.df[[col for col in columns if col in self.df.columns]].head(n)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(path, index=False)
        return path

    def _columns_for_where(self, where: str) -> list[str]:
        where = where.lower()
        if where in {"label", "ground_truth", "truth", "cwes"}:
            return ["cwes"]
        if where == "tools":
            return self.tool_columns
        if where == "any":
            return ["cwes", *self.tool_columns]
        if where in self.tool_columns:
            return [where]
        raise ValueError(f"Unknown where={where!r}. Use 'label', 'tools', 'any', or one of: {self.tool_columns}")

    def _row_cwes(self, row, where: str = "label") -> list[str]:
        values = []
        for col in self._columns_for_where(where):
            if col in row:
                values.extend(as_list(row[col]))
        return [cwe for cwe in (normalise_cwe(value) for value in values) if cwe]

    def filter_by_branch(self, branches: str | Iterable[str]) -> "FeastDataset":
        branches = {str(branch).lower() for branch in values_as_list(branches)}
        mask = self.df["branch"].astype(str).str.lower().isin(branches)
        return self._clone(self.df[mask], f"branch in {sorted(branches)}")

    def filter_by_dataset(self, datasets: str | Iterable[str], contains: bool = False) -> "FeastDataset":
        datasets = values_as_list(datasets)
        source = self.df["source"].astype(str)
        if contains:
            escaped = pd.Series(datasets).astype(str).str.replace(r"([\\.^$*+?{}\[\]|()])", r"\\\1", regex=True)
            pattern = "|".join(escaped)
            mask = source.str.contains(pattern, case=False, na=False, regex=True)
            step = f"source contains {datasets}"
        else:
            wanted = {str(dataset).lower() for dataset in datasets}
            mask = source.str.lower().isin(wanted)
            step = f"source in {sorted(wanted)}"
        return self._clone(self.df[mask], step)

    filter_by_source = filter_by_dataset

    def filter_by_language(self, languages: str | Iterable[str]) -> "FeastDataset":
        wanted = {_language_slug(str(lang)) for lang in values_as_list(languages)}
        normalised = self.df["language"].astype(str).str.lower().map(_language_slug)
        mask = normalised.isin(wanted)
        return self._clone(self.df[mask], f"language in {sorted(wanted)}")

    def filter_by_tool(self, tools: str | Iterable[str], cwes: str | Iterable[str] | None = None) -> "FeastDataset":
        tools = values_as_list(tools)
        missing = [tool for tool in tools if tool not in self.tool_columns]
        if missing:
            raise ValueError(f"Unknown tool column(s): {missing}. Available: {self.tool_columns}")

        wanted_cwes = normalise_cwes(cwes)
        mask = pd.Series(False, index=self.df.index)
        for tool in tools:
            if wanted_cwes:
                mask |= self.df[tool].map(lambda values: bool(normalise_cwes(as_list(values)) & wanted_cwes))
            else:
                mask |= self.df[tool].map(lambda values: len(as_list(values)) > 0)
        detail = f"tools {tools}" if not wanted_cwes else f"tools {tools} with {sorted(wanted_cwes)}"
        return self._clone(self.df[mask], detail)

    def filter_by_cwe(self, cwes: str | Iterable[str], where: str = "label", match: Literal["any", "all"] = "any") -> "FeastDataset":
        wanted = normalise_cwes(cwes)
        if not wanted:
            raise ValueError("Provide at least one valid CWE ID, e.g. 'CWE-79'.")

        def keep(row) -> bool:
            present = set(self._row_cwes(row, where=where))
            return bool(present & wanted) if match == "any" else wanted <= present

        mask = self.df.apply(keep, axis=1)
        return self._clone(self.df[mask], f"{where} CWE {match} {sorted(wanted)}")

    def filter_by_min_cwe_count(self, min_count: int, cwes: str | Iterable[str] | None = None, where: str = "label") -> "FeastDataset":
        if min_count < 1:
            raise ValueError("min_count must be >= 1")

        counts = self.cwe_counts(where=where)
        allowed = {cwe for cwe, count in counts.items() if count >= min_count}
        requested = normalise_cwes(cwes) if cwes is not None else None
        if requested is not None:
            allowed &= requested
        if not allowed:
            return self._clone(self.df.iloc[0:0], f"{where} CWE count >= {min_count}")

        mask = self.df.apply(lambda row: bool(set(self._row_cwes(row, where=where)) & allowed), axis=1)
        return self._clone(self.df[mask], f"{where} CWE count >= {min_count}")

    def filter_by_cwe_occurrences(self, cwe: str, min_occurrences: int = 1, where: str = "tools") -> "FeastDataset":
        wanted = normalise_cwe(cwe)
        if wanted is None:
            raise ValueError(f"Invalid CWE: {cwe!r}")
        if min_occurrences < 1:
            raise ValueError("min_occurrences must be >= 1")

        mask = self.df.apply(lambda row: self._row_cwes(row, where=where).count(wanted) >= min_occurrences, axis=1)
        return self._clone(self.df[mask], f"{wanted} occurs >= {min_occurrences} in {where}")

    def cwe_counts(self, where: str = "label") -> Counter:
        counts = Counter()
        for _, row in self.df.iterrows():
            counts.update(set(self._row_cwes(row, where=where)))
        return counts

    def cwe_counts_df(self, where: str = "label", min_count: int = 1) -> pd.DataFrame:
        counts = self.cwe_counts(where=where)
        rows = [(cwe, count) for cwe, count in counts.items() if count >= min_count]
        return pd.DataFrame(rows, columns=["cwe", "count"]).sort_values(["count", "cwe"], ascending=[False, True]).reset_index(drop=True)

    def branch_counts(self) -> pd.DataFrame:
        return self.df["branch"].value_counts(dropna=False).rename_axis("branch").reset_index(name="count")

    def dataset_counts(self, n: int | None = 25) -> pd.DataFrame:
        out = self.df["source"].value_counts(dropna=False).rename_axis("source").reset_index(name="count")
        return out if n is None else out.head(n)

    def tool_coverage(self) -> pd.DataFrame:
        rows = []
        total = len(self.df)
        for tool in self.tool_columns:
            if tool not in self.df.columns:
                continue
            hits = int(self.df[tool].map(lambda values: len(as_list(values)) > 0).sum())
            rows.append({"tool": tool, "rows_with_findings": hits, "coverage": hits / total if total else 0.0})
        return pd.DataFrame(rows).sort_values("rows_with_findings", ascending=False).reset_index(drop=True)

    def summary(self) -> dict:
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "languages": self.df["language"].value_counts(dropna=False).to_dict() if "language" in self.df else {},
            "branches": self.df["branch"].value_counts(dropna=False).to_dict() if "branch" in self.df else {},
            "tools": self.tool_columns,
            "filters": list(self.history),
            "source_paths": [str(path) for path in self.source_paths],
        }
