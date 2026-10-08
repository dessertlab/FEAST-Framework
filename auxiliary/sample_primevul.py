"""
One-off script: filter PrimeVul safe samples in c_cpp_merged.parquet.
Keeps all vuln (label=1) PrimeVul rows + 10 000 random safe (label=0) PrimeVul rows (seed=42).
All other dataset rows are untouched.
The original file is renamed to c_cpp_merged_unsampled.parquet.
"""

import shutil
from pathlib import Path

import pandas as pd

PARQUET_PATH = Path("data/merged/c_cpp_merged.parquet")
UNSAMPLED_PATH = PARQUET_PATH.with_stem(PARQUET_PATH.stem + "_unsampled")
N_SAFE = 10_000
SEED = 42

df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df):,} rows from {PARQUET_PATH}")

is_primevul = df["source"] == "PrimeVul"
primevul_vuln = df[is_primevul & (df["label"] == 1)]
primevul_safe_all = df[is_primevul & (df["label"] == 0)]
other = df[~is_primevul]

print(f"PrimeVul vuln: {len(primevul_vuln):,}")
print(f"PrimeVul safe (before): {len(primevul_safe_all):,}")
print(f"Other datasets: {len(other):,}")

primevul_safe_sampled = primevul_safe_all.sample(n=N_SAFE, random_state=SEED)

result = pd.concat([other, primevul_vuln, primevul_safe_sampled], ignore_index=True)

print(f"\nPrimeVul safe (after): {len(primevul_safe_sampled):,}")
print(f"Total rows after sampling: {len(result):,}")

shutil.move(PARQUET_PATH, UNSAMPLED_PATH)
print(f"\nOriginal renamed to: {UNSAMPLED_PATH}")

result.to_parquet(PARQUET_PATH, index=False)
print(f"Sampled parquet saved to: {PARQUET_PATH}")
