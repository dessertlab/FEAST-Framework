# FEAST

**Fusing Evidence Across Static Analysis Tools for CWE-Specific Vulnerability Detection**

Multi-language vulnerability dataset pipeline. Collects, normalises, and synthesises labelled code samples from 24 public sources across C/C++, Java, and Python, each annotated with CWE IDs from the MITRE catalogue. Then runs a full fusion experiment that calibrates per-tool reliability and compares every fusion strategy under cross-validation.

---

## Pipeline

```
Stage 0  Download         00_download_datasets.ipynb | main.py download
           Download all raw datasets  ->  data/raw/

Stage 1  Statistics       01_c_cpp.ipynb | 01_java.ipynb | 01_python.ipynb
           Per-language quality report  ->  outputs/stage1_<lang>_stats.xlsx

Stage 2  Synthesis        02_synthesis.ipynb | main.py synthesize
           CWE-filter + deduplicate    ->  data/processed/  and  data/merged/

Stage 3  Materialization  03_materialize.ipynb | main.py materialize
           Write source files          ->  data/materialized/

Stage 4  Tool enrichment  main.py enrich
           Collapse SAT JSON reports + merged parquets  ->  data/enriched/<lang>.parquet

Stage 5  Fusion           main.py fusion
           Calibrate reliability + compare fusion strategies  ->  data/results/<lang>/

Stage 5b Diagnostics      main.py diagnose
           Tool-complementarity analysis (oracle/diversity/CV)  ->  data/results/<lang>/diagnostics/

Stage 5c Replot           main.py plots
           Regenerate plots + CI report from existing fusion CSVs (no re-fusion)
            ->  data/results/<lang>/.../plots/
```

---

## Repository structure

```
FEAST/
├── main.py                          # CLI (see Usage below)
├── ingestion/                       # Stage 0–4 extraction library
│   ├── schema.py                    # FunctionSample dataclass
│   ├── cwe_navigator.py             # MITRE CWE XML parser and tree walker
│   ├── utils.py                     # shared helpers (CWE regex, NVD placeholders)
│   └── <source>.py                  # one extractor module per dataset
├── analysis/                        # Stage 5 fusion analysis library
│   ├── experiment.py                # end-to-end pipeline: canonicalise → fold → fuse → report
│   ├── calibration.py               # per-(tool, family) reliability metrics
│   ├── canonical.py                 # CWE → canonical family mapping (primary-path rule)
│   ├── folds.py                     # multilabel-stratified k-fold splitting
│   ├── aggregation.py               # fold-mean + support-weighted family aggregation
│   ├── complementarity.py           # oracle/diversity/CV tool-complementarity diagnostics
│   ├── reporting.py                 # CSV + plot writers
│   └── fusion/
│       ├── common.py                # shared plumbing (fire index, evidence_row, FUSER_ORDER)
│       ├── baselines.py             # single-tool + OR-of-N baselines
│       ├── traditional.py           # K-of-N voting
│       ├── weighted.py              # reliability-weighted voting (PPV/NPV, spec/sens, FPR/FNR)
│       ├── dst.py                   # Dempster-Shafer (Dempster, PCR6, Yager)
│       ├── bayes.py                 # naive Bayes over log-likelihood ratios
│       ├── bks.py                   # Behavior-Knowledge Space (empirical pattern lookup)
│       ├── logistic.py              # per-family logistic regression (+ pairwise interactions)
│       ├── ml.py                    # Decision Tree, Random Forest, Gradient Boosting, XGBoost
│       └── predictions.py           # evaluation helpers and tau-variant expansion
├── notebooks/
│   ├── 00_download_datasets.ipynb   # Stage 0 – download
│   ├── 01_c_cpp.ipynb               # Stage 1 – C/C++ statistics
│   ├── 01_java.ipynb                # Stage 1 – Java statistics
│   ├── 01_python.ipynb              # Stage 1 – Python statistics
│   ├── 02_synthesis.ipynb           # Stage 2 – process + merge
│   └── 03_materialize.ipynb         # Stage 3 – write source files
├── data/
│   ├── raw/                         # downloaded datasets (git-ignored)
│   ├── cwec_latest.xml              # MITRE CWE catalogue (auto-downloaded)
│   ├── processed/                   # per-dataset CWE-filtered parquets
│   │   ├── c_cpp/
│   │   ├── java/
│   │   └── python/
│   ├── merged/                      # final deduplicated parquets
│   │   ├── c_cpp_merged.parquet
│   │   ├── java_merged.parquet
│   │   └── python_merged.parquet
│   ├── materialized/                # individual source files for static analysis
│   │   ├── c_cpp/
│   │   │   ├── <dataset>/           # one directory per source dataset
│   │   │   │   └── <id>.c           # one file per sample
│   │   │   └── index.parquet        # sample_id -> source, label, cwes, branch
│   │   ├── java/
│   │   └── python/
│   ├── SAT-reports/                 # static-analysis JSON reports
│   ├── enriched/                    # per-language merged samples + per-tool CWE columns
│   └── results/                     # fusion experiment outputs
│       └── <lang>/pillar_child/
│           ├── config.json          # experiment parameters
│           ├── canonical_map.csv    # raw CWE -> canonical family
│           ├── folds.csv            # per-row fold assignment
│           ├── calibration_reliability.csv  # per-(tool,family,fold) reliability metrics
│           ├── fusion_metrics_per_family.csv  # per-(strategy,family) metrics (mean over folds)
│           ├── fusion_metrics_overall.csv     # support-weighted aggregate per strategy
│           ├── fusion_detection_overall.csv   # vuln/safe detection metrics per strategy
│           ├── fusion_tau_sweep.csv           # all (strategy, tau) combinations
│           ├── fusion_operating_points.csv    # best-F1 tau per strategy
│           ├── fusion_tau_selection_by_fold.csv  # tau picked per outer fold
│           ├── fusion_tau_selected.csv        # aggregated nested tau selection (reused by `plots`)
│           ├── fusion_score_histogram.csv     # (score -> n, n_positive) per fold/strategy/family
│           └── plots/               # per-metric plots, per-family heatmaps, CI forest plot
├── outputs/
│   ├── stage1_c_cpp_stats.xlsx
│   ├── stage1_java_stats.xlsx
│   └── stage1_python_stats.xlsx
├── tests/                           # unit tests
└── pyproject.toml
```

---

## Datasets

25 sources organised by language and branch.

### C/C++ — 11 sources

| Dataset | Branch | Positives | Negatives |
|---------|--------|-----------|-----------|
| PrimeVul | real | `target=1`, single-function commit, CWE non-empty | all `target=0` (explicit) |
| ICVul | real | `before_change=True`, `fc_hash` in CVE-FC mapping | none |
| CVEfixes(C) | real | C/C++ language, single-function commit, CWE non-empty | none |
| MegaVul | real | single-function commit, CWE non-empty | none |
| SecVulEval | real | `is_vulnerable=True` | all `is_vulnerable=False` |
| CrossVul(C) | real | `bad_*` files (vulnerable functions) | `good_*` files (fix-paired) |
| SVEN(C) | real | `func_src_before`, CWE from `vul_type` | `func_src_after` (fix-paired) |
| Juliet(C) | synth | `*_bad.c` files | `*_good*.c` files |
| CASTLE | synth | `vulnerable=True` | `vulnerable=False` |
| FormAI | ai | `VULNERABLE` / ESBMC `error_type` mapped to CWE | `NON-VULNERABLE` / safe samples |
| LLMSecEval(C) | ai | `gen_scenario/*.c` (Copilot completions) | none |

### Java — 5 sources

| Dataset | Branch | Positives | Negatives |
|---------|--------|-----------|-----------|
| CVEfixes(Java) | real | Java language, single-function commit | none |
| CrossVul(Java) | real | `bad_*` files | `good_*` files (fix-paired) |
| Juliet(Java) | synth | `*_bad.java` files | `*_good*.java` files |
| OWASP(Java) | synth | `real vulnerability=true` | `real vulnerability=false` |
| CAPEC_LLM(Java) | ai | LLM-generated snippets for CAPEC entries | none |

### Python — 9 sources

| Dataset | Branch | Positives | Negatives |
|---------|--------|-----------|-----------|
| CVEfixes(Python) | real | Python language, single-function commit | none |
| PatchEval | real | `vul_func` where `language=Python` | `fix_func` (fix-paired) |
| CrossVul(Python) | real | `bad_*` files | `good_*` files (fix-paired) |
| PyVul | real | `code_before`, CWE from commits map | `code_after` (fix-paired) |
| SVEN(Python) | real | `func_src_before` | `func_src_after` (fix-paired) |
| OWASP(Python) | synth | `real vulnerability=true` | `real vulnerability=false` |
| LLMSecEval | ai | `gen_scenario/*.py` (Copilot completions) | `Secure/*.py` files |
| SecurityEval | ai | all samples (vulnerable-only dataset) | none |
| CAPEC_LLM(Python) | ai | LLM-generated snippets for CAPEC entries | none |

**Branch semantics:**

| Branch | Meaning |
|--------|---------|
| `real` | Functions extracted from actual CVE patches or real-world codebases |
| `synth` | Template/rule-based synthesised code (Juliet test suite, OWASP Benchmark) |
| `ai` | LLM-generated code (Copilot completions, ChatGPT-generated snippets) |

---

## Output schema

Every extractor returns `list[FunctionSample]`:

```python
@dataclass
class FunctionSample:
    code: str        # function body
    cwes: list[str]  # CWE IDs (e.g. ["CWE-79"]); empty list for label=0
    label: int       # 1 = vulnerable,  0 = safe
    branch: str      # "real" | "synth" | "ai"
    language: str    # "C/C++" | "Java" | "Python"
    sample_id: str   # stable content-derived ID: SHA-256(norm(code))[:16]
```

`sample_id` is assigned at extraction time via a registry-level wrapper and is stable across runs (content-derived, not positional).

Parquet files produced by Stage 2 add two columns:

| Column | Description |
|--------|-------------|
| `source` | Dataset name (e.g. `"PyVul"`) |
| `code_hash` | Full SHA-256 of normalised code (used for deduplication) |
| `sample_id` | First 16 hex chars of `code_hash`; used as filename stem in Stage 3 |

---

## Static analysis tools

The eight SATs, their versions, and the exact rulesets they were run with. Every tool is
configured to maximise security coverage and to operate without compilation artifacts,
using source-only extraction where the tool supports it. Reports land in
`data/SAT-reports/<lang>.json` and are collapsed into per-tool CWE columns by
`main.py enrich`.

| Tool | Version | C/C++ | Java | Python |
|------|---------|:-----:|:----:|:------:|
| CodeQL | 2.23.5 | ✓ | ✓ | ✓ |
| Semgrep | 1.163.0 | ✓ | ✓ | ✓ |
| Joern | 4.0.543 | ✓ | ✓ | — |
| Cppcheck | 2.10 | ✓ | — | — |
| Flawfinder | 2.0.20 | ✓ | — | — |
| Ikos | 3.5 | ✓ | — | — |
| Bandit | 1.9.4 | — | — | ✓ |
| Pylint | 4.0.5 | — | — | ✓ |

### CodeQL

Run in source-only extraction mode (`--build-mode=none`). Three query specifications per
language — the `security-and-quality` suite plus the security query directories, including
the experimental ones:

```
codeql/cpp-queries:codeql-suites/cpp-security-and-quality.qls
codeql/cpp-queries:Security/CWE
codeql/cpp-queries:experimental/Security/CWE

codeql/java-queries:codeql-suites/java-security-and-quality.qls
codeql/java-queries:Security/CWE
codeql/java-queries:experimental/Security/CWE

codeql/python-queries:codeql-suites/python-security-and-quality.qls
codeql/python-queries:Security
codeql/python-queries:experimental/Security
```

Note the asymmetry: the Python query pack uses a flat `Security` layout, the C/C++ and
Java packs nest under `Security/CWE`. Note also that `security-and-quality.qls` already
pulls in the non-experimental security queries, so the second line of each block is
redundant with the first; CodeQL deduplicates, so the overlap is harmless.

### Semgrep

Nineteen registry packs, grouped here by what they cover:

| Purpose | Packs |
|---------|-------|
| Language-specific | `p/c`, `p/cpp-audit`, `p/c-audit-banned-functions`, `p/java`, `p/python`, `p/flask` |
| General security | `p/default`, `p/security-audit`, `p/r2c-security-audit`, `p/r2c-bug-scan`, `p/secure-defaults`, `p/security-code-scan` |
| Standards-driven | `p/cwe-top-25`, `p/owasp-top-ten` |
| Targeted classes | `p/sql-injection`, `p/command-injection`, `p/security-headers`, `p/secrets` |

### Remaining tools

| Tool | Configuration |
|------|---------------|
| Joern | Built-in scanner query database; findings at severity `error` and `warning` only |
| Cppcheck | Warning-level checks enabled |
| Flawfinder | Minimum severity threshold of 3 out of 5, to suppress low-confidence findings |
| Ikos | Full suite of internal static analyzers |
| Bandit | Default built-in ruleset |
| Pylint | All non-security checks suppressed; only security-related message codes enabled |

### CWE attribution

CodeQL, Semgrep, Bandit, Cppcheck and Flawfinder report CWE identifiers natively in their
structured output. Pylint, Joern and Ikos emit tool-specific message codes or free text, so
each distinct rule was mapped to a CWE by hand against the MITRE catalogue and the tool
documentation — 89 rules in total (16 Pylint, 51 Joern, 22 Ikos). The mapping was produced
independently by three annotators and reconciled to full agreement; see
`rule_cwe_annotation_table.csv`.

---

## CWE classification

The pipeline classifies every CWE ID against `cwec_latest.xml` from MITRE:

| Type | Meaning |
|------|---------|
| `leaf` | Most specific weakness; no children in MITRE hierarchy |
| `non-leaf` | Parent or intermediate node (broad attribution) |
| `category` | MITRE organisational grouping, not a proper weakness |
| `deprecated` | Superseded weakness (name starts with `DEPRECATED:`) |
| `unknown` | ID not found in `cwec_latest.xml` |

Stage 2 retains only `leaf` and `non-leaf` by default. The Excel reports colour-code CWE columns by type (green, amber, purple, pink, gray).

---

## Setup

```bash
# recommended: install with uv
uv sync --all-extras

# alternative: pip
pip install -e ".[dev]"
```

Requires Python >= 3.10.

```bash
# run tests
uv run pytest
```

---

## Notebooks

Run in order. All notebooks are idempotent.

### `00_download_datasets.ipynb` — Stage 0

Downloads all 17 raw datasets to `data/raw/`. Each section skips if the target path already exists. Run this **once** before any other notebook. Equivalent to `main.py download`.

Two datasets require manual download from Zenodo and cannot be fetched programmatically:
- **CrossVul** — place `crossvul.zip` at `data/raw/crossvul.zip`
- **LLMSecEval (vulnerable)** — place `copilot-cwe-scenarios-dataset.zip` at `data/raw/copilot-cwe-scenarios-dataset.zip` (Zenodo record 5225651)

### `01_c_cpp.ipynb` / `01_java.ipynb` / `01_python.ipynb` — Stage 1

Per-language quality analysis. For each source dataset:

1. Extracts `FunctionSample` collections via the `ingestion/` library
2. Computes total, vulnerable, and safe sample counts
3. Builds a per-CWE distribution matrix
4. Classifies every CWE ID by MITRE type

**Output:** `outputs/stage1_{c_cpp,java,python}_stats.xlsx`

Each workbook contains:
- One sheet per branch: `{lang}_Real`, `{lang}_Synth`, `{lang}_AI`
- A `Filters` sheet documenting extraction methodology for every source
- A `Legend` sheet explaining CWE colour coding

### `02_synthesis.ipynb` — Stage 2

For each language:

1. **Process** — applies the CWE filter (leaf + non-leaf only by default) to every source; saves one parquet per dataset to `data/processed/<lang>/`
2. **Merge** — concatenates all processed datasets, deduplicates by SHA-256 of normalised code (higher-quality branch wins: `real > synth > ai`); saves the result to `data/merged/<lang>_merged.parquet`

### `03_materialize.ipynb` — Stage 3

For each language reads `data/merged/<lang>_merged.parquet` and writes:

- One source file per sample: `data/materialized/<lang>/<dataset>/<sample_id>.<ext>`
- A lookup index: `data/materialized/<lang>/index.parquet` mapping `sample_id` to `source`, `label`, `cwes`, `branch`, and `code_hash`

Files are skipped if they already exist (set `OVERWRITE = True` to force re-write). Existing files are never deleted.

> **Java note:** Samples are function bodies extracted from CVE patches, not compilable top-level classes. Static analysis tools that require a full compilation unit will need an additional wrapping step.

### `main.py enrich` — Stage 4

Reads SAT report JSON files from `data/SAT-reports/` and merged parquet files from `data/merged/`, then writes one enriched parquet per language to `data/enriched/<lang>.parquet` (`c_cpp.parquet`, `java.parquet`, `python.parquet`). The output preserves the merged dataset columns and adds one list-valued column per static-analysis tool that actually ran for that language. Each tool column contains the unique CWE IDs reported by that tool for the sample, or an empty list when the tool reported no CWE for that sample.

Matching primarily uses the materialized path shape `<dataset>/<sample_id>.<ext>` and falls back to `sample_id` within the same language when older report paths differ slightly. The enrich step reads the per-sample `runs[].tools` section first so tools with zero findings are still represented, and then folds in `findings` as additional evidence.

---

## CLI

`main.py` exposes the full pipeline from the command line.

### `download` — fetch all datasets (Stage 0)

```bash
uv run python main.py download
```

Downloads all 17 datasets to `data/raw/`. Idempotent: already-present paths are skipped. Two datasets require manual download from Zenodo; the command prints instructions for these when they are missing:

| Dataset | File to place in `data/raw/` |
|---------|------------------------------|
| CrossVul | `crossvul.zip` |
| LLMSecEval (vulnerable) | `copilot-cwe-scenarios-dataset.zip` (Zenodo record 5225651) |

### `list` — show all available sources

```bash
uv run python main.py list
```

### `synthesize` — build processed and merged parquets

```bash
uv run python main.py synthesize [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--lang LANG` | `all` | Language to process: `c`, `java`, `python`, or `all` |
| `--sources SRC1,SRC2,...` | all available | Comma-separated dataset names to include |
| `--cwe-types TYPES` | `leaf,non-leaf` | CWE node types to retain for vulnerable samples |
| `--cwes CWE-79,CWE-89,...` | all | Explicit whitelist of CWE IDs |
| `--min-cwe-count N` | `1` (off) | Drop CWEs with fewer than N vulnerable samples after merge |
| `--branches BRANCHES` | `all` | Source branches to include: `real`, `synth`, `ai`, or comma-separated |
| `--data-dir DIR` | `data/raw/` | Override the raw data directory |

`--cwe-types`, `--cwes`, and `--min-cwe-count` compose independently: a vulnerable sample is kept only if it satisfies all active filters simultaneously.

**Examples:**

```bash
# full pipeline, all languages, default filters
uv run python main.py synthesize

# Python only
uv run python main.py synthesize --lang python

# select specific sources (exact names or slug-style both accepted)
uv run python main.py synthesize --lang python --sources "CVEfixes(Python),PyVul,PatchEval"
uv run python main.py synthesize --lang c --sources primevul,icvul,secvuleval

# keep only the most specific CWEs
uv run python main.py synthesize --cwe-types leaf

# target a specific set of CWEs
uv run python main.py synthesize --cwes CWE-79,CWE-89,CWE-22,CWE-78

# drop CWEs that appear in fewer than 20 vulnerable samples (after merge)
uv run python main.py synthesize --min-cwe-count 20

# exclude AI-generated data
uv run python main.py synthesize --branches real,synth

# compose multiple filters
uv run python main.py synthesize \
    --lang python \
    --branches real \
    --cwe-types leaf \
    --min-cwe-count 10
```

Output is written to `data/processed/<lang>/` (one parquet per source) and `data/merged/<lang>_merged.parquet` (final deduplicated dataset). Both directories are created automatically if they do not exist.

### `materialize` — write source files for static analysis

```bash
uv run python main.py materialize [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--lang LANG` | `all` | Language to materialize: `c`, `java`, `python`, or `all` |
| `--overwrite` | off | Re-write files that already exist |

Reads `data/merged/<lang>_merged.parquet` and writes one file per sample to `data/materialized/<lang>/<dataset>/<sample_id>.<ext>`. Also writes a per-language `index.parquet` lookup table. Existing files are skipped unless `--overwrite` is set.

**Examples:**

```bash
# materialize all languages
uv run python main.py materialize

# Python only
uv run python main.py materialize --lang python

# force re-write all existing files
uv run python main.py materialize --overwrite
```

### `enrich` — add SAT results to merged samples

```bash
uv run python main.py enrich [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--merged-dir DIR` | `data/merged/` | Directory containing merged parquet files |
| `--reports-dir DIR` | `data/SAT-reports/` | Directory containing SAT report JSON files |
| `--out-dir DIR` | `data/enriched/` | Output directory for per-language parquet files |

Reads every `*.parquet` in `data/merged/` and every language-named `*.json` in `data/SAT-reports/`, then writes one parquet per language with one additional list-valued column per tool that ran for that language.

**Examples:**

```bash
# default enrichment
uv run python main.py enrich

# custom output directory
uv run python main.py enrich --out-dir data/enriched_experiment
```

### `fusion` — calibrate reliability and compare fusion strategies (Stage 5)

```bash
uv run python main.py fusion [OPTIONS]
# aliases: fuse, f, analyze
```

Canonicalises CWE IDs to the direct children of CWE-1000 pillars (primary-path rule), calibrates per-(tool, family) reliability with exact matching, and runs every fusion strategy under stratified k-fold cross-validation. Results are written to `data/results/<lang>/pillar_child/`.

| Option | Default | Description |
|--------|---------|-------------|
| `--lang LANG` | `all` | Language: `c`, `java`, `python`, or `all` |
| `--exclude TOOL1,...` | none | Comma-separated tools to drop from the ensemble |
| `--n-splits N` | `5` | Number of cross-validation folds |
| `--tier TIER` | `base` | Analysis tier (see below) |
| `--min-cwe-count M` | set by `--tier` | Override the tier's family support floor |
| `--threshold K` | `2` | K for the traditional K-of-N voting baseline |
| `--calibration M1,...` | all pairs | Calibration metric pairs to include: `ppv`, `npv`, `sensitivity`, `specificity`, `fpr`, `fnr`. The paper (Eq. 2-3, 6-7) uses only `sensitivity,specificity` — pass it explicitly to reproduce the published numbers; the default additionally explores the `ppv`/`npv` and `fpr`/`fnr` pairs as an ablation. |
| `--taumin T` | `0.1` | Lower bound of the τ sweep grid |
| `--taumax T` | `0.9` | Upper bound of the τ sweep grid |
| `--seed S` | `42` | Random seed for fold assignment |

#### Analysis tiers

The `--tier` flag controls two things simultaneously: the minimum number of ground-truth occurrences required to include a CWE family, and which ML-based strategies are activated.

| Tier | Family support floor | ML strategies added | Use when |
|------|---------------------|---------------------|----------|
| `base` | ≥ n\_splits (default 5) | none | Maximum CWE coverage; existing strategies only |
| `medium` | ≥ 30 | Decision Tree | Balanced coverage + one ML baseline |
| `full` | ≥ 100 | Decision Tree + Random Forest + Gradient Boosting + XGBoost | Highest-confidence families only; full ML comparison |

The ML classifiers use tool fire indicators (one binary feature per tool) as input and are trained on the calibration split of each fold. Class imbalance is handled via `class_weight='balanced'` (DT, RF) or inverse-frequency sample weights (GB), replacing SMOTE which is inapplicable on binary feature spaces. Hyperparameters are adapted from D'Abruzzo Pereira et al. (2024) to the 4-binary-feature regime of FEAST (see `analysis/fusion/ml.py`).

`--min-cwe-count` overrides the tier's support floor if you need a custom threshold. The tier still determines which ML strategies are included.

**Outputs** written to `data/results/<lang>/pillar_child/`:

| File | Description |
|------|-------------|
| `config.json` | All experiment parameters |
| `canonical_map.csv` | Raw CWE → canonical family mapping |
| `folds.csv` | Per-row fold assignment |
| `calibration_reliability.csv` | Per-(tool, family, fold): TP/FP/TN/FN, PPV, NPV, FPR, FNR, sensitivity, specificity |
| `fusion_metrics_per_family.csv` | Per-(strategy, family): precision, recall, F1, F2, MCC, ROC-AUC (mean over folds) |
| `fusion_metrics_overall.csv` | Support-weighted aggregate per strategy |
| `fusion_detection_overall.csv` | Vuln/safe binary detection metrics per strategy |
| `fusion_tau_sweep.csv` | All (base\_strategy, τ) combinations |
| `fusion_operating_points.csv` | Best-F1 τ per strategy (F1 is the pipeline's only τ criterion) |
| `fusion_tau_selection_by_fold.csv` | τ chosen independently in each outer fold (fold-to-fold variability) |
| `fusion_tau_selected.csv` | The aggregated nested τ selection actually used for the report; `plots` reuses it |
| `fusion_score_histogram.csv` | Per (fold, strategy, family, score): how many rows carry that score and how many are positive. The sufficient statistic for any threshold metric, so τ sweeps, PR curves and operating-point questions are answerable without re-running fusion |
| `plots/mean_difference_ci_f1.csv` | Paired per-family comparison vs the 2ooN baseline: Wilcoxon p, Hodges–Lehmann, 95% CI, support-weighted difference and effective N |
| `plots/` | Per-metric bar charts, per-family heatmaps and the CI forest plot |

**Metric aggregation.** Every headline number is built in two explicit stages
(`analysis/aggregation.py`), and nothing in the repo re-implements them:

1. **mean over folds** — per `(strategy, family)`, average each metric over the 5 held-out
   folds; family support is *summed*, which equals the family's total ground-truth
   occurrences because the folds partition the rows exactly once.
2. **support-weighted mean over families** — per `(strategy, metric)`, weight the family
   values by that support. This is the headline (`*_weighted`); the unweighted macro mean
   (`*_macro`) and the median (`*_median`) are kept as context.

Two consequences worth knowing before reading any CSV:

* **`*_weighted` and `*_macro` answer different questions.** The headline table and the
  τ selection are weighted and macro respectively: τ selection and the Wilcoxon test in
  `plots/mean_difference_ci_f1.csv` treat each CWE family as one observation, while the
  overall table weights by support. The CI report carries both (`mean_difference` and
  `weighted_difference`) plus `n_effective`, Kish's effective family count, because the
  support distribution is concentrated (on C/C++ one family holds ~44% of the positives).
* **`f1_weighted` is not recomputable from `precision_weighted` and `recall_weighted`.**
  F1 is computed inside each family from its own TP/FP/FN and then averaged; the harmonic
  mean of two separately averaged columns is a different number. The columns are not
  meant to reconcile.

**Missing values** follow one convention, `analysis.aggregation.zero_fill`, applied at every
aggregation site. Metrics that go NaN because the *strategy* produced a degenerate
prediction vector (`precision`, `npv`, `f1`, `f2`, `mcc`) are scored 0.0 — abandoning a
family is a failure, not a gap. Metrics that go NaN because the *(family, fold) slice* is
degenerate (`recall`/`fnr` with no positives, `specificity`/`fpr` with no negatives,
`roc_auc`/`pr_auc` needing both) stay NaN and drop out of the average, identically for
every strategy.

**τ selection** uses F1 and nothing else. It runs on a nested split *inside* the
calibration fold (`cal_fit`/`cal_tune`, 75/25), never on the held-out fold being reported;
the chosen variants are persisted to `fusion_tau_selected.csv` so `main.py plots`
reproduces the same report instead of re-picking τ on held-out data.

**Fusion strategies** included in every tier:

| Strategy | Type |
|----------|------|
| `tool:<name>` | Single-tool baseline (one per tool) |
| `or_1_of_N` | OR of all tools |
| `traditional_K_of_N` | K-of-N majority vote |
| `weighted_fire_<fire>_silence_<silence>` | Reliability-weighted voting (3 metric pairs) |
| `dst_<rule>_fire_<fire>_silence_<silence>` | Dempster-Shafer (Dempster, PCR6, Yager × 3 pairs) |
| `naive_bayes` | Naive Bayes over log-likelihood ratios |
| `bks` | Behavior-Knowledge Space (empirical pattern lookup) |
| `logistic_regression` | Per-family logistic regression on fire indicators |
| `logistic_interactions` | Same + pairwise tool-interaction features |

**Examples:**

```bash
# base tier: all CWE families, existing strategies
uv run python main.py fusion --lang python

# medium tier: families with ≥ 30 samples, adds Decision Tree
uv run python main.py fusion --lang python --tier medium

# full tier: families with ≥ 100 samples, adds DT + RF + GB
# (note: this alone does not reproduce the paper's numbers -- see below)
uv run python main.py fusion --lang python --tier full

# reproduce the paper's headline results (Fig. 3/4): full tier restricted to the
# sensitivity/specificity calibration pair the paper's equations use, all languages
uv run python main.py fusion --tier full --calibration sensitivity,specificity

# exclude one tool
uv run python main.py fusion --lang python --exclude pylint

# custom min-cwe-count (overrides the tier floor)
uv run python main.py fusion --lang python --tier full --min-cwe-count 50

# restrict calibration metrics
uv run python main.py fusion --lang python --calibration ppv,npv
```

### `diagnose` — tool-complementarity diagnostics (Stage 5b)

```bash
uv run python main.py diagnose [OPTIONS]
# alias: diag
```

Runs tool-complementarity diagnostics on the same canonicalised data used by `fusion`: oracle/coverage headroom, error diversity, per-(tool, family) reliability heatmap, and cross-validated marginal contribution and conditional value per tool. Outputs are written to `data/results/<lang>/pillar_child/diagnostics/`.

| Option | Default | Description |
|--------|---------|-------------|
| `--lang LANG` | `all` | Language: `c`, `java`, `python`, or `all` |
| `--exclude TOOL1,...` | none | Comma-separated tools to exclude |
| `--n-splits N` | `5` | Number of cross-validation folds |
| `--min-cwe-count M` | `= n_splits` | Min GT occurrences per family |
| `--seed S` | `42` | Random seed |

**Examples:**

```bash
# diagnostics for all languages
uv run python main.py diagnose

# Python only
uv run python main.py diagnose --lang python

# exclude a tool before computing marginal contributions
uv run python main.py diagnose --lang python --exclude devaic
```

