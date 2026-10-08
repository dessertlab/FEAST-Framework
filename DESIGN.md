# FEAST — Design of the analysis pipeline (canonical-family fusion)

This document records the methodology and module layout of the analysis half of FEAST
(Stage 5 calibration + fusion). The ingestion/synthesis/enrichment half is documented in
`README.md` and is unchanged.

## 1. Why canonical families

The earlier pipeline matched tool output and ground truth with an **asymmetric** rule:
the tool side required the *exact* CWE, the ground-truth side accepted any *vertically
related* CWE (ancestor/descendant closure). On a per-CWE one-vs-rest grid this caused:

* **TP+FN double scoring** — a tool firing CWE-564 on a CWE-89 row scored a TP for 564
  *and* an FN for 89, inflating false negatives (the metric the project targets);
* **no evidence pooling** — related CWEs were voted in isolation, so a tool calibrated on
  a parent CWE never contributed to a sibling/child during fusion, even though
  calibration had credited it for exactly those cases. Calibration and fusion disagreed.

**Fix:** collapse both tool outputs and ground truth to a **CWE family** — an ancestor in
the MITRE CWE-1000 (Research Concepts) hierarchy — and then match families **exactly**.
The hierarchy is consulted in exactly one place (canonicalisation); everything downstream
is plain set membership, identical in calibration and fusion.

## 2. Family definition (primary-path rule)

CWE-1000 is a DAG: some weaknesses have multiple `ChildOf` parents. To get a
deterministic partition we follow only the parent marked `Ordinal="Primary"` within
`View_ID="1000"` (`CWENavigator.primary_parent` / `primary_path`). This discards
misleading cross-view edges (for example CWE-120's CWE-20 parent in view 700).

The analysis now uses exactly one canonical family level: **the direct children of
CWE-1000 pillars**. For each raw CWE, follow its primary path up to a Pillar and map it
to the node immediately below that Pillar:

| primary path | family |
|--------------|--------|
| 79 -> 74 -> 707 | CWE-74 |
| 89 -> 943 -> 74 -> 707 | CWE-74 |
| 120 -> 787 -> 119 -> 118 -> 664 | CWE-118 |
| 1024 -> 697 | CWE-1024 |

Pillars themselves are not analysis families: the target family set is only the
first-order children below pillars. The previous `pillar`, `subcategory`, and `class`
experiment levels were removed; this single `pillar_child` rule is now the only
canonicalisation strategy.

## 3. Matching, confusion, calibration

After canonicalisation a tool "fires" family `f` iff `f` is in its family set, and the GT
is positive for `f` iff `f` is in the GT family set — **exact, symmetric** membership.
`analysis/calibration.py` builds the one-vs-rest confusion per `(tool, family)`
(vectorised with boolean incidence matrices) and derives `ppv` (fire credibility), `npv`
(silence credibility), `fpr`/`fnr`, their complements, `positive_support` and `supported`
(`tp+fp>0`). These reliability numbers are the fusion configuration.

## 4. Fusion strategies (`analysis/fusion/`)

One package, one strategy per module, sharing the plumbing in `common.py` (fire index,
exact labels, metric lookup, prediction schema):

* `weighted.py` — reliability-weighted voting (3 fire/silence metric pairs: ppv/npv, specificity/sensitivity, fpr/fnr);
* `traditional.py` — K-of-N baseline over all tools;
* `dst.py` — Dempster-Shafer (Dempster, PCR6, Yager) on the binary frame {V,S}, expanded over the same 3 fire/silence metric pairs;
* `bayes.py` — naive Bayes over per-tool log-likelihood ratios;
* `bks.py` — empirical behavior-knowledge-space lookup over tool fire patterns;
* `logistic.py` — per-family logistic regression on the fire indicators.

Noisy-OR is intentionally not part of the supported strategy set. Every scored fusion
strategy is materialised at the discrete thresholds `τ ∈ {0.1, …, 0.9}` as explicit
strategy names (for example `naive_bayes_tau_0_7`), so per-family, detection, plots and
CSV reports all operate on the same strategy identifiers. Baseline single-tool, OR, and
traditional K-of-N rows remain unswept discrete references. Every strategy emits the same
per-(row, family) schema, so they all feed `predictions.evaluate_predictions` (metrics per
`(strategy, family)`) and `detection_from_predictions` (per-row vuln/safe collapse +
`cwe_attribution_acc`).

## 5. Two-stage aggregation (`analysis/aggregation.py`)

1. **mean over folds** — average each metric across the K folds per `(strategy, family)`;
   family support is *summed* (→ total GT occurrences).
2. **support-weighted mean over families** — per `(strategy, metric)`, weight the family
   values by that support. This weighted value is the headline; the macro mean and median
   are kept as context.

**Missing-value convention.** A metric is NaN when its defining ratio is 0/0, and the two
causes are handled differently (`aggregation.zero_fill`, the single definition used by
both stages, and by tau selection and the paired test in `mean_difference_ci`):

* *strategy failure* — `precision`, `npv`, `f1`, `f2`, `mcc` go NaN because the strategy's
  own prediction vector is degenerate (no positive predictions, all-positive, or
  constant). "No detection" is a failure, not a missing observation, so these are filled
  with `0.0` at every aggregation site. Otherwise a family a strategy abandons drops out
  of its own average instead of lowering it, which biases comparisons between strategies
  that fail on different families.
* *degenerate group* — `recall`/`fnr` (no positives in the slice), `specificity`/`fpr` (no
  negatives), `roc_auc`/`pr_auc` (needs both) go NaN as a property of the (family, fold)
  slice itself, identically for every strategy. These stay NaN and are dropped from the
  reduction; filling them would invent a score nobody earned, and for the
  lower-is-better `fpr`/`fnr` it would invent a perfect one.

**Open point — weighted vs unweighted.** The headline table is support-weighted, while
tau selection and the Wilcoxon/HL test are **unweighted over families** (one family, one
observation). Both are internally consistent, but they are different quantities and must
not be read against each other: on C/C++ the support-weighted gap over the 2ooN baseline
is roughly 3x the macro one. Which of the two becomes *the* reported quantity is not
settled yet.

**Score persistence.** Per-row scores are not written out — for C/C++ that is ~14M rows
per language — but their *sufficient statistic* is: `fusion_score_histogram.csv` records,
per `(fold, strategy, family, score)`, how many rows carry that score and how many are
positive. Every threshold metric is a function of (TP, FP, FN) at a cut, and those follow
from that table, so a τ sweep, a PR curve or "does any operating point dominate the
baseline" is exact arithmetic on a few hundred KB instead of a two-hour re-run.
`analysis.fusion.metrics_from_histogram` does the recovery; a test asserts it matches
metrics computed on the rows.

## 6. Restriction (`analysis/experiment.py`)

A family is analysed iff it is **supported** (fired by ≥1 tool) **and** has **≥K
ground-truth occurrences** (`K = --min-cwe-count`, default `= --n-splits`; the necessary
condition for a per-fold-stable confusion matrix). The header prints the kept/present
family counts, unsupported families, below-K families, and retained GT-occurrence coverage.

## 7. Folding

`analysis/folds.py` — multilabel-stratified k-fold (`iterative-stratification`, a hard
dependency for reproducibility) on the **canonical family** labels, with rare-family
pruning at the same K. `before`/`after` fix-paired rows are split per-row (documented as
*correlated observations*, not leakage, because fusion features are tool verdicts, not
code).

## 8. Module map & outputs

```
ingestion/cwe_navigator.py   primary_parent / primary_path / abstraction (View_ID+Ordinal)
analysis/canonical.py        CWE -> direct child below CWE-1000 pillar
analysis/folds.py            stratified k-fold + rare-family pruning
analysis/calibration.py      exact per-(tool, family) confusion + reliability
analysis/fusion/             strategies + predictions + detection
analysis/aggregation.py      two-stage aggregation
analysis/reporting.py        CSV + SVG/PNG writers
analysis/experiment.py       CV orchestration per language at the single canonical level
main.py  fusion              thin CLI entry point
```

Outputs per `data/results/<language>/pillar_child/`: `config.json` (run config + restriction
stats), `canonical_map.csv`, `folds.csv`, `calibration_reliability.csv`,
`fusion_metrics_per_family.csv`, `fusion_metrics_overall.csv`,
`fusion_detection_overall.csv`, `fusion_tau_sweep.csv`, `fusion_operating_points.csv`,
`plots/<metric>.{svg,png}`, and `plots/best_variants/<metric>.{svg,png}`. The
`fusion_tau_sweep.csv` and operating-point reports are
derived from the explicit `*_tau_*` strategy rows rather than from a hidden post-hoc
thresholding pass.

## 9. Key decisions

* Matching is **exact on families**; the old vertical-closure matcher is removed.
* Support weight = `positive_support` (GT occurrences), summed across folds.
* The only canonical level is `pillar_child`: direct children of CWE-1000 pillars.
* Restriction = supported ∧ ≥K GT occurrences.
* Superseded modules (`metrics`, `split`, `report`, `visualization`, monolithic `fusion`)
  were removed rather than deprecated; the strategy math was preserved and relocated.
