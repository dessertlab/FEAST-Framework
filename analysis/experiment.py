"""End-to-end fusion experiment for one language at the canonical CWE level.

Pipeline (everything operates on canonical families with exact matching):

    load enriched  ->  canonicalise GT + tool columns  ->  build family grid
      ->  restrict to supported & frequent families  ->  k-fold stratified on families
      ->  per fold: calibrate reliability  ->  run fusion strategies  ->  evaluate
      ->  aggregate (mean over folds, then support-weighted over families)  ->  write outputs

The same canonicalisation is applied once up front, so calibration and fusion always see
the identical family space — there is no asymmetry between the two phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from analysis.aggregation import aggregate_fusion_metrics, mean_over_folds
from analysis.calibration import compute_reliability, gt_family_support
from analysis.canonical import CweCanonicalizer, all_raw_cwes
from analysis.dataset import as_list, load_enriched
from analysis.folds import multilabel_stratified_kfold, split_train_validation
from analysis.family_performance_plots import save_family_performance_plots
from analysis.fusion import (
    TIER_MIN_COUNT,
    detection_from_predictions,
    detection_metrics,
    evaluate_predictions,
    order_by_strategy,
    run_fusion,
)
from analysis.fusion.common import (
    canonical_strategy_order,
    metric_pairs_for_calibration,
    split_tau_strategy,
    taus_in_range,
)
from analysis.fusion.predictions import score_histogram, tau_variant_table
from analysis.mean_difference_ci import best_tau_per_strategy, fusion_strategies, save_mean_difference_ci_report
from analysis.reporting import save_best_variant_plots, save_config, save_csv, save_strategy_plots

console = Console()


def _select_tools(available: list[str], exclude: list[str]) -> list[str]:
    excluded = {t.strip().lower() for t in exclude if t.strip()}
    return [t for t in available if t.lower() not in excluded]


def _present_families(df: pd.DataFrame, tools: list[str]) -> list[str]:
    present: set[str] = set()
    for column in ["cwes", *tools]:
        if column in df.columns:
            for values in df[column]:
                present |= set(as_list(values))
    return sorted(present, key=lambda c: int(str(c).removeprefix("CWE-")))


def _fired_families(df: pd.DataFrame, tools: list[str]) -> set[str]:
    fired: set[str] = set()
    for tool in tools:
        if tool in df.columns:
            for values in df[tool]:
                fired |= set(as_list(values))
    return fired


@dataclass
class CanonicalPrep:
    """Everything the analysis needs after canonicalisation + restriction + folding.

    Shared by the fusion experiment and the complementarity diagnostics so both see the
    identical family space, tool set and folds.
    """
    language: str
    level: str
    df: pd.DataFrame            # original enriched rows (raw CWEs)
    cdf: pd.DataFrame           # canonicalised: GT + tool columns hold families
    tools: list[str]
    families: list[str]
    folds: pd.DataFrame
    restriction: dict
    canonical_map: pd.DataFrame
    support: dict[str, int]


def prepare_canonical(
    language: str,
    level: str,
    *,
    n_splits: int = 5,
    min_cwe_count: int | None = None,
    exclude: list[str] | None = None,
    seed: int = 42,
    enriched_dir: str | Path = "data/enriched",
) -> CanonicalPrep:
    """Load + canonicalise + restrict + fold one (language, level), without running fusion."""
    exclude = exclude or []
    min_cwe_count = n_splits if min_cwe_count is None else min_cwe_count

    dataset = load_enriched(language, enriched_dir=enriched_dir)
    df = dataset.to_pandas().reset_index(drop=True)
    tools = _select_tools(list(dataset.tool_columns), exclude)
    if language == "python":
        # DEVAIC labels its Python findings with its own "Top 25 CWE" list, which does
        # not map onto CWE-1000 (Research Concepts) -- so it can't be canonicalised to a
        # family and must always be dropped here, independent of the --exclude flag.
        tools = [t for t in tools if t.lower() != "devaic"]
    if not tools:
        raise ValueError(f"{language}/{level}: no tools left after exclusion")

    # canonicalise GT + tool columns to families (once)
    canon = CweCanonicalizer.from_xml(level)
    canonical_map = canon.canonical_map(all_raw_cwes(df, tools))
    cdf = canon.canonicalize_frame(df, tools)

    # family grid + restriction (supported AND >= K GT occurrences)
    present = _present_families(cdf, tools)
    fired = _fired_families(cdf, tools)
    support = gt_family_support(cdf, present)
    supported = {f for f in present if f in fired}
    frequent = {f for f in present if support.get(f, 0) >= min_cwe_count}
    families = sorted(supported & frequent, key=lambda c: int(c.removeprefix("CWE-")))

    total_gt_occ = sum(support.values())
    restriction = {
        "families_present": len(present),
        "families_kept": len(families),
        "dropped_unsupported": len([f for f in present if f not in supported]),
        "dropped_below_min_count": len([f for f in present if f in supported and f not in frequent]),
        "min_cwe_count": min_cwe_count,
        "gt_occurrences_total": total_gt_occ,
        "gt_occurrences_kept": sum(support[f] for f in families),
        "gt_occurrences_kept_frac": (sum(support[f] for f in families) / total_gt_occ) if total_gt_occ else 0.0,
    }

    # folds are reproducible from (cdf, n_splits, seed); empty if nothing survives
    folds = (
        multilabel_stratified_kfold(cdf, n_splits=n_splits, random_state=seed, min_cwe_count=min_cwe_count)
        if families else pd.DataFrame(columns=["row_index", "fold"])
    )
    return CanonicalPrep(language, level, df, cdf, tools, families, folds, restriction, canonical_map, support)


def _build_operating_points(
    tau_overall: pd.DataFrame,
    detection_overall: pd.DataFrame,
) -> pd.DataFrame:
    """Best-F1 τ per fuser strategy, plus baseline rows (traditional / OR) appended.

    F1 is the pipeline's single selection criterion, the same one used for the per-family
    comparison (``_select_tau_nested``) and for the paper's reported results; no other
    metric selects τ anywhere.

    Baselines have no τ sweep so they're taken directly from detection_overall with
    tau=NaN and base_strategy equal to the strategy name itself.
    """
    fuser_pts = (
        tau_overall.sort_values("f1", ascending=False, na_position="last")
        .groupby("base_strategy", as_index=False)
        .first()
        if not tau_overall.empty else tau_overall.iloc[0:0].copy()
    )

    baseline_mask = detection_overall["strategy"].str.match(
        r"^(or_1_of_|traditional_)"
    )
    baselines = detection_overall[baseline_mask].copy()
    if not baselines.empty:
        baselines.insert(1, "base_strategy", baselines["strategy"])
        baselines.insert(2, "tau", float("nan"))

    return pd.concat([fuser_pts, baselines], ignore_index=True)


TAU_SELECTION_FILE = "fusion_tau_selected.csv"


def _selected_tau_table(selected_strategies: list[str]) -> pd.DataFrame:
    """One row per base strategy: the tau-suffixed name chosen by the nested selection."""
    rows = []
    for strategy in selected_strategies:
        base, tau = split_tau_strategy(str(strategy))
        rows.append({"base_strategy": base, "tau": tau, "strategy": strategy})
    return pd.DataFrame(rows, columns=["base_strategy", "tau", "strategy"])


def _load_selected_tau(results_dir: Path) -> list[str] | None:
    """Read back the persisted nested selection, or None when the run predates it."""
    path = results_dir / TAU_SELECTION_FILE
    if not path.exists():
        return None
    table = pd.read_csv(path)
    if "strategy" not in table.columns or table.empty:
        return None
    return table["strategy"].astype(str).tolist()


def _select_tau_nested(
    cdf: pd.DataFrame,
    folds: pd.DataFrame,
    tools: list[str],
    families: list[str],
    *,
    threshold: int,
    seed: int,
    calibration_metrics: list[str] | None,
    taus: tuple[float, ...],
    tier: str,
    metric: str = "f1",
    tune_splits: int = 4,
) -> tuple[list[str], pd.DataFrame]:
    """Pick tau for each strategy from calibration-internal data only.

    For each of the 5 outer folds, ``cal_df`` (already excludes the outer held-out
    ``val_df``) is further split into an inner ``cal_fit``/``cal_tune`` pair
    (stratified, tune_splits=4 -> a 75/25 split). Reliability/ML models are fit on
    ``cal_fit`` and tau is chosen, per strategy, by maximising ``metric`` on
    ``cal_tune`` -- never on ``val_df``, which this function never touches.

    Returns:
      - ``selected_strategies``: one tau-suffixed strategy name per base strategy,
        picked from the 5-fold-aggregated inner-tune metrics (mirrors what
        ``best_tau_per_strategy`` does today, just fed calibration-internal data
        instead of the held-out ``per_family`` table).
      - ``fold_tau_choices``: the tau chosen independently in each of the 5 outer
        folds (before aggregating across folds) -- the fold-to-fold variability of
        tau that reviewers asked to see reported.
    """
    tune_frames: list[pd.DataFrame] = []
    fold_choice_rows: list[dict] = []

    for fold in sorted(folds["fold"].unique()):
        cal_df, _val_df = split_train_validation(cdf, folds, validation_fold=fold)

        inner_folds = multilabel_stratified_kfold(
            cal_df, n_splits=tune_splits, random_state=seed * 97 + int(fold), min_cwe_count=1,
        )
        cal_fit, cal_tune = split_train_validation(cal_df, inner_folds, validation_fold=0)

        reliability_fit = compute_reliability(cal_fit, tools, families)
        predictions_tune = run_fusion(
            cal_fit, cal_tune, reliability_fit, tools, families,
            threshold=threshold, seed=seed, calibration_metrics=calibration_metrics,
            taus=taus, tier=tier,
        )
        per_family_tune = evaluate_predictions(predictions_tune, group_cols=("strategy", "family"))
        per_family_tune.insert(0, "fold", fold)
        tune_frames.append(per_family_tune)

        present = set(per_family_tune["strategy"].astype(str))
        all_fusion = canonical_strategy_order(tools, fusion_strategies(present))
        fold_selected = best_tau_per_strategy(per_family_tune, all_fusion, metric)
        for strategy in fold_selected:
            base, tau = split_tau_strategy(strategy)
            fold_choice_rows.append({"fold": fold, "base_strategy": base, "tau": tau, "strategy": strategy})

    per_family_tune_all = pd.concat(tune_frames, ignore_index=True)
    per_family_tune_mean = mean_over_folds(per_family_tune_all)
    present = set(per_family_tune_mean["strategy"].astype(str))
    all_fusion = canonical_strategy_order(tools, fusion_strategies(present))
    selected_strategies = best_tau_per_strategy(per_family_tune_mean, all_fusion, metric)

    fold_tau_choices = pd.DataFrame(fold_choice_rows)
    return selected_strategies, fold_tau_choices


def run_language_level(
    language: str,
    level: str,
    *,
    n_splits: int = 5,
    min_cwe_count: int | None = None,
    threshold: int = 2,
    calibration_metrics: list[str] | None = None,
    tau_min: float = 0.1,
    tau_max: float = 0.9,
    exclude: list[str] | None = None,
    seed: int = 42,
    enriched_dir: str | Path = "data/enriched",
    results_root: str | Path = "data/results",
    tier: str = "base",
    write_plots: bool = True,
) -> dict:
    """Run the full experiment for one (language, canonical level).

    ``tier`` controls both the minimum family support and the ML strategy set:
      base   — min_cwe_count default (= n_splits); existing strategies only
      medium — min_cwe_count = 30; adds Decision Tree
      full   — min_cwe_count = 100; adds Decision Tree + Random Forest + Gradient Boosting + XGBoost

    An explicit ``min_cwe_count`` overrides the tier's default floor.
    """
    effective_min = TIER_MIN_COUNT.get(tier, n_splits) if min_cwe_count is None else min_cwe_count
    try:
        prep = prepare_canonical(language, level, n_splits=n_splits, min_cwe_count=effective_min,
                                 exclude=exclude, seed=seed, enriched_dir=enriched_dir)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return {}
    df, cdf, tools, families = prep.df, prep.cdf, prep.tools, prep.families
    canonical_map, restriction, folds = prep.canonical_map, prep.restriction, prep.folds

    results_dir = Path(results_root) / language / level / tier
    results_dir.mkdir(parents=True, exist_ok=True)
    _print_header(language, level, df, tools, restriction)

    if not families:
        console.print(f"[red]{language}/{level}: no families survive restriction. Skipping.[/red]")
        return {"language": language, "level": level, **restriction}

    metric_pairs = metric_pairs_for_calibration(calibration_metrics)
    taus = taus_in_range(tau_min, tau_max)

    # ── cross-validation ──────────────────────────────────────────────────────
    reliability_frames, per_family_frames, detection_frames, histogram_frames = [], [], [], []
    for fold in sorted(folds["fold"].unique()):
        console.print(f"  fold {fold}: calibrating + fusing …")
        cal_df, val_df = split_train_validation(cdf, folds, validation_fold=fold)

        reliability = compute_reliability(cal_df, tools, families)
        reliability.insert(0, "fold", fold)
        reliability_frames.append(reliability)

        predictions = run_fusion(cal_df, val_df, reliability.drop(columns="fold"),
                                 tools, families, threshold=threshold, seed=seed,
                                 calibration_metrics=calibration_metrics, taus=taus,
                                 tier=tier)
        per_family = evaluate_predictions(predictions, group_cols=("strategy", "family"))
        per_family.insert(0, "fold", fold)
        per_family_frames.append(per_family)

        detection = detection_metrics(detection_from_predictions(predictions, val_df))
        detection.insert(0, "fold", fold)
        detection_frames.append(detection)

        # Sufficient statistic for every threshold question, so that later analyses
        # (tau sweeps, PR curves, dominance checks) do not have to re-run fusion just to
        # recover scores this run already computed. See predictions.score_histogram.
        histogram = score_histogram(predictions)
        histogram.insert(0, "fold", fold)
        histogram_frames.append(histogram)

    per_family_per_fold = pd.concat(per_family_frames, ignore_index=True)
    per_family_mean, overall = aggregate_fusion_metrics(per_family_per_fold)
    detection_overall = (
        pd.concat(detection_frames, ignore_index=True)
        .drop(columns="fold")
        .groupby("strategy", as_index=False)
        .mean(numeric_only=True)
    )

    # The tau sweep is already materialised as explicit strategy names. These two
    # compatibility reports parse those names back into (base_strategy, tau).
    tau_overall = tau_variant_table(detection_overall)
    operating_points = _build_operating_points(tau_overall, detection_overall)

    # Fixed strategy order everywhere: single tools → OR → traditional → fusers.
    per_family_mean = order_by_strategy(per_family_mean, tools, extra_sort=["family"])
    overall = order_by_strategy(overall, tools)
    detection_overall = order_by_strategy(detection_overall, tools)
    tau_overall = order_by_strategy(tau_overall, tools, extra_sort=["tau"])
    operating_points = order_by_strategy(operating_points, tools)

    # ── write outputs ─────────────────────────────────────────────────────────
    save_config({"language": language, "level": level, "tools": tools, "n_splits": n_splits,
                 "threshold": threshold, "seed": seed, "excluded": exclude or [],
                 "calibration_metrics": calibration_metrics,
                 "calibration_metric_pairs": [list(pair) for pair in metric_pairs],
                 "tau_min": tau_min, "tau_max": tau_max, "taus": list(taus),
                 "restriction": restriction}, results_dir / "config.json")
    save_csv(canonical_map, results_dir / "canonical_map.csv")
    save_csv(folds, results_dir / "folds.csv")
    save_csv(pd.concat(reliability_frames, ignore_index=True), results_dir / "calibration_reliability.csv")
    save_csv(pd.concat(histogram_frames, ignore_index=True), results_dir / "fusion_score_histogram.csv")
    save_csv(per_family_per_fold, results_dir / "fusion_metrics_per_family_per_fold.csv")
    save_csv(per_family_mean, results_dir / "fusion_metrics_per_family.csv")
    save_csv(overall, results_dir / "fusion_metrics_overall.csv")
    save_csv(detection_overall, results_dir / "fusion_detection_overall.csv")
    save_csv(tau_overall, results_dir / "fusion_tau_sweep.csv")
    save_csv(operating_points, results_dir / "fusion_operating_points.csv")
    if write_plots:
        save_strategy_plots(overall, results_dir / "plots")
        save_best_variant_plots(overall, results_dir / "plots" / "best_variants")
        save_family_performance_plots(per_family_mean, overall, results_dir / "plots" / "top_cwe_families", tools=tools)

        # Tau for the headline (strategy, family) comparison is picked from
        # calibration-internal data only (nested per outer fold), never from the
        # val_df values being reported here -- see _select_tau_nested's docstring.
        console.print("  selecting tau via nested calibration-internal split …")
        selected_strategies, fold_tau_choices = _select_tau_nested(
            cdf, folds, tools, families, threshold=threshold, seed=seed,
            calibration_metrics=calibration_metrics, taus=taus, tier=tier, metric="f1",
        )
        save_csv(fold_tau_choices, results_dir / "fusion_tau_selection_by_fold.csv")
        # Persist the aggregated selection too: ``regenerate_plots`` must reuse exactly
        # these strategies instead of re-picking tau from the held-out values it is
        # about to report on (which would silently undo the nested selection).
        save_csv(_selected_tau_table(selected_strategies), results_dir / TAU_SELECTION_FILE)
        save_mean_difference_ci_report(
            per_family_mean, results_dir / "plots", metric="f1", tools=tools,
            preselected_strategies=selected_strategies,
        )

    _print_overall(language, level, overall, detection_overall)
    if not operating_points.empty:
        _print_operating_points(language, level, operating_points)
    return {"language": language, "level": level, "results_dir": str(results_dir), **restriction}


def run(
    language: str = "all",
    level: str = "pillar_child",
    *,
    languages: list[str] | None = None,
    **kwargs,
) -> list[dict]:
    """Run one or all languages at the single supported canonical level."""
    from analysis.canonical import CANONICAL_LEVEL

    if level in {"all", None}:
        level = CANONICAL_LEVEL
    langs = languages if languages is not None else ([language] if language != "all" else ["c_cpp", "java", "python"])
    summaries = []
    for lang in langs:
        try:
            summaries.append(run_language_level(lang, level, **kwargs))
        except FileNotFoundError as exc:
            console.print(f"[yellow]{lang}/{level}: {exc}[/yellow]")
    return summaries


def regenerate_plots(
    language: str,
    level: str,
    *,
    tier: str = "base",
    results_root: str | Path = "data/results",
) -> bool:
    """Regenerate all plots and tables for one (language, level, tier) from existing CSVs.

    Reads the CSV files written by ``run_language_level`` and re-runs the plotting
    and CI steps without touching the enriched data or re-running fusion.  Returns
    True if the results directory was found and plots were written, False otherwise.
    """
    results_dir = Path(results_root) / language / level / tier
    config_path = results_dir / "config.json"
    overall_path = results_dir / "fusion_metrics_overall.csv"
    per_family_path = results_dir / "fusion_metrics_per_family.csv"
    detection_path = results_dir / "fusion_detection_overall.csv"
    tau_path = results_dir / "fusion_tau_sweep.csv"

    missing = [p for p in (config_path, overall_path, per_family_path, detection_path) if not p.exists()]
    if missing:
        console.print(f"[red]{language}/{level}: missing files — run 'fusion' first:[/red]")
        for p in missing:
            console.print(f"  [red]{p}[/red]")
        return False

    import json
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tools: list[str] = config.get("tools", [])

    overall = pd.read_csv(overall_path)
    per_family_mean = pd.read_csv(per_family_path)
    detection_overall = pd.read_csv(detection_path)

    console.print(f"[bold]Regenerating plots[/bold]  {language}/{level}  ({len(overall)} strategies)")

    save_strategy_plots(overall, results_dir / "plots")
    save_best_variant_plots(overall, results_dir / "plots" / "best_variants")
    save_family_performance_plots(per_family_mean, overall, results_dir / "plots" / "top_cwe_families", tools=tools)
    # Tau must come from the nested calibration-internal selection persisted by
    # ``run_language_level``. Falling back to ``best_tau_per_strategy`` here would
    # re-pick tau on the very held-out values this report is about to present, so the
    # fallback is loud rather than silent.
    preselected = _load_selected_tau(results_dir)
    if preselected is None:
        console.print(
            f"[yellow]{language}/{level}: {TAU_SELECTION_FILE} missing — tau will be "
            f"re-selected from the held-out per-family values, which is NOT the nested "
            f"selection used by 'fusion'. Re-run 'fusion' to get a comparable report."
            "[/yellow]"
        )
    save_mean_difference_ci_report(
        per_family_mean, results_dir / "plots", metric="f1", tools=tools,
        preselected_strategies=preselected,
    )

    tau_overall = tau_variant_table(detection_overall) if tau_path.exists() else pd.DataFrame()
    operating_points = _build_operating_points(tau_overall, detection_overall)
    operating_points = order_by_strategy(operating_points, tools)
    if not operating_points.empty:
        _print_operating_points(language, level, operating_points)

    console.print(f"[green]Done →[/green] {results_dir / 'plots'}")
    return True


def regenerate_plots_all(
    language: str = "all",
    level: str = "pillar_child",
    *,
    languages: list[str] | None = None,
    tier: str = "base",
    results_root: str | Path = "data/results",
) -> list[bool]:
    """Regenerate plots for one or all languages."""
    from analysis.canonical import CANONICAL_LEVEL

    if level in {"all", None}:
        level = CANONICAL_LEVEL
    langs = languages if languages is not None else (
        [language] if language != "all" else ["c_cpp", "java", "python"]
    )
    return [regenerate_plots(lang, level, tier=tier, results_root=results_root) for lang in langs]


# ── console output ──────────────────────────────────────────────────────────

def _print_header(language: str, level: str, df: pd.DataFrame, tools: list[str], restriction: dict) -> None:
    from rich.panel import Panel
    console.print(Panel(
        f"[bold]language[/bold]   {language}\n"
        f"[bold]level[/bold]      {level}\n"
        f"[bold]rows[/bold]       {len(df):,}\n"
        f"[bold]tools[/bold]      {', '.join(tools)}\n"
        f"[bold]families[/bold]   {restriction['families_kept']} kept / {restriction['families_present']} present  "
        f"([red]{restriction['dropped_unsupported']}[/red] unsupported, "
        f"[yellow]{restriction['dropped_below_min_count']}[/yellow] < {restriction['min_cwe_count']} occ)\n"
        f"[bold]coverage[/bold]   {restriction['gt_occurrences_kept']:,}/{restriction['gt_occurrences_total']:,} "
        f"GT occurrences kept ({restriction['gt_occurrences_kept_frac']:.1%})",
        title=f"fusion · {language} · {level}", expand=False,
    ))


def _print_overall(language: str, level: str, overall: pd.DataFrame, detection: pd.DataFrame) -> None:
    table = Table(title=f"{language} · {level} · overall (support-weighted across families)")
    table.add_column("strategy", style="cyan", no_wrap=True)
    for metric in ("precision", "recall", "f1", "f2", "mcc"):
        table.add_column(metric, justify="right")
    for _i, r in overall.iterrows():   # already in canonical strategy order
        table.add_row(str(r["strategy"]), *[
            "—" if pd.isna(r.get(f"{m}_weighted")) else f"{r[f'{m}_weighted']:.3f}"
            for m in ("precision", "recall", "f1", "f2", "mcc")
        ])
    console.print(table)

    det = Table(title=f"{language} · {level} · vuln/safe detection (mean over folds)")
    det.add_column("strategy", style="cyan", no_wrap=True)
    for metric in ("precision", "recall", "f1", "f2", "mcc", "cwe_attribution_acc"):
        det.add_column(metric, justify="right")
    for _i, r in detection.iterrows():   # already in canonical strategy order
        det.add_row(str(r["strategy"]), *[
            "—" if pd.isna(r.get(m)) else f"{r[m]:.3f}"
            for m in ("precision", "recall", "f1", "f2", "mcc", "cwe_attribution_acc")
        ])
    console.print(det)


def _print_operating_points(language: str, level: str, operating_points: pd.DataFrame) -> None:
    """Best-F1 operating point per fuser strategy + baseline rows (traditional, OR).

    Fuser strategies show their best-F1 explicit τ; baselines have no τ sweep so τ
    is shown as '—'.
    """
    table = Table(title=f"{language} · {level} · detection @ best-F1 explicit τ")
    table.add_column("strategy", style="cyan", no_wrap=True)
    for col in ("tau", "precision", "recall", "f1", "f2", "accuracy", "mcc", "roc_auc"):
        table.add_column(col, justify="right")
    for _i, r in operating_points.iterrows():
        tau_val = r.get("tau")
        tau_str = "—" if tau_val is None or pd.isna(tau_val) else f"{float(tau_val):.1f}"
        table.add_row(
            str(r["base_strategy"] if "base_strategy" in r else r["strategy"]),
            tau_str,
            *["—" if pd.isna(r.get(m)) else f"{r[m]:.3f}" for m in ("precision", "recall", "f1", "f2", "accuracy", "mcc", "roc_auc")],
        )
    console.print(table)
