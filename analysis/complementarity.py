"""Tool-complementarity diagnostics (downstream of canonicalisation).

Answers the question that decides whether multi-tool fusion can ever beat the best single
tool on these data: **is there exploitable complementarity, or are the tools redundant?**

Lenses (all on the canonical family space from ``analysis.experiment.prepare_canonical``):

* coverage / oracle  — per-tool recall, the OR (union) recall, unique catches, and the
  oracle recall ceiling. ``headroom = oracle_recall - best_tool_recall`` is the decider.
* diversity          — pairwise double-fault and Q-statistic (Kuncheva): do tools fail
  together (redundant) or on different samples (complementary)?
* reliability heatmap — per-(tool, family) recall, to localise where each tool is strong.
* marginal contribution (CV) — greedy ROC-AUC curve as tools are added: does adding tools
  keep helping, and which ones?
* conditional value (CV)     — for each tool, the AUC gain and likelihood-ratio test of
  adding it *beyond the best tool* (does it carry information bandit doesn't?).

Honest metrics throughout: recall / MCC / ROC-AUC, never F2 on the positive-majority
detection set (a trivial "always vulnerable" predictor scores high F2 there).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.calibration import compute_reliability
from analysis.dataset import as_list

# ── matrix builders ───────────────────────────────────────────────────────────

def family_matrices(cdf: pd.DataFrame, tools: list[str], families: list[str]):
    """Boolean incidence over the family grid.

    Returns ``(fires, gt)`` where ``fires[tool]`` and ``gt`` are ``[n_rows, n_families]``
    boolean arrays (family fired by the tool / present in ground truth).
    """
    index = {f: j for j, f in enumerate(families)}
    n, fdim = len(cdf), len(families)

    def incidence(series):
        m = np.zeros((n, fdim), dtype=bool)
        for i, values in enumerate(series):
            for f in as_list(values):
                j = index.get(f)
                if j is not None:
                    m[i, j] = True
        return m

    gt = incidence(cdf["cwes"])
    fires = {tool: incidence(cdf[tool]) for tool in tools}
    return fires, gt


def detection_arrays(cdf: pd.DataFrame, fires: dict[str, np.ndarray]):
    """Per-row detection view: tool fires any kept family; label = row is vulnerable."""
    label = (cdf["label"].astype(int) == 1).to_numpy()
    fire_det = {tool: m.any(axis=1) for tool, m in fires.items()}
    return fire_det, label


# ── coverage / oracle ─────────────────────────────────────────────────────────

def coverage_report(catch: dict[str, np.ndarray], positive: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Recall-side complementarity over positive instances.

    ``catch[tool]`` is a boolean array (tool caught the instance); ``positive`` marks the
    positive instances (a flat boolean array aligned with the catch arrays). Returns a
    per-tool table and a summary with the OR/oracle recall and the headroom.
    """
    tools = list(catch)
    pos = positive
    n_pos = int(pos.sum())
    stacked = np.vstack([catch[t] for t in tools])           # [T, M]
    any_catch = stacked.any(axis=0)

    rows = []
    for ti, tool in enumerate(tools):
        caught = catch[tool] & pos
        others = np.delete(stacked, ti, axis=0).any(axis=0) if len(tools) > 1 else np.zeros_like(pos)
        unique = caught & ~others
        rows.append({
            "tool": tool,
            "recall": caught.sum() / n_pos if n_pos else float("nan"),
            "unique_catches": int(unique.sum()),
        })
    per_tool = pd.DataFrame(rows).sort_values("recall", ascending=False, ignore_index=True)

    or_recall = (any_catch & pos).sum() / n_pos if n_pos else float("nan")
    best = float(per_tool["recall"].max()) if len(per_tool) else float("nan")
    summary = {
        "n_positives": n_pos,
        "best_tool": per_tool.iloc[0]["tool"] if len(per_tool) else None,
        "best_tool_recall": best,
        "or_recall": float(or_recall),            # oracle recall: catchable iff any tool fires
        "headroom": float(or_recall) - best,      # the decision number
        "missed_by_all": int((pos & ~any_catch).sum()),
    }
    return per_tool, summary


def catch_combinations(catch: dict[str, np.ndarray], positive: np.ndarray) -> pd.DataFrame:
    """UpSet-style counts: how many positives are caught by exactly which tool set."""
    tools = list(catch)
    counts: dict[frozenset, int] = {}
    idx = np.where(positive)[0]
    for i in idx:
        combo = frozenset(t for t in tools if catch[t][i])
        counts[combo] = counts.get(combo, 0) + 1
    rows = [{"tools": "∅ (missed)" if not k else "+".join(sorted(k)), "n_tools": len(k), "count": v}
            for k, v in counts.items()]
    return pd.DataFrame(rows).sort_values("count", ascending=False, ignore_index=True)


# ── diversity (Kuncheva) ──────────────────────────────────────────────────────

def diversity_report(correct: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise error-diversity: double-fault and Q-statistic over correctness vectors.

    ``correct[tool]`` is a boolean array (tool's decision == label) over instances.
    Q→1 means the pair errs alike (redundant), Q→0 independent, Q<0 complementary;
    high double-fault means they are wrong on the same instances (voting can't recover).
    """
    rows = []
    for a, b in itertools.combinations(correct, 2):
        ca, cb = correct[a], correct[b]
        n11 = int((ca & cb).sum())
        n00 = int((~ca & ~cb).sum())
        n10 = int((ca & ~cb).sum())
        n01 = int((~ca & cb).sum())
        denom = n11 * n00 + n01 * n10
        q = (n11 * n00 - n01 * n10) / denom if denom else float("nan")
        rows.append({
            "tool_a": a, "tool_b": b,
            "double_fault": n00 / len(ca),
            "disagreement": (n10 + n01) / len(ca),
            "q_statistic": q,
        })
    return pd.DataFrame(rows)


# ── reliability heatmap ───────────────────────────────────────────────────────

def reliability_heatmap(cdf: pd.DataFrame, tools: list[str], families: list[str],
                        metric: str = "sensitivity") -> pd.DataFrame:
    """Per-(tool, family) ``metric`` (default recall) as a tool×family pivot."""
    rel = compute_reliability(cdf, tools, families)
    return rel.pivot(index="tool", columns="family", values=metric)


def save_heatmap_png(pivot: pd.DataFrame, path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.3 * pivot.shape[1]), 0.6 * pivot.shape[0] + 1.5))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── model-based lenses (cross-validated) ──────────────────────────────────────

def _fold_array(folds: pd.DataFrame, n: int) -> np.ndarray:
    return folds.set_index("row_index")["fold"].reindex(range(n)).to_numpy()


def _oof_scores(X: np.ndarray, y: np.ndarray, fold_arr: np.ndarray) -> np.ndarray:
    """Out-of-fold positive-class probabilities from a per-fold logistic regression."""
    from sklearn.linear_model import LogisticRegression
    oof = np.zeros(len(y), dtype=float)
    for f in np.unique(fold_arr):
        tr, va = fold_arr != f, fold_arr == f
        if X.shape[1] == 0 or len(np.unique(y[tr])) < 2:
            oof[va] = float(y[tr].mean()) if tr.any() else 0.0
        else:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            oof[va] = clf.predict_proba(X[va])[:, 1]
    return oof


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def marginal_contribution_cv(X: np.ndarray, y: np.ndarray, fold_arr: np.ndarray, tools: list[str]) -> pd.DataFrame:
    """Greedy ablation: add the tool that most improves out-of-fold detection AUC.

    A flat curve after the first tool means the rest are redundant; a rising curve means
    real complementarity (and names which tools carry it).
    """
    index = {t: i for i, t in enumerate(tools)}
    chosen: list[str] = []
    remaining = set(tools)
    rows = []
    while remaining:
        best_tool, best_auc = None, -1.0
        for t in remaining:
            cols = [index[x] for x in (*chosen, t)]
            auc = _auc(y, _oof_scores(X[:, cols], y, fold_arr))
            if auc > best_auc:
                best_auc, best_tool = auc, t
        chosen.append(best_tool)
        remaining.discard(best_tool)
        rows.append({"n_tools": len(chosen), "added": best_tool, "oof_auc": best_auc})
    return pd.DataFrame(rows)


def _loglik(X: np.ndarray, y: np.ndarray) -> float:
    """In-sample Bernoulli log-likelihood of an unregularised logistic fit (for LR-test)."""
    from sklearn.linear_model import LogisticRegression
    # C=inf == unregularised MLE (penalty=None is deprecated in sklearn >=1.8), needed for
    # a valid likelihood-ratio test.
    clf = LogisticRegression(C=np.inf, max_iter=2000)
    clf.fit(X, y)
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def conditional_value_cv(X: np.ndarray, y: np.ndarray, fold_arr: np.ndarray, tools: list[str]) -> tuple[pd.DataFrame, str]:
    """For each tool, the value it adds *beyond the best single tool*.

    ``auc_gain`` is the out-of-fold AUC improvement of (best + tool) over best alone;
    ``lr_p`` is the likelihood-ratio test p-value (χ²₁) for adding the tool — small p means
    the tool carries detection information the best tool does not.
    """
    from scipy.stats import chi2
    index = {t: i for i, t in enumerate(tools)}
    single = {t: _auc(y, _oof_scores(X[:, [index[t]]], y, fold_arr)) for t in tools}
    best = max(single, key=lambda t: (single[t] if single[t] == single[t] else -1))
    auc_best = single[best]
    ll_best = _loglik(X[:, [index[best]]], y)

    rows = []
    for t in tools:
        if t == best:
            continue
        cols = [index[best], index[t]]
        auc_pair = _auc(y, _oof_scores(X[:, cols], y, fold_arr))
        ll_pair = _loglik(X[:, cols], y)
        stat = 2.0 * (ll_pair - ll_best)
        rows.append({
            "tool": t,
            "auc_gain_over_best": auc_pair - auc_best,
            "lr_stat": stat,
            "lr_p": float(chi2.sf(stat, 1)) if stat > 0 else 1.0,
        })
    table = pd.DataFrame(rows).sort_values("auc_gain_over_best", ascending=False, ignore_index=True)
    return table, best


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_diagnostics(
    language: str,
    level: str,
    *,
    n_splits: int = 5,
    min_cwe_count: int | None = None,
    exclude: list[str] | None = None,
    seed: int = 42,
    enriched_dir: str | Path = "data/enriched",
    results_root: str | Path = "data/results",
) -> dict:
    """Run every complementarity lens for one language at the canonical level."""
    import json

    from rich.console import Console
    from rich.table import Table

    from analysis.experiment import prepare_canonical

    console = Console()
    prep = prepare_canonical(language, level, n_splits=n_splits, min_cwe_count=min_cwe_count,
                             exclude=exclude, seed=seed, enriched_dir=enriched_dir)
    cdf, tools, families = prep.cdf, prep.tools, prep.families
    if not families:
        console.print(f"[red]{language}/{level}: no families survive restriction.[/red]")
        return {}

    out_dir = Path(results_root) / language / level / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    fires, gt = family_matrices(cdf, tools, families)
    fire_det, label_det = detection_arrays(cdf, fires)

    # coverage / oracle — family and detection
    cov_fam, sum_fam = coverage_report({t: fires[t].flatten() for t in tools}, gt.flatten())
    cov_det, sum_det = coverage_report(fire_det, label_det)
    combos_fam = catch_combinations({t: fires[t].flatten() for t in tools}, gt.flatten())

    # diversity — family-cell correctness
    div_fam = diversity_report({t: (fires[t] == gt).flatten() for t in tools})

    # reliability heatmap (recall per tool×family)
    heatmap = reliability_heatmap(cdf, tools, families, metric="sensitivity")
    save_heatmap_png(heatmap, out_dir / "reliability_heatmap.png",
                     f"{language}/{level} — per-(tool,family) recall")

    # model-based — detection level
    X = np.column_stack([fire_det[t].astype(float) for t in tools])
    y = label_det.astype(int)
    fold_arr = _fold_array(prep.folds, len(cdf))
    marginal = marginal_contribution_cv(X, y, fold_arr, tools)
    conditional, best_tool = conditional_value_cv(X, y, fold_arr, tools)

    # verdict
    auc_one = float(marginal.iloc[0]["oof_auc"])
    auc_all = float(marginal.iloc[-1]["oof_auc"])
    verdict = {
        "language": language, "level": level, "tools": tools, "n_families": len(families),
        "family_best_tool": sum_fam["best_tool"], "family_best_tool_recall": sum_fam["best_tool_recall"],
        "family_or_recall": sum_fam["or_recall"], "family_recall_headroom": sum_fam["headroom"],
        "detection_best_tool_recall": sum_det["best_tool_recall"], "detection_or_recall": sum_det["or_recall"],
        "detection_recall_headroom": sum_det["headroom"],
        "auc_best_single": auc_one, "auc_all_tools": auc_all, "auc_gain_full_ensemble": auc_all - auc_one,
        "best_tool_by_auc": best_tool,
        "any_tool_adds_signal": bool((conditional["lr_p"] < 0.05).any()) if len(conditional) else False,
    }

    # write
    cov_fam.to_csv(out_dir / "coverage_family.csv", index=False)
    cov_det.to_csv(out_dir / "coverage_detection.csv", index=False)
    combos_fam.to_csv(out_dir / "catch_combinations_family.csv", index=False)
    div_fam.to_csv(out_dir / "diversity_family.csv", index=False)
    heatmap.to_csv(out_dir / "reliability_heatmap.csv")
    marginal.to_csv(out_dir / "marginal_contribution.csv", index=False)
    conditional.to_csv(out_dir / "conditional_value.csv", index=False)
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")

    _print_verdict(console, Table, language, level, sum_fam, sum_det, marginal, conditional, verdict)
    return verdict


def _print_verdict(console, Table, language, level, sum_fam, sum_det, marginal, conditional, verdict) -> None:
    t = Table(title=f"complementarity · {language}/{level}")
    t.add_column("signal", style="cyan"); t.add_column("value", justify="right")
    t.add_row("family best-tool recall", f"{sum_fam['best_tool_recall']:.3f} ({sum_fam['best_tool']})")
    t.add_row("family OR / oracle recall", f"{sum_fam['or_recall']:.3f}")
    t.add_row("family recall headroom", f"{sum_fam['headroom']:.3f}")
    t.add_row("detection recall headroom", f"{sum_det['headroom']:.3f}")
    t.add_row("AUC best single tool", f"{verdict['auc_best_single']:.3f} ({verdict['best_tool_by_auc']})")
    t.add_row("AUC full ensemble", f"{verdict['auc_all_tools']:.3f}")
    t.add_row("AUC gain (ensemble − best)", f"{verdict['auc_gain_full_ensemble']:+.3f}")
    t.add_row("any tool adds signal (p<.05)", "yes" if verdict["any_tool_adds_signal"] else "no")
    console.print(t)
    recoverable = verdict["auc_gain_full_ensemble"] >= 0.02 or verdict["family_recall_headroom"] >= 0.05
    msg = ("[green]headroom exists — fusion can in principle beat the best single tool[/green]"
           if recoverable else
           "[yellow]tools look redundant — fusion ≈ best single tool on these data[/yellow]")
    console.print(msg)
