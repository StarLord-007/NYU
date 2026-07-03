"""
fable_compare.py
================
Head-to-head benchmark of the five champion classifiers:

    XGBoost (reference) | KNN | Decision Tree | MLP | SVM

Guarantees of fairness
----------------------
* One canonical dataset load (fable_common via fable_models.load_dataset).
* Outer StratifiedGroupKFold splits are computed ONCE and the same index
  arrays are fed to every model (interpolation protocol).
* LOPO splits are by construction identical (deterministic paper iteration).
* Thresholds are tuned per model on inner held-out-paper folds only, using
  one shared strategy (fable_train.optimal_thresholds, weighted variant for
  estimators without fit-time weights — documented in fable_models).
* Identical metrics, identical weighting policy, identical feature set.

Run:
    python fable_compare.py --data Microgravity_Database.xlsm
    python fable_compare.py --quick          # smoke test (1 seed, no LOPO)

Outputs -> ``<this script dir>/compare_outputs/``:
    interpolation_folds.csv / interpolation_summary.csv
    pairwise_tests.csv
    lopo_per_paper.csv / lopo_pooled.json
    comparison_summary.md / comparison_summary.json
    fig_compare_interpolation.png / fig_compare_lopo_pooled.png /
    fig_compare_lopo_per_paper.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Fable" / "dt"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from fable_models import (
    RNG,
    benchmark_interpolation,
    ModelSpec,
    load_dataset,
    lopo_evaluate,
    inner_thresholds,
    make_outer_splits,
    plot_fold_distribution,
    spec_weight_fn,
    summarise_folds,
    to_markdown_table,
    xgb_champion_factory,
)
from fable_knn import SPEC as KNN_SPEC
from fable_mlp import SPEC as MLP_SPEC
from fable_svm import SPEC as SVM_SPEC


def make_dt(params: dict, seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=seed, **params)


DT_SPEC = ModelSpec(
    key="dt",
    display="Decision Tree",
    estimator_factory=make_dt,
    param_space={
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "criterion": ["gini", "entropy"],
    },
    champion_params={"max_depth": 7, "min_samples_leaf": 2, "criterion": "entropy"},
    supports_sample_weight=True,
)


SK_SPECS = [KNN_SPEC, DT_SPEC, MLP_SPEC, SVM_SPEC]


def model_registry(random_state: int = RNG) -> dict[str, tuple]:
    """name -> (zero-arg model factory, threshold-stage weight_fn or None).

    To add another algorithm to the benchmark, write a ModelSpec module
    (see fable_knn.py) and append its SPEC to SK_SPECS above — nothing else
    changes.
    """
    reg = {"XGBoost": (xgb_champion_factory(random_state), None)}
    for spec in SK_SPECS:
        reg[spec.display] = (spec.champion_factory(random_state), spec_weight_fn(spec))
    return reg


# ---------------------------------------------------------------------------
# Pairwise statistics (paired over identical folds -> partition variance
# cancels; Wilcoxon signed-rank, as in fable_eval.paired_analysis)
# ---------------------------------------------------------------------------

def pairwise_tests(
    folds: pd.DataFrame,
    metrics: tuple[str, ...] = ("roc_auc", "pr_auc", "mcc", "balanced_accuracy"),
) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    models = list(folds["method"].unique())
    piv = {m: folds[folds["method"] == m].set_index(["seed", "fold"]) for m in models}
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            common = piv[a].index.intersection(piv[b].index)
            rec = {"model_a": a, "model_b": b, "n_folds": len(common)}
            for metric in metrics:
                d = (piv[a].loc[common, metric] - piv[b].loc[common, metric]).to_numpy()
                rec[f"delta_{metric}"] = float(np.mean(d))
                try:
                    rec[f"wilcoxon_p_{metric}"] = (
                        float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                    )
                except ValueError:
                    rec[f"wilcoxon_p_{metric}"] = np.nan
            rows.append(rec)
    if not rows:
        return pd.DataFrame(
            columns=[
                "model_a",
                "model_b",
                "n_folds",
                "delta_roc_auc",
                "wilcoxon_p_roc_auc",
                "delta_pr_auc",
                "wilcoxon_p_pr_auc",
                "delta_mcc",
                "wilcoxon_p_mcc",
                "delta_balanced_accuracy",
                "wilcoxon_p_balanced_accuracy",
            ]
        )
    return pd.DataFrame(rows).sort_values("delta_roc_auc", ascending=False)


def benchmark_random_holdout(
    name: str,
    model_factory,
    df: pd.DataFrame,
    n_seeds: int = 5,
) -> pd.DataFrame:
    """Repeated stratified 80/20 hold-out on rows, nested threshold tuning only on training data."""
    X = df
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"]
    rows = []
    for seed in range(n_seeds):
        tr, te = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=RNG + seed
        )
        Xtr, ytr, ptr = X.iloc[tr], y[tr], papers.iloc[tr]
        Xte, yte = X.iloc[te], y[te]
        thr = inner_thresholds(model_factory, Xtr, ytr, ptr, seed=RNG + seed)
        model = model_factory()
        model.fit(Xtr, ytr, ptr)
        p = model.predict_proba(Xte)
        rows.append(
            {
                "method": name,
                "seed": seed,
                "fold": 0,
                "n_test": len(te),
                "n_test_papers": int(papers.iloc[te].nunique()),
                "roc_auc": roc_auc_score(yte, p),
                "pr_auc": average_precision_score(yte, p),
                "mcc": matthews_corrcoef(yte, (p >= thr["mcc"]).astype(int)),
                "f1": f1_score(yte, (p >= thr["f1"]).astype(int)),
                "balanced_accuracy": balanced_accuracy_score(
                    yte, (p >= thr["balanced_accuracy"]).astype(int)
                ),
                "youden_j_balacc": balanced_accuracy_score(
                    yte, (p >= thr["youden_j"]).astype(int)
                ),
                "thr_mcc": thr["mcc"],
                "thr_f1": thr["f1"],
                "thr_balacc": thr["balanced_accuracy"],
                "thr_youden": thr["youden_j"],
            }
        )
    return pd.DataFrame(rows)


def benchmark_stratified_kfold(name: str, model_factory, df: pd.DataFrame) -> pd.DataFrame:
    """Stratified 5-fold CV on rows, nested threshold tuning only on training data."""
    X = df
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"]
    rows = []
    skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, ytr, ptr = X.iloc[tr], y[tr], papers.iloc[tr]
        Xte, yte = X.iloc[te], y[te]
        thr = inner_thresholds(model_factory, Xtr, ytr, ptr, seed=RNG + fold)
        model = model_factory()
        model.fit(Xtr, ytr, ptr)
        p = model.predict_proba(Xte)
        rows.append(
            {
                "method": name,
                "seed": 0,
                "fold": fold,
                "n_test": len(te),
                "n_test_papers": int(papers.iloc[te].nunique()),
                "roc_auc": roc_auc_score(yte, p),
                "pr_auc": average_precision_score(yte, p),
                "mcc": matthews_corrcoef(yte, (p >= thr["mcc"]).astype(int)),
                "f1": f1_score(yte, (p >= thr["f1"]).astype(int)),
                "balanced_accuracy": balanced_accuracy_score(
                    yte, (p >= thr["balanced_accuracy"]).astype(int)
                ),
                "youden_j_balacc": balanced_accuracy_score(
                    yte, (p >= thr["youden_j"]).astype(int)
                ),
                "thr_mcc": thr["mcc"],
                "thr_f1": thr["f1"],
                "thr_balacc": thr["balanced_accuracy"],
                "thr_youden": thr["youden_j"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison figures (same visual language as fable_eval)
# ---------------------------------------------------------------------------

def figures(folds, summary, lopo_all, lopo_pooled, out_dir: Path) -> None:
    # 1. interpolation forest plot (reuses the shared style helper)
    ordered = folds.copy()
    order = summary.index.tolist()
    ordered["method"] = pd.Categorical(ordered["method"], categories=order, ordered=True)
    plot_fold_distribution(
        ordered.sort_values("method"),
        "Five-classifier comparison: repeated grouped CV on unseen papers",
        out_dir / "fig_compare_grouped_cv.png",
    )

    if lopo_pooled:
        # 2. pooled LOPO extrapolation bar chart
        names = list(lopo_pooled)
        vals = [lopo_pooled[n]["pooled_roc_auc"] for n in names]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        yy = np.arange(len(names))
        ax.barh(yy, vals, color="#27598e")
        ax.set_yticks(yy)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("pooled LOPO ROC-AUC")
        ax.set_xlim(0.4, 1.0)
        ax.axvline(0.5, color="k", ls=":", lw=0.8)
        ax.set_title("Extrapolation to entirely new papers (Leave-One-Paper-Out)")
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "fig_compare_lopo_pooled.png", bbox_inches="tight")
        plt.close(fig)

    if lopo_all is not None and len(lopo_all):
        # 3. per-paper LOPO AUC distribution per model
        fig, ax = plt.subplots(figsize=(7, 3.6))
        names = list(lopo_all["method"].unique())
        data = [lopo_all[lopo_all["method"] == n]["auc"].dropna() for n in names]
        bp = ax.boxplot(data, orientation="horizontal", tick_labels=names, patch_artist=True, widths=0.6)
        for patch in bp["boxes"]:
            patch.set_facecolor("#27598e")
            patch.set_alpha(0.55)
        ax.axvline(0.5, color="k", ls=":", lw=0.8)
        ax.set_xlabel("within-paper AUC (LOPO)")
        ax.set_title("Per-paper extrapolation performance by classifier")
        plt.tight_layout()
        plt.savefig(out_dir / "fig_compare_lopo_per_paper.png", bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="Microgravity_Database.xlsm")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "compare_outputs"))
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--skip-lopo", action="store_true")
    ap.add_argument("--quick", action="store_true", help="1 seed, no LOPO")
    ap.add_argument("--random-state", type=int, default=RNG)
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.quick:
        out_dir = out_dir.with_name(out_dir.name + "_quick")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    print(
        f"Loaded {len(df)} rows / {df['paper_id'].nunique()} canonical papers "
        f"({int(df['ignition_binary'].sum())} ignition, "
        f"{int((1 - df['ignition_binary']).sum())} no-ignition)\n"
    )

    registry = model_registry(args.random_state)
    n_seeds = 1 if args.quick else args.n_seeds

    # ---- interpolation: row-level protocols (same distribution as training) ----
    interp_random_frames = []
    interp_kfold_frames = []
    for i, (name, (factory, _)) in enumerate(registry.items()):
        print(f"[interp random {i + 1}/{len(registry)}] {name} ...")
        folds = benchmark_random_holdout(
            name, factory, df, n_seeds=5 if not args.quick else 1
        )
        print(
            f"    random 80/20 ROC-AUC = {folds['roc_auc'].mean():.4f} "
            f"+/- {folds['roc_auc'].std():.4f}"
        )
        interp_random_frames.append(folds)

        print(f"[interp 5-fold {i + 1}/{len(registry)}] {name} ...")
        folds = benchmark_stratified_kfold(name, factory, df)
        print(
            f"    stratified 5-fold ROC-AUC = {folds['roc_auc'].mean():.4f} "
            f"+/- {folds['roc_auc'].std():.4f}"
        )
        interp_kfold_frames.append(folds)

    interp_random = pd.concat(interp_random_frames, ignore_index=True)
    interp_kfold = pd.concat(interp_kfold_frames, ignore_index=True)
    interp_random.to_csv(out_dir / "interpolation_random_folds.csv", index=False)
    interp_kfold.to_csv(out_dir / "interpolation_stratified_folds.csv", index=False)

    interp_random_summary = summarise_folds(interp_random)
    interp_kfold_summary = summarise_folds(interp_kfold)
    interp_random_summary.to_csv(out_dir / "interpolation_random_summary.csv")
    interp_kfold_summary.to_csv(out_dir / "interpolation_stratified_summary.csv")

    md_random = to_markdown_table(interp_random_summary)
    md_kfold = to_markdown_table(interp_kfold_summary)
    print("\n" + md_random + "\n")
    print("\n" + md_kfold + "\n")

    # ---- extrapolation: grouped paper splits + LOPO ----
    grouped_splits = make_outer_splits(df, n_seeds=n_seeds, base_seed=args.random_state)
    grouped_frames = []
    for i, (name, (factory, weight_fn)) in enumerate(registry.items()):
        print(f"[extrap grouped {i + 1}/{len(registry)}] {name} ...")
        folds = benchmark_interpolation(
            name, factory, df, grouped_splits, weight_fn, base_seed=args.random_state
        )
        print(
            f"    grouped ROC-AUC = {folds['roc_auc'].mean():.4f} "
            f"+/- {folds['roc_auc'].std():.4f} | MCC = {folds['mcc'].mean():.3f}"
        )
        grouped_frames.append(folds)

    grouped = pd.concat(grouped_frames, ignore_index=True)
    grouped.to_csv(out_dir / "extrapolation_grouped_folds.csv", index=False)

    grouped_summary = summarise_folds(grouped)
    grouped_summary.to_csv(out_dir / "extrapolation_grouped_summary.csv")
    md_grouped = to_markdown_table(grouped_summary)
    print("\n" + md_grouped + "\n")

    pw = pairwise_tests(grouped)
    pw.to_csv(out_dir / "pairwise_tests.csv", index=False)
    print("Pairwise paired-fold deltas (Wilcoxon):")
    print(pw[["model_a", "model_b", "delta_roc_auc", "wilcoxon_p_roc_auc"]]
        .round(4).to_string(index=False))

    # ---- extrapolation: LOPO for every model on identical paper splits ----
    lopo_all, lopo_pooled = None, {}
    if not (args.quick or args.skip_lopo):
        lopo_frames = []
        for i, (name, (factory, _)) in enumerate(registry.items()):
            print(f"[LOPO {i + 1}/{len(registry)}] {name} ...")
            per_paper, pooled, _ = lopo_evaluate(name, factory, df)
            lopo_frames.append(per_paper)
            lopo_pooled[name] = pooled
            print(f"    pooled LOPO ROC-AUC = {pooled['pooled_roc_auc']:.4f}")
        lopo_all = pd.concat(lopo_frames, ignore_index=True)
        lopo_all.to_csv(out_dir / "lopo_per_paper.csv", index=False)
        with open(out_dir / "lopo_pooled.json", "w") as f:
            json.dump(lopo_pooled, f, indent=2)

    # ---- reports -----------------------------------------------------------
    lines = [
        "# Fable five-classifier benchmark",
        "",
        "All interpolation protocols use row-level splits; all extrapolation protocols hold out whole papers. Thresholds are tuned on inner held-out-paper folds only.",
        "",
        "## Interpolation",
        "",
        "### Random stratified 80/20 (x5 seeds)",
        "",
        md_random,
        "",
        "### Stratified 5-fold",
        "",
        md_kfold,
        "",
        "## Extrapolation",
        "",
        "### Repeated grouped CV (StratifiedGroupKFold 5x seeds)",
        "",
        md_grouped,
    ]
    if lopo_pooled:
        lines += ["", "### Leave-One-Paper-Out (pooled)", "",
                  "| Model | Pooled ROC-AUC | Pooled PR-AUC |", "|---|---|---|"]
        for name, p in sorted(lopo_pooled.items(),
                              key=lambda kv: -kv[1]["pooled_roc_auc"]):
            lines.append(f"| {name} | {p['pooled_roc_auc']:.3f} "
                         f"| {p['pooled_pr_auc']:.3f} |")
    lines += ["", "### Pairwise comparisons on grouped CV (paired folds, Wilcoxon)", "",
              "| A | B | dROC-AUC | p | dMCC | p |", "|---|---|---|---|---|---|"]
    for _, r in pw.iterrows():
        lines.append(
            f"| {r['model_a']} | {r['model_b']} "
            f"| {r['delta_roc_auc']:+.3f} | {r['wilcoxon_p_roc_auc']:.3g} "
            f"| {r['delta_mcc']:+.3f} | {r['wilcoxon_p_mcc']:.3g} |"
        )
    (out_dir / "comparison_summary.md").write_text("\n".join(lines) + "\n")

    with open(out_dir / "comparison_summary.json", "w") as f:
        json.dump(
            {
                "n_rows": int(len(df)),
                "n_papers": int(df["paper_id"].nunique()),
                "n_seeds": n_seeds,
                "interpolation_random": interp_random_summary.reset_index().to_dict(orient="records"),
                "interpolation_stratified": interp_kfold_summary.reset_index().to_dict(orient="records"),
                "extrapolation_grouped": grouped_summary.reset_index().to_dict(orient="records"),
                "lopo_pooled": lopo_pooled,
                "pairwise": pw.to_dict(orient="records"),
            },
            f, indent=2,
        )

    figures(grouped, grouped_summary, lopo_all, lopo_pooled, out_dir)
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()