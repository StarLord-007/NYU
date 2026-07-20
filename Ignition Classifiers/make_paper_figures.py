"""Generate supplementary publication figures for PAPER.md.

Reads only persisted pipeline outputs (results/splits, results/evaluation) and the
data-validation report; it never retrains or re-evaluates models. Figures are written
to results/report/paper_figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
OUT = RESULTS / "report" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

BLUE = "#1f77b4"
ORANGE = "#d95f02"
GREEN = "#1b9e77"
GRAY = "#666666"

MODEL_ORDER_LABELS = {
    "xgb_all_unweighted": "XGBoost (all)",
    "xgb_physics_unweighted": "XGBoost (physics)",
    "xgb_all_class_paper_weighted": "XGBoost (all, weighted)",
    "xgb_physics_class_paper_weighted_monotone_o2": "XGBoost (physics, weighted, monotone O$_2$)",
    "xgb_physics_paper_bagging": "XGBoost (physics, paper-bagged)",
    "xgb_all_focal_class_paper_weighted": "XGBoost (all, focal loss)",
    "xgb_physics_focal_class_paper_weighted": "XGBoost (physics, focal loss)",
    "mlp_all": "MLP (all)",
    "mlp_physics": "MLP (physics)",
    "svm_all": "SVM (all)",
    "svm_physics": "SVM (physics)",
    "knn_all": "KNN (all)",
    "knn_physics": "KNN (physics)",
    "decision_tree_all": "Decision tree (all)",
    "decision_tree_physics": "Decision tree (physics)",
}


def fig_dataset_overview() -> None:
    validation = json.loads((RESULTS / "splits" / "data_validation_report.json").read_text())
    splits_meta = json.loads((RESULTS / "splits" / "splits_metadata.json").read_text())

    per_paper = pd.read_csv(RESULTS / "evaluation" / "per_paper_metrics.csv")
    lopo = (
        per_paper[
            (per_paper["protocol"] == "lopo") & (per_paper["model_id"] == "xgb_physics_unweighted")
        ]
        .sort_values("n_unique_rows", ascending=False)
        .drop_duplicates("paper_id", keep="first")
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    ax = axes[0]
    counts = splits_meta["label_counts"]
    bars = ax.bar(
        ["Ignition\n(positive)", "No ignition\n(negative)"],
        [counts["1"], counts["0"]],
        color=[ORANGE, BLUE],
        width=0.55,
    )
    for bar, n in zip(bars, [counts["1"], counts["0"]]):
        share = n / (counts["1"] + counts["0"])
        ax.annotate(
            f"{n:,}\n({share:.1%})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 4100)
    ax.set_ylabel("Experimental records")
    ax.set_title("(a) Class balance (n = 4,499)")

    ax = axes[1]
    rows_per_paper = lopo["n_unique_rows"].to_numpy()
    ax.hist(rows_per_paper, bins=np.geomspace(1, rows_per_paper.max(), 18), color=GREEN, edgecolor="white")
    ax.set_xscale("log")
    ax.axvline(np.median(rows_per_paper), color="black", linestyle="--", linewidth=1)
    ax.annotate(
        f"median = {np.median(rows_per_paper):.0f}",
        (np.median(rows_per_paper) * 1.15, ax.get_ylim()[1] * 0.9),
        fontsize=9,
    )
    ax.set_xlabel("Records contributed per paper (log scale)")
    ax.set_ylabel("Papers")
    ax.set_title("(b) Records per source paper (85 papers)")

    ax = axes[2]
    prevalence = lopo["ignition_prevalence"].to_numpy()
    ax.hist(prevalence, bins=np.linspace(0, 1, 21), color=BLUE, edgecolor="white")
    ax.axvline(splits_meta["label_prevalence"], color="black", linestyle="--", linewidth=1)
    ax.annotate(
        f"pooled prevalence = {splits_meta['label_prevalence']:.2f}",
        (0.02, ax.get_ylim()[1] * 0.9),
        fontsize=9,
    )
    ax.set_xlabel("Within-paper ignition prevalence")
    ax.set_ylabel("Papers")
    ax.set_title("(c) Ignition prevalence by paper")

    fig.tight_layout()
    fig.savefig(OUT / "fig_dataset_overview.png", bbox_inches="tight")
    plt.close(fig)

    single_class = int(((prevalence == 0) | (prevalence == 1)).sum())
    print(
        json.dumps(
            {
                "rows_per_paper_median": float(np.median(rows_per_paper)),
                "rows_per_paper_max": int(rows_per_paper.max()),
                "rows_per_paper_min": int(rows_per_paper.min()),
                "single_class_papers": single_class,
                "papers": int(len(lopo)),
                "missing_validation_row_count": validation["row_count"],
            },
            indent=2,
        )
    )


def fig_missingness() -> None:
    validation = json.loads((RESULTS / "splits" / "data_validation_report.json").read_text())
    miss = validation["missingness_by_model_feature"]
    n = validation["row_count"]
    manifest = {f["feature"]: f["role"] for f in validation["feature_manifest"]}

    frac = {k: v / n for k, v in miss.items() if k in manifest}
    order = sorted(frac, key=frac.get)
    colors = [GREEN if manifest[f] == "physics" else GRAY for f in order]

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(range(len(order)), [frac[f] for f in order], color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Fraction of records missing")
    ax.set_title("Missingness of model features (n = 4,499 records)")
    ax.set_xlim(0, 1.05)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=GREEN, label="physics feature (both feature sets)"),
            Patch(color=GRAY, label="apparatus feature ('all' set only)"),
        ],
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_feature_missingness.png", bbox_inches="tight")
    plt.close(fig)


def fig_bootstrap_forest() -> None:
    ci = pd.read_csv(RESULTS / "evaluation" / "bootstrap_intervals.csv")
    roc = ci[ci["metric"] == "roc_auc"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), sharey=True)
    for ax, protocol, color, title in [
        (axes[0], "interpolation_stratified", BLUE, "Interpolation (stratified 5-fold, row-level bootstrap)"),
        (axes[1], "extrapolation_grouped", ORANGE, "Extrapolation (grouped 5-fold, paper-cluster bootstrap)"),
    ]:
        sub = roc[roc["protocol"] == protocol].set_index("model_id")
        # order by extrapolation estimate for a stable, comparable ordering
        order_source = roc[roc["protocol"] == "extrapolation_grouped"].set_index("model_id")["estimate"]
        order = order_source.sort_values().index.tolist()
        y = np.arange(len(order))
        est = sub.loc[order, "estimate"]
        lo = sub.loc[order, "ci_low"]
        hi = sub.loc[order, "ci_high"]
        ax.errorbar(
            est,
            y,
            xerr=[est - lo, hi - est],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.6,
            capsize=3,
            markersize=5,
        )
        ax.axvline(0.5, color="black", linestyle=":", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_ORDER_LABELS.get(m, m) for m in order], fontsize=9)
        ax.set_xlabel("ROC-AUC (95% bootstrap CI)")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0.38, 1.0)
    axes[0].annotate("chance", (0.505, 0.2), fontsize=8, color="black", rotation=90)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bootstrap_forest.png", bbox_inches="tight")
    plt.close(fig)


def fig_champion_across_protocols() -> None:
    ci = pd.read_csv(RESULTS / "evaluation" / "bootstrap_intervals.csv")
    protocols = [
        ("interpolation_holdout", "Repeated 80/20\nholdout"),
        ("interpolation_stratified", "Stratified\n5-fold CV"),
        ("extrapolation_grouped", "Grouped 5-fold\n(unseen papers)"),
        ("lopo", "Leave-one-\npaper-out"),
    ]
    champions = [
        ("xgb_all_unweighted", "Interpolation champion: XGBoost (all features)", BLUE),
        ("xgb_physics_unweighted", "Extrapolation champion: XGBoost (physics features)", ORANGE),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, metric, label in [(axes[0], "roc_auc", "ROC-AUC"), (axes[1], "pr_auc", "PR-AUC")]:
        x = np.arange(len(protocols))
        for i, (model, name, color) in enumerate(champions):
            sub = ci[(ci["model_id"] == model) & (ci["metric"] == metric)].set_index("protocol")
            est = np.array([sub.loc[p, "estimate"] for p, _ in protocols])
            lo = np.array([sub.loc[p, "ci_low"] for p, _ in protocols])
            hi = np.array([sub.loc[p, "ci_high"] for p, _ in protocols])
            ax.errorbar(
                x + (i - 0.5) * 0.14,
                est,
                yerr=[est - lo, hi - est],
                fmt="o",
                color=color,
                capsize=4,
                markersize=6,
                label=name,
            )
        if metric == "roc_auc":
            ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
            ax.annotate("chance", (2.98, 0.512), fontsize=8)
        else:
            ax.axhline(0.754, color="black", linestyle=":", linewidth=1)
            ax.annotate("prevalence baseline (0.754)", (1.8, 0.762), fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in protocols], fontsize=9)
        ax.set_ylabel(f"{label} (95% bootstrap CI)")
        ax.set_ylim(0.45, 1.0)
    axes[0].legend(loc="lower left", fontsize=8, frameon=True)
    fig.suptitle("Champion performance across evaluation protocols", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_champion_across_protocols.png", bbox_inches="tight")
    plt.close(fig)


def fig_lopo_per_paper() -> None:
    per_paper = pd.read_csv(RESULTS / "evaluation" / "per_paper_metrics.csv")
    lopo = per_paper[
        (per_paper["protocol"] == "lopo") & (per_paper["model_id"] == "xgb_physics_unweighted")
    ].copy()
    # A couple of canonical papers appear twice with partial row subsets; keep the
    # larger evaluation per paper so every bar corresponds to one unique paper.
    lopo = (
        lopo.sort_values("n_unique_rows", ascending=False)
        .drop_duplicates("paper_id", keep="first")
    )

    defined = lopo.dropna(subset=["roc_auc"]).sort_values("roc_auc")
    undefined = lopo[lopo["roc_auc"].isna()]

    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.6, 1.0]}
    )

    ax = axes[0]
    colors = [ORANGE if v < 0.5 else BLUE for v in defined["roc_auc"]]
    ax.bar(range(len(defined)), defined["roc_auc"], color=colors, width=0.85)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
    median = defined["roc_auc"].median()
    ax.axhline(median, color=GREEN, linestyle="--", linewidth=1.2)
    ax.annotate(f"median = {median:.2f}", (1, median + 0.02), color=GREEN, fontsize=9)
    ax.set_xlabel(f"Held-out paper (n = {len(defined)} papers with both outcomes present)")
    ax.set_ylabel("Within-paper ROC-AUC")
    ax.set_title("(a) LOPO ROC-AUC per held-out paper — extrapolation champion")
    ax.set_xticks([])
    ax.set_ylim(0, 1.02)

    ax = axes[1]
    ax.scatter(
        defined["n_unique_rows"],
        defined["roc_auc"],
        c=[ORANGE if v < 0.5 else BLUE for v in defined["roc_auc"]],
        s=28,
        alpha=0.85,
    )
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Records in held-out paper (log scale)")
    ax.set_ylabel("Within-paper ROC-AUC")
    ax.set_title("(b) LOPO ROC-AUC vs. paper size")
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(OUT / "fig_lopo_per_paper.png", bbox_inches="tight")
    plt.close(fig)

    print(
        json.dumps(
            {
                "lopo_defined_papers": int(len(defined)),
                "lopo_undefined_papers": int(len(undefined)),
                "lopo_median_auc": float(median),
                "lopo_papers_below_chance": int((defined["roc_auc"] < 0.5).sum()),
                "lopo_papers_above_0.9": int((defined["roc_auc"] > 0.9).sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    fig_dataset_overview()
    fig_missingness()
    fig_bootstrap_forest()
    fig_champion_across_protocols()
    fig_lopo_per_paper()
    print(f"Figures written to {OUT}")
