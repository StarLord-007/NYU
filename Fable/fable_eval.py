"""
fable_eval.py
=============
Benchmark and evaluation harness for unseen-paper generalisation.

Protocol
--------
* **Primary protocol:** StratifiedGroupKFold(5) over canonical papers,
  repeated with ``--n-seeds`` shuffles (default 3 -> 15 held-out-paper test
  folds). Every test fold contains only papers the model never saw.
* **Thresholds** (MCC / F1 / balanced accuracy / Youden J) are optimised on
  *inner* validation folds (StratifiedGroupKFold(3) over the training
  papers of each outer fold) and then applied, frozen, to the outer test
  fold. The test labels never influence the decision rule.
* **Secondary protocols** for the validation-strategy comparison:
  random stratified hold-out, stratified 5-fold, GroupKFold,
  Leave-One-Paper-Out (LOPO, for selected methods).

Reported per method (mean +/- std over outer folds):
  Group ROC-AUC, Group PR-AUC, MCC, F1, Balanced Accuracy (each at the
  threshold tuned for that metric on inner folds).

Outputs -> ``Fable/eval_outputs/``:
  * ``comparison_table.csv`` / ``comparison_table.md``
  * ``protocol_comparison.csv``  (random vs stratified vs grouped vs LOPO)
  * ``lopo_baseline_vs_best.csv``
  * publication figures (``fig_*.png``)

Run::

    python Fable/fable_eval.py --data Microgravity_Database_Latest.csv
    python Fable/fable_eval.py --quick          # smoke test (1 seed, fewer methods)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from fable_common import load_clean
from fable_train import (
    DEFAULT_PARAMS,
    GROUP_TUNED_PARAMS,
    FableConfig,
    FableModel,
    optimal_thresholds,
)

RNG = 42

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def method_registry() -> dict[str, FableConfig]:
    """All benchmarked configurations.

    The first block isolates the weighting axis (requirements A & B), the
    second the loss axis (D), the third hyperparameters (E) and domain
    generalisation (F). Components are then combined.
    """
    D = dict(DEFAULT_PARAMS)
    G = dict(GROUP_TUNED_PARAMS)
    m: dict[str, FableConfig] = {}

    # --- baseline & class weighting -------------------------------------
    m["baseline (spw, v2 config)"] = FableConfig(
        feature_set="all", paper_weight="none", class_weight="spw", params=D
    )
    m["class-only (balanced)"] = FableConfig(
        feature_set="all", paper_weight="none", class_weight="balanced", params=D
    )

    # --- paper weighting (A) ---------------------------------------------
    m["paper 1/N"] = FableConfig(
        feature_set="all", paper_weight="inverse", class_weight="none", params=D
    )
    m["paper 1/sqrt(N)"] = FableConfig(
        feature_set="all", paper_weight="sqrt", class_weight="none", params=D
    )
    m["paper effective-number"] = FableConfig(
        feature_set="all", paper_weight="effective", class_weight="none", params=D
    )
    m["paper 1/(1+lnN)"] = FableConfig(
        feature_set="all", paper_weight="log", class_weight="none", params=D
    )

    # --- combined class x paper (B) ---------------------------------------
    m["combined 1/N x class"] = FableConfig(
        feature_set="all", paper_weight="inverse", class_weight="balanced", params=D
    )
    m["combined 1/sqrt(N) x class"] = FableConfig(
        feature_set="all", paper_weight="sqrt", class_weight="balanced", params=D
    )

    # --- focal loss (D) ----------------------------------------------------
    m["focal g=1"] = FableConfig(
        feature_set="all", paper_weight="none", class_weight="none",
        objective="focal", focal_gamma=1.0, params=D,
    )
    m["focal g=2"] = FableConfig(
        feature_set="all", paper_weight="none", class_weight="none",
        objective="focal", focal_gamma=2.0, params=D,
    )
    m["focal g=2 + paper sqrt"] = FableConfig(
        feature_set="all", paper_weight="sqrt", class_weight="none",
        objective="focal", focal_gamma=2.0, params=D,
    )

    # --- group-tuned hyperparameters (E) -----------------------------------
    m["group-tuned HP"] = FableConfig(
        feature_set="all", paper_weight="none", class_weight="spw", params=G
    )
    m["group-tuned HP + combined w"] = FableConfig(
        feature_set="all", paper_weight="sqrt", class_weight="balanced", params=G
    )

    # --- domain generalisation (F) ------------------------------------------
    m["physics features"] = FableConfig(
        feature_set="physics", paper_weight="none", class_weight="spw", params=D
    )
    m["physics + combined w"] = FableConfig(
        feature_set="physics", paper_weight="sqrt", class_weight="balanced", params=D
    )
    m["physics + combined w + group HP"] = FableConfig(
        feature_set="physics", paper_weight="sqrt", class_weight="balanced", params=G
    )
    m["fable (= above + monotone O2)"] = FableConfig(
        feature_set="physics", paper_weight="sqrt", class_weight="balanced",
        monotone_o2=True, params=G,
    )
    m["fable + paper-bagging 25"] = FableConfig(
        feature_set="physics", paper_weight="sqrt", class_weight="balanced",
        monotone_o2=True, paper_bagging=25, params=G,
    )
    return m


QUICK_METHODS = [
    "baseline (spw, v2 config)",
    "combined 1/sqrt(N) x class",
    "fable (= above + monotone O2)",
]


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

def _inner_thresholds(
    cfg: FableConfig,
    X: pd.DataFrame,
    y: np.ndarray,
    papers: pd.Series,
    seed: int,
) -> dict[str, float]:
    """Thresholds tuned on inner held-out-paper folds of the training data."""
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in inner.split(X, y, groups=papers):
        m = FableModel(cfg)
        m.fit(X.iloc[tr], y[tr], papers.iloc[tr])
        oof[te] = m.predict_proba(X.iloc[te])
    return optimal_thresholds(y, oof)


def benchmark_method(
    name: str,
    cfg: FableConfig,
    df: pd.DataFrame,
    n_seeds: int = 3,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Repeated StratifiedGroupKFold evaluation with nested threshold tuning."""
    X = df
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"]
    rows = []
    for seed in range(n_seeds):
        outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RNG + seed)
        for fold, (tr, te) in enumerate(outer.split(X, y, groups=papers)):
            Xtr, ytr, ptr = X.iloc[tr], y[tr], papers.iloc[tr]
            Xte, yte = X.iloc[te], y[te]

            thr = _inner_thresholds(cfg, Xtr, ytr, ptr, seed=RNG + seed)
            model = FableModel(cfg)
            model.fit(Xtr, ytr, ptr)
            p = model.predict_proba(Xte)

            row = {
                "method": name,
                "seed": seed,
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
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    agg = folds.groupby("method").agg(
        group_roc_auc=("roc_auc", "mean"),
        group_roc_auc_std=("roc_auc", "std"),
        pr_auc=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        mcc=("mcc", "mean"),
        mcc_std=("mcc", "std"),
        f1=("f1", "mean"),
        f1_std=("f1", "std"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
    )
    return agg.sort_values("group_roc_auc", ascending=False)


def to_markdown_table(summary: pd.DataFrame) -> str:
    lines = [
        "| Method | Group ROC-AUC | PR-AUC | MCC | Balanced Accuracy |",
        "|---|---|---|---|---|",
    ]
    for name, r in summary.iterrows():
        lines.append(
            f"| {name} "
            f"| {r['group_roc_auc']:.3f} +/- {r['group_roc_auc_std']:.3f} "
            f"| {r['pr_auc']:.3f} +/- {r['pr_auc_std']:.3f} "
            f"| {r['mcc']:.3f} +/- {r['mcc_std']:.3f} "
            f"| {r['balanced_accuracy']:.3f} +/- {r['balanced_accuracy_std']:.3f} |"
        )
    return "\n".join(lines)


def paired_analysis(folds: pd.DataFrame, baseline: str, out_dir: Path) -> pd.DataFrame:
    """Paired per-fold comparison of every method against the baseline.

    All methods are evaluated on identical (seed, fold) partitions, so paired
    differences remove the (large, +/-0.05-0.07) partition variance and give
    the honest answer to "does method X beat the baseline on unseen papers?".
    Wilcoxon signed-rank test over the 15 paired folds.
    """
    from scipy.stats import wilcoxon

    base = folds[folds["method"] == baseline].set_index(["seed", "fold"])
    rows = []
    for name, sub in folds.groupby("method"):
        if name == baseline:
            continue
        sub = sub.set_index(["seed", "fold"])
        common = base.index.intersection(sub.index)
        rec = {"method": name, "n_folds": len(common)}
        for metric in ["roc_auc", "pr_auc", "mcc", "balanced_accuracy"]:
            d = (sub.loc[common, metric] - base.loc[common, metric]).to_numpy()
            rec[f"delta_{metric}"] = float(np.mean(d))
            try:
                rec[f"wilcoxon_p_{metric}"] = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            except ValueError:
                rec[f"wilcoxon_p_{metric}"] = np.nan
        rows.append(rec)
    res = pd.DataFrame(rows).sort_values("delta_roc_auc", ascending=False)
    res.to_csv(out_dir / "paired_vs_baseline.csv", index=False)
    return res


# ---------------------------------------------------------------------------
# Validation-protocol comparison (audit question 3)
# ---------------------------------------------------------------------------

def protocol_comparison(df: pd.DataFrame, cfg: FableConfig, out_dir: Path) -> pd.DataFrame:
    """Same model, four validation protocols. Shows the optimism ladder."""
    X = df
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"]
    results = []

    # 1. random stratified hold-out (5 repeats)
    aucs = []
    for s in range(5):
        tr, te = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=RNG + s
        )
        m = FableModel(cfg)
        m.fit(X.iloc[tr], y[tr], papers.iloc[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])))
    results.append(("random stratified 80/20 hold-out", np.mean(aucs), np.std(aucs)))

    # 2. stratified 5-fold
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=RNG).split(X, y):
        m = FableModel(cfg)
        m.fit(X.iloc[tr], y[tr], papers.iloc[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])))
    results.append(("stratified 5-fold CV", np.mean(aucs), np.std(aucs)))

    # 3. GroupKFold by paper
    aucs = []
    for tr, te in GroupKFold(5).split(X, y, groups=papers):
        m = FableModel(cfg)
        m.fit(X.iloc[tr], y[tr], papers.iloc[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])))
    results.append(("GroupKFold(5) by paper", np.mean(aucs), np.std(aucs)))

    # 4. StratifiedGroupKFold (3 seeds)
    aucs = []
    for s in range(3):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=RNG + s).split(
            X, y, groups=papers
        ):
            m = FableModel(cfg)
            m.fit(X.iloc[tr], y[tr], papers.iloc[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])))
    results.append(("StratifiedGroupKFold(5) x3 seeds", np.mean(aucs), np.std(aucs)))

    # 5. Leave-One-Paper-Out (pooled AUC over all out-of-paper predictions)
    oof = np.full(len(y), np.nan)
    for pid in papers.unique():
        mask = (papers == pid).to_numpy()
        m = FableModel(cfg)
        m.fit(X.iloc[~mask], y[~mask], papers[~mask])
        oof[mask] = m.predict_proba(X.iloc[mask])
    results.append(("Leave-One-Paper-Out (pooled)", roc_auc_score(y, oof), np.nan))

    prot = pd.DataFrame(results, columns=["protocol", "roc_auc_mean", "roc_auc_std"])
    prot.to_csv(out_dir / "protocol_comparison.csv", index=False)
    return prot


# ---------------------------------------------------------------------------
# LOPO head-to-head: baseline vs best
# ---------------------------------------------------------------------------

def lopo_head_to_head(
    df: pd.DataFrame, cfg_a: FableConfig, cfg_b: FableConfig, names: tuple[str, str], out_dir: Path
) -> pd.DataFrame:
    X = df
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"]
    rows = []
    oof = {names[0]: np.full(len(y), np.nan), names[1]: np.full(len(y), np.nan)}
    for pid in papers.unique():
        mask = (papers == pid).to_numpy()
        rec = {
            "paper_id": pid,
            "paper_label": df.loc[mask, "paper_label"].iloc[0],
            "n_rows": int(mask.sum()),
            "ignition_rate": float(y[mask].mean()),
        }
        for cfg, nm in zip((cfg_a, cfg_b), names):
            m = FableModel(cfg)
            m.fit(X.iloc[~mask], y[~mask], papers[~mask])
            p = m.predict_proba(X.iloc[mask])
            oof[nm][mask] = p
            rec[f"auc_{nm}"] = (
                roc_auc_score(y[mask], p) if len(np.unique(y[mask])) == 2 else np.nan
            )
        rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "lopo_baseline_vs_best.csv", index=False)
    pooled = {nm: float(roc_auc_score(y, oof[nm])) for nm in names}
    with open(out_dir / "lopo_pooled.json", "w") as f:
        json.dump(pooled, f, indent=2)
    return res


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figures(
    folds: pd.DataFrame,
    summary: pd.DataFrame,
    prot: pd.DataFrame,
    lopo: pd.DataFrame | None,
    out_dir: Path,
) -> None:
    # 1. method comparison: grouped ROC-AUC with per-fold scatter
    order = summary.index.tolist()[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.6))
    for i, mname in enumerate(order):
        sub = folds[folds["method"] == mname]["roc_auc"]
        ax.scatter(sub, np.full(len(sub), i), alpha=0.45, s=14, color="#27598e", zorder=2)
        ax.errorbar(
            sub.mean(), i, xerr=sub.std(), fmt="o", color="#c23b22",
            capsize=3, markersize=6, zorder=3,
        )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("held-out-paper ROC-AUC (StratifiedGroupKFold 5x3)")
    ax.set_title("Unseen-paper discrimination by method\n(blue = individual folds, red = mean +/- sd)")
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_method_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # 2. protocol optimism ladder
    fig, ax = plt.subplots(figsize=(7, 3.2))
    yy = np.arange(len(prot))
    ax.barh(yy, prot["roc_auc_mean"], xerr=prot["roc_auc_std"].fillna(0), color="#27598e", capsize=3)
    ax.set_yticks(yy)
    ax.set_yticklabels(prot["protocol"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC")
    ax.set_xlim(0.5, 1.0)
    ax.set_title("Same model, different validation protocols:\nrandom splits overstate generalisation to new papers")
    for i, v in enumerate(prot["roc_auc_mean"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_protocol_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # 3. LOPO per-paper baseline vs best scatter
    if lopo is not None:
        cols = [c for c in lopo.columns if c.startswith("auc_")]
        a, b = cols[0], cols[1]
        two = lopo.dropna(subset=[a, b])
        fig, ax = plt.subplots(figsize=(4.8, 4.6))
        sizes = 8 + 60 * (two["n_rows"] / two["n_rows"].max())
        ax.scatter(two[a], two[b], s=sizes, alpha=0.6, color="#27598e")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.axhline(0.5, color="gray", ls=":", lw=0.6)
        ax.axvline(0.5, color="gray", ls=":", lw=0.6)
        ax.set_xlabel(f"within-paper AUC: {a[4:]}")
        ax.set_ylabel(f"within-paper AUC: {b[4:]}")
        n_up = int((two[b] > two[a]).sum())
        ax.set_title(
            f"Leave-One-Paper-Out, paper by paper\n"
            f"{n_up}/{len(two)} papers improved (marker size = paper size)"
        )
        plt.tight_layout()
        plt.savefig(out_dir / "fig_lopo_scatter.png", bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="Microgravity_Database_Latest.csv")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "eval_outputs"))
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="1 seed, 3 methods, no LOPO")
    ap.add_argument("--skip-lopo", action="store_true")
    ap.add_argument("--skip-protocols", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.quick:
        # never clobber the full benchmark artifacts with a smoke test
        out_dir = out_dir.with_name(out_dir.name + "_quick")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean(args.data)
    print(
        f"Loaded {len(df)} rows / {df['paper_id'].nunique()} canonical papers "
        f"({int(df['ignition_binary'].sum())} ignition, "
        f"{int((1 - df['ignition_binary']).sum())} no-ignition)\n"
    )

    methods = method_registry()
    if args.quick:
        methods = {k: v for k, v in methods.items() if k in QUICK_METHODS}
    n_seeds = 1 if args.quick else args.n_seeds

    all_folds = []
    for i, (name, cfg) in enumerate(methods.items()):
        print(f"[{i + 1}/{len(methods)}] {name} ...")
        folds = benchmark_method(name, cfg, df, n_seeds=n_seeds)
        m = folds["roc_auc"].mean()
        s = folds["roc_auc"].std()
        print(f"    group ROC-AUC = {m:.4f} +/- {s:.4f} | MCC = {folds['mcc'].mean():.3f}")
        all_folds.append(folds)
    folds = pd.concat(all_folds, ignore_index=True)
    folds.to_csv(out_dir / "benchmark_folds.csv", index=False)

    summary = summarise(folds)
    summary.to_csv(out_dir / "comparison_table.csv")
    md = to_markdown_table(summary)
    (out_dir / "comparison_table.md").write_text(md + "\n")
    print("\n" + md + "\n")

    base_name = "baseline (spw, v2 config)"
    if base_name in folds["method"].unique() and folds["method"].nunique() > 1:
        paired = paired_analysis(folds, base_name, out_dir)
        print("Paired per-fold deltas vs baseline (Wilcoxon over identical folds):")
        cols = ["method", "delta_roc_auc", "wilcoxon_p_roc_auc", "delta_mcc", "delta_balanced_accuracy"]
        print(paired[cols].round(4).to_string(index=False))

    prot = None
    if not args.skip_protocols:
        print("Protocol comparison (baseline config) ...")
        prot = protocol_comparison(df, methods.get(
            "baseline (spw, v2 config)",
            FableConfig(feature_set="all", paper_weight="none", class_weight="spw",
                        params=dict(DEFAULT_PARAMS)),
        ), out_dir)
        print(prot.round(4).to_string(index=False))

    lopo = None
    if not (args.quick or args.skip_lopo):
        print("\nLOPO head-to-head: baseline vs fable ...")
        best_name = "fable (= above + monotone O2)"
        lopo = lopo_head_to_head(
            df,
            methods["baseline (spw, v2 config)"],
            methods[best_name],
            ("baseline", "fable"),
            out_dir,
        )
        two = lopo.dropna(subset=["auc_baseline", "auc_fable"])
        print(
            f"  papers improved: {(two['auc_fable'] > two['auc_baseline']).sum()}/{len(two)} | "
            f"mean within-paper AUC {two['auc_baseline'].mean():.3f} -> {two['auc_fable'].mean():.3f}"
        )

    if prot is not None:
        figures(folds, summary, prot, lopo, out_dir)
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
