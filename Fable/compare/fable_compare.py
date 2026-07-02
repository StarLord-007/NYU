from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alt_model_support import (
    benchmark_on_splits,
    lopo_n_way,
    make_repeated_splits,
    pairwise_wilcoxon_table,
    summarise_folds,
    to_markdown_table,
    load_clean_any,
)
from fable_train import DEFAULT_PARAMS as XGB_DEFAULT_PARAMS, FableConfig, FableModel
from knn.fable_knn import GROUP_TUNED_PARAMS as KNN_TUNED_PARAMS, KNNConfig, KNNModel
from dt.fable_dt import GROUP_TUNED_PARAMS as DT_TUNED_PARAMS, DTConfig, DTModel
from mlp.fable_mlp import GROUP_TUNED_PARAMS as MLP_TUNED_PARAMS, MLPConfig, MLPModel
from svm.fable_svm import GROUP_TUNED_PARAMS as SVM_TUNED_PARAMS, SVMConfig, SVMModel


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


# XGBoost champion: focal g=1 from the top-ranked grouped benchmark row in Fable/eval_outputs/comparison_table.md.
XGB_CHAMPION = FableConfig(
    feature_set="all",
    paper_weight="none",
    class_weight="none",
    objective="focal",
    focal_gamma=1.0,
    params=dict(XGB_DEFAULT_PARAMS),
)

# KNN / DT / MLP / SVM champions are the group-tuned configs selected by the per-model search scripts.
KNN_CHAMPION = KNNConfig(feature_set="all", paper_weight="sqrt", class_weight="balanced", params=dict(KNN_TUNED_PARAMS))
DT_CHAMPION = DTConfig(feature_set="all", paper_weight="sqrt", class_weight="balanced", params=dict(DT_TUNED_PARAMS))
MLP_CHAMPION = MLPConfig(feature_set="all", paper_weight="sqrt", class_weight="balanced", params=dict(MLP_TUNED_PARAMS))
SVM_CHAMPION = SVMConfig(feature_set="all", paper_weight="sqrt", class_weight="balanced", params=dict(SVM_TUNED_PARAMS))


def factory_from_cfg(model_cls, cfg):
    return lambda cfg=cfg, model_cls=model_cls: model_cls(cfg)


def champions() -> dict[str, callable]:
    return {
        "xgb": factory_from_cfg(FableModel, XGB_CHAMPION),
        "knn": factory_from_cfg(KNNModel, KNN_CHAMPION),
        "dt": factory_from_cfg(DTModel, DT_CHAMPION),
        "mlp": factory_from_cfg(MLPModel, MLP_CHAMPION),
        "svm": factory_from_cfg(SVMModel, SVM_CHAMPION),
    }


def lopo_five_way(df: pd.DataFrame, factories: dict[str, callable], out_dir: Path) -> pd.DataFrame:
    return lopo_n_way(df, factories, out_dir, prefix="lopo_5way")


def plot_interpolation_5way(folds: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    order = summary.index.tolist()[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 0.48 * len(order) + 1.7))
    for i, mname in enumerate(order):
        sub = folds[folds["method"] == mname]["roc_auc"]
        ax.scatter(sub, np.full(len(sub), i), alpha=0.45, s=14, color="#27598e", zorder=2)
        ax.errorbar(sub.mean(), i, xerr=sub.std(), fmt="o", color="#c23b22", capsize=3, markersize=6, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("held-out-paper ROC-AUC (StratifiedGroupKFold 5x3)")
    ax.set_title("Five-way interpolation comparison\n(blue = individual folds, red = mean +/- sd)")
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_interpolation_5way.png", bbox_inches="tight")
    plt.close(fig)


def plot_lopo_5way(lopo: pd.DataFrame, out_dir: Path) -> None:
    cols = {c[4:]: c for c in lopo.columns if c.startswith("auc_")}
    xgb = cols["xgb"]
    others = [name for name in ["knn", "dt", "mlp", "svm"] if name in cols]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.2))
    axes = axes.ravel()
    for ax, name in zip(axes, others):
        sub = lopo.dropna(subset=[xgb, cols[name]])
        sizes = 10 + 70 * (sub["n_rows"] / sub["n_rows"].max())
        ax.scatter(sub[xgb], sub[cols[name]], s=sizes, alpha=0.65, color="#27598e")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.axhline(0.5, color="gray", ls=":", lw=0.6)
        ax.axvline(0.5, color="gray", ls=":", lw=0.6)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("XGBoost paper AUC")
        ax.set_ylabel(f"{name.upper()} paper AUC")
        ax.set_title(f"XGB vs {name.upper()}\n{int((sub[cols[name]] > sub[xgb]).sum())}/{len(sub)} papers improved")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_lopo_5way.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="Microgravity_Database.xlsm")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "compare_outputs"))
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.quick:
        out_dir = out_dir.with_name(out_dir.name + "_quick")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean_any(args.data)
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"]
    print(
        f"Loaded {len(df)} rows / {papers.nunique()} canonical papers ({int(y.sum())} ignition, {int((1 - y).sum())} no-ignition)"
    )

    factories = champions()
    outer_splits = make_repeated_splits(
        df,
        y,
        papers,
        n_seeds=1 if args.quick else args.n_seeds,
        n_splits=5,
        seed_base=RNG,
    )

    all_folds = []
    for name, factory in factories.items():
        folds = benchmark_on_splits(
            name,
            factory,
            df,
            outer_splits,
            threshold_weighted=getattr(factory(), "threshold_weighted", False),
        )
        all_folds.append(folds)
        print(f"  {name}: group ROC-AUC = {folds['roc_auc'].mean():.4f} +/- {folds['roc_auc'].std():.4f}")

    folds = pd.concat(all_folds, ignore_index=True)
    folds.to_csv(out_dir / "interpolation_5way_folds.csv", index=False)

    summary = summarise_folds(folds)
    summary.to_csv(out_dir / "comparison_table.csv")
    md = to_markdown_table(summary)
    (out_dir / "interpolation_5way_summary.md").write_text(md + "\n")
    print("\n" + md + "\n")

    wilcoxon = pairwise_wilcoxon_table(folds)
    wilcoxon.to_csv(out_dir / "interpolation_5way_wilcoxon.csv", index=False)
    print(wilcoxon.round(4).to_string(index=False))

    lopo = None
    if not args.quick:
        lopo = lopo_five_way(df, factories, out_dir)
        pooled = json.loads((out_dir / "lopo_5way_pooled.json").read_text())
        print(json.dumps(pooled, indent=2))

    plot_interpolation_5way(folds, summary, out_dir)
    if lopo is not None:
        plot_lopo_5way(lopo, out_dir)

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()