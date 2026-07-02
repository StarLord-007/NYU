from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alt_model_support import (
    BaseAltConfig,
    BaseAltSklearnModel,
    benchmark_on_splits,
    group_aware_search,
    load_clean_any,
    lopo_two_way,
    make_repeated_splits,
    pairwise_wilcoxon_table,
    summarise_folds,
    to_markdown_table,
)


RNG = 42
warnings.filterwarnings("ignore", category=ConvergenceWarning)
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


DEFAULT_PARAMS = dict(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    alpha=1e-3,
)

GROUP_TUNED_PARAMS = dict(
    hidden_layer_sizes=(100, 100),
    activation="tanh",
    alpha=1e-4,
)

MLP_PARAM_SPACE = {
    "hidden_layer_sizes": [(50,), (100,), (100, 50), (200, 100), (100, 100)],
    "activation": ["relu", "tanh"],
    "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
}


@dataclass
class MLPConfig(BaseAltConfig):
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))


class MLPModel(BaseAltSklearnModel):
    threshold_weighted = True

    def _build_estimator(self, y: np.ndarray):
        params = dict(self.cfg.params)
        return MLPClassifier(
            random_state=self.cfg.random_state,
            max_iter=800,
            solver="adam",
            learning_rate_init=0.001,
            n_iter_no_change=25,
            tol=1e-4,
            **params,
        )


def factory_from_cfg(cfg: MLPConfig):
    return lambda cfg=cfg: MLPModel(cfg)


def build_registry(base_cfg: MLPConfig, tuned_params: dict, random_state: int) -> dict[str, MLPConfig]:
    tuned_cfg = MLPConfig(
        feature_set=base_cfg.feature_set,
        paper_weight=base_cfg.paper_weight,
        class_weight=base_cfg.class_weight,
        random_state=random_state,
        params=dict(tuned_params),
    )
    return {"baseline": base_cfg, "group-tuned": tuned_cfg}


def plot_figures(folds: pd.DataFrame, summary: pd.DataFrame, lopo: pd.DataFrame | None, out_dir: Path) -> None:
    order = summary.index.tolist()[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.6))
    for i, mname in enumerate(order):
        sub = folds[folds["method"] == mname]["roc_auc"]
        ax.scatter(sub, np.full(len(sub), i), alpha=0.45, s=14, color="#27598e", zorder=2)
        ax.errorbar(sub.mean(), i, xerr=sub.std(), fmt="o", color="#c23b22", capsize=3, markersize=6, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("held-out-paper ROC-AUC (StratifiedGroupKFold 5x3)")
    ax.set_title("MLP unseen-paper discrimination\n(blue = individual folds, red = mean +/- sd)")
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_method_comparison.png", bbox_inches="tight")
    plt.close(fig)

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
        ax.set_xlabel("within-paper AUC: baseline")
        ax.set_ylabel("within-paper AUC: group-tuned")
        ax.set_title("MLP LOPO baseline vs group-tuned\n(marker size = paper size)")
        plt.tight_layout()
        plt.savefig(out_dir / "fig_lopo_scatter.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="Microgravity_Database.xlsm")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "mlp_outputs"))
    ap.add_argument("--feature-set", choices=["all", "physics"], default="all")
    ap.add_argument("--paper-weight", choices=["none", "inverse", "sqrt", "effective", "log"], default="sqrt")
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--n-iter", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=RNG)
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

    base_cfg = MLPConfig(
        feature_set=args.feature_set,
        paper_weight=args.paper_weight,
        class_weight=args.class_weight,
        params=dict(DEFAULT_PARAMS),
        random_state=args.random_state,
    )
    tuned_params = dict(GROUP_TUNED_PARAMS)
    if args.tune:
        print("Group-aware MLP search ...")
        best_params, best_score, history = group_aware_search(
            base_cfg,
            MLPModel,
            MLP_PARAM_SPACE,
            df,
            y,
            papers,
            n_iter=args.n_iter or None,
            seed=args.random_state,
            log_path=out_dir / "group_search_history.csv",
        )
        tuned_params = best_params
        print(f"  best group ROC-AUC = {best_score:.4f}")
        print(f"  best params = {best_params}")
    registry = build_registry(base_cfg, tuned_params, args.random_state)

    outer_splits = make_repeated_splits(
        df,
        y,
        papers,
        n_seeds=1 if args.quick else args.n_seeds,
        n_splits=5,
        seed_base=args.random_state,
    )
    all_folds = []
    for name, cfg in registry.items():
        folds = benchmark_on_splits(
            name,
            factory_from_cfg(cfg),
            df,
            outer_splits,
            threshold_weighted=MLPModel.threshold_weighted,
        )
        all_folds.append(folds)
        print(f"  {name}: group ROC-AUC = {folds['roc_auc'].mean():.4f} +/- {folds['roc_auc'].std():.4f}")
    folds = pd.concat(all_folds, ignore_index=True)
    folds.to_csv(out_dir / "benchmark_folds.csv", index=False)

    summary = summarise_folds(folds)
    summary.to_csv(out_dir / "comparison_table.csv")
    md = to_markdown_table(summary)
    (out_dir / "comparison_table.md").write_text(md + "\n")
    print("\n" + md + "\n")

    paired = pairwise_wilcoxon_table(folds)
    paired.to_csv(out_dir / "pairwise_wilcoxon.csv", index=False)
    print(paired.round(4).to_string(index=False))

    lopo = None
    if not args.quick:
        lopo = lopo_two_way(
            df,
            factory_from_cfg(registry["baseline"]),
            factory_from_cfg(registry["group-tuned"]),
            ("baseline", "group-tuned"),
            out_dir,
        )
        two = lopo.dropna(subset=["auc_baseline", "auc_group-tuned"])
        print(
            f"LOPO pooled baseline -> group-tuned: {float(two['auc_baseline'].mean()):.3f} -> {float(two['auc_group-tuned'].mean()):.3f}"
        )

    plot_figures(folds, summary, lopo, out_dir)

    final = MLPModel(registry["group-tuned"])
    final.fit(df, y, papers)
    joblib.dump(final, out_dir / "mlp_model.joblib")
    with open(out_dir / "mlp_model_card.json", "w") as handle:
        json.dump(
            {
                "config": {**registry["group-tuned"].__dict__, "params": registry["group-tuned"].params},
                "n_rows": int(len(df)),
                "n_papers": int(papers.nunique()),
                "benchmark_summary": summary.to_dict(orient="index"),
            },
            handle,
            indent=2,
        )
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()