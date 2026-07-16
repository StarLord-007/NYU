"""
fable_models.py
===============
Model-agnostic core of the Fable benchmarking platform.

This module generalises the XGBoost reference architecture (fable_train.py /
fable_eval.py) so that *any* classifier exposing the FableModel public API

    .fit(X, y, papers)
    .predict_proba(X)          # -> 1-D P(ignition)

can be trained, tuned, benchmarked and compared under *identical*
experimental conditions:

* identical canonical-paper grouping (fable_common),
* identical outer StratifiedGroupKFold splits (interpolation protocol),
* identical Leave-One-Paper-Out splits (extrapolation protocol),
* identical nested (inner-fold-only) threshold optimisation,
* identical metrics.

Design
------
* ``SkFableModel``   — FableModel-compatible wrapper around a sklearn
                       estimator, with the mirrored preprocessing pipeline
                       and the shared paper x class weighting policy.
* ``ModelSpec``      — declarative description of one algorithm (estimator
                       factory, search space, champion hyperparameters).
                       Each model script defines exactly one ModelSpec.
* ``group_aware_search`` — generic version of fable_train's random search.
* ``benchmark_interpolation`` / ``lopo_evaluate`` — factory-based versions
  of the evaluation loops in fable_eval.py.
* ``run_model_cli``  — the complete per-model entry point (train / tune /
                       benchmark / evaluate / plots / model card), shared by
                       all model scripts.

Sample-weighting policy (documented difference)
-----------------------------------------------
XGBoost, Decision Tree and SVM accept ``sample_weight`` in ``fit``: they
receive the combined paper x class weights (fable_common.combined_weights)
during fitting, and thresholds are then tuned unweighted on inner-fold OOF
predictions — exactly the existing XGBoost pipeline.

KNN and MLP (sklearn) cannot take sample weights during fitting. For those,
the *same* combined weights are instead applied during threshold
optimisation (weighted ROC operating points). This is the closest weighted
equivalent available and is recorded in each model card so the asymmetry is
explicit. No model ever receives information unavailable to the others.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fable_common import combined_weights, feature_lists, load_clean
from fable_train import (
    GROUP_TUNED_PARAMS,
    FableConfig,
    FableModel,
    grouped_oof_predictions,
    optimal_thresholds,
)

RNG = 42
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "Microgravity_Database_reduced.csv"

# Preserve the existing plotting style (fable_eval.py).
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
# Data loading (xlsm support without touching fable_common.py)
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the database through fable_common.load_clean.

    The default input is Microgravity_Database_reduced.csv. Legacy Excel inputs
    are still transparently converted to a sidecar
    CSV that preserves the original layout (header row skipped by
    load_clean itself), then routed through the canonical loader so every
    model sees byte-identical cleaned data.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsm", ".xlsx", ".xls"}:
        raw = pd.read_excel(path, header=None)
        sidecar = path.with_suffix(path.suffix + ".converted.csv")
        raw.to_csv(sidecar, index=False, header=False, encoding="utf-8")
        return load_clean(sidecar)
    return load_clean(path)


# ---------------------------------------------------------------------------
# Preprocessing (mirrors the XGBoost pipeline; adds imputation + scaling)
# ---------------------------------------------------------------------------

def make_sk_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Numeric: median imputation -> StandardScaler.
    Categorical: constant 'Unknown' imputation -> One-Hot (as in fable_train).
    """
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# FableModel-compatible sklearn wrapper
# ---------------------------------------------------------------------------

class SkFableModel:
    """Preprocessor + sklearn estimator with the exact FableModel public API.

    Deliberately *not* an sklearn estimator (mirrors FableModel's rationale):
    weights must be recomputed inside every CV fold from the fold's own
    (y, papers), which this class keeps explicit and auditable.
    """

    def __init__(
        self,
        estimator_factory: Callable[[dict, int], Any],
        params: dict | None = None,
        feature_set: str = "all",
        paper_weight: str = "sqrt",
        class_weight: str = "balanced",
        supports_sample_weight: bool = False,
        random_state: int = RNG,
    ):
        self.estimator_factory = estimator_factory
        self.params = dict(params or {})
        self.feature_set = feature_set
        self.paper_weight = paper_weight
        self.class_weight = class_weight
        self.supports_sample_weight = supports_sample_weight
        self.random_state = random_state
        self.numeric, self.categorical = feature_lists(feature_set)
        self.features = self.numeric + self.categorical
        self.pre = make_sk_preprocessor(self.numeric, self.categorical)
        self.model_: Any = None

    # shared weighting policy (same call as the XGBoost path)
    def sample_weight(self, y: np.ndarray, papers: pd.Series) -> np.ndarray | None:
        if self.paper_weight == "none" and self.class_weight != "balanced":
            return None
        return combined_weights(
            y, papers,
            paper_strategy=self.paper_weight,
            use_class=(self.class_weight == "balanced"),
        )

    # -- public API (identical to FableModel) -------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray, papers: pd.Series) -> "SkFableModel":
        Xt = self.pre.fit_transform(X[self.features]).astype(np.float32)
        self.model_ = self.estimator_factory(self.params, self.random_state)
        if self.supports_sample_weight:
            self.model_.fit(Xt, y, sample_weight=self.sample_weight(y, papers))
        else:
            # KNN / MLP: no fit-time weights available; the same combined
            # weights are applied at threshold optimisation instead
            # (see module docstring).
            self.model_.fit(Xt, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.pre.transform(X[self.features]).astype(np.float32)
        return self.model_.predict_proba(Xt)[:, 1]


# ---------------------------------------------------------------------------
# Model specification (one per algorithm)
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    key: str                                          # short id: "knn", "svm", ...
    display: str                                      # pretty name for reports
    estimator_factory: Callable[[dict, int], Any]     # (params, seed) -> estimator
    param_space: dict[str, list]                      # random-search space
    champion_params: dict                             # recommended configuration
    supports_sample_weight: bool
    feature_set: str = "all"
    paper_weight: str = "sqrt"
    class_weight: str = "balanced"
    notes: str = ""

    def make_model(self, params: dict | None = None, random_state: int = RNG) -> SkFableModel:
        return SkFableModel(
            self.estimator_factory,
            params if params is not None else self.champion_params,
            feature_set=self.feature_set,
            paper_weight=self.paper_weight,
            class_weight=self.class_weight,
            supports_sample_weight=self.supports_sample_weight,
            random_state=random_state,
        )

    def champion_factory(self, random_state: int = RNG) -> Callable[[], SkFableModel]:
        return lambda: self.make_model(random_state=random_state)


def spec_weight_fn(spec: ModelSpec) -> Callable[[np.ndarray, pd.Series], np.ndarray] | None:
    """Threshold-stage weight function.

    None for estimators that already consumed the weights in fit()
    (thresholds then tuned unweighted, exactly like the XGBoost pipeline);
    otherwise the identical combined paper x class weights.
    """
    if spec.supports_sample_weight:
        return None
    return lambda y, papers: combined_weights(
        y, papers, paper_strategy=spec.paper_weight,
        use_class=(spec.class_weight == "balanced"),
    )


def xgb_champion_factory(random_state: int = RNG) -> Callable[[], FableModel]:
    """The existing XGBoost champion (recommended fable_train defaults):
    combined 1/sqrt(N_paper) x class weighting + group-tuned hyperparameters,
    on the same 'all' feature set as the sklearn models."""
    cfg = FableConfig(
        feature_set="all",
        paper_weight="sqrt",
        class_weight="balanced",
        params=dict(GROUP_TUNED_PARAMS),
        random_state=random_state,
    )
    return lambda: FableModel(cfg)


# ---------------------------------------------------------------------------
# Threshold optimisation (weighted extension of fable_train.optimal_thresholds)
# ---------------------------------------------------------------------------

def optimal_thresholds_weighted(
    y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray | None = None
) -> dict[str, float]:
    """Identical operating-point selection to fable_train.optimal_thresholds
    (MCC / F1 / balanced accuracy / Youden J on validation predictions only),
    optionally weighted. With sample_weight=None it delegates verbatim to the
    reference implementation, so all models share one strategy."""
    if sample_weight is None:
        return optimal_thresholds(y_true, y_score)

    sw = np.asarray(sample_weight, dtype=float)
    fpr, tpr, thr = roc_curve(y_true, y_score, sample_weight=sw)
    P = float(sw[y_true == 1].sum())
    N = float(sw[y_true == 0].sum())
    tp = tpr * P
    fp = fpr * N
    fn = P - tp
    tn = N - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        mcc_num = tp * tn - fp * fn
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(mcc_den > 0, mcc_num / mcc_den, 0.0)
        f1 = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        bal = 0.5 * (tpr + (1 - fpr))
        youden = tpr - fpr

    valid = slice(1, None)  # roc_curve prepends an artificial threshold

    def _pick(metric: np.ndarray) -> float:
        idx = int(np.argmax(metric[valid])) + 1
        return min(max(float(thr[idx]), 0.0), 1.0)

    return {
        "mcc": _pick(mcc),
        "f1": _pick(f1),
        "balanced_accuracy": _pick(bal),
        "youden_j": _pick(youden),
    }


# ---------------------------------------------------------------------------
# Generic group-aware hyperparameter search (same philosophy as fable_train)
# ---------------------------------------------------------------------------

def group_aware_search(
    make_model: Callable[[dict], Any],
    param_space: dict[str, list],
    X: pd.DataFrame,
    y: np.ndarray,
    papers: pd.Series,
    n_iter: int = 40,
    n_splits: int = 5,
    seed: int = RNG,
    log_path: Path | None = None,
) -> tuple[dict, float, pd.DataFrame]:
    """Random search scored by held-out-paper ROC-AUC (StratifiedGroupKFold),
    weights recomputed inside every training fold via the model itself.
    Generic replication of fable_train.group_aware_search for arbitrary
    model factories."""
    rng = np.random.default_rng(seed)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(sgkf.split(X, y, groups=papers))
    records, best_score, best_params = [], -np.inf, None
    for it in range(n_iter):
        params = {k: v[int(rng.integers(len(v)))] for k, v in param_space.items()}
        fold_scores = []
        for tr, te in splits:
            m = make_model(params)
            m.fit(X.iloc[tr], y[tr], papers.iloc[tr])
            fold_scores.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])))
        score = float(np.mean(fold_scores))
        records.append(
            {**{k: str(v) for k, v in params.items()},
             "group_roc_auc": score, "std": float(np.std(fold_scores))}
        )
        marker = ""
        if score > best_score:
            best_score, best_params = score, params
            marker = "  <-- new best"
        print(f"  [{it + 1:3d}/{n_iter}] group ROC-AUC = {score:.4f}{marker}")
    history = pd.DataFrame(records).sort_values("group_roc_auc", ascending=False)
    if log_path is not None:
        history.to_csv(log_path, index=False)
    return best_params, best_score, history


# ---------------------------------------------------------------------------
# Model-agnostic evaluation loops (factory-based versions of fable_eval's)
# ---------------------------------------------------------------------------

def make_outer_splits(
    df: pd.DataFrame, n_seeds: int = 3, n_splits: int = 5, base_seed: int = RNG
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Precomputed outer StratifiedGroupKFold splits. Computing them once and
    passing the same list to every model *guarantees* identical partitions."""
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"]
    splits = []
    for seed in range(n_seeds):
        outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=base_seed + seed)
        for fold, (tr, te) in enumerate(outer.split(df, y, groups=papers)):
            splits.append((seed, fold, tr, te))
    return splits


def inner_thresholds(
    model_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: np.ndarray,
    papers: pd.Series,
    seed: int,
    weight_fn: Callable | None = None,
) -> dict[str, float]:
    """Thresholds tuned on inner held-out-paper folds of the training data
    only (requirement C, nested exactly as in fable_eval)."""
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in inner.split(X, y, groups=papers):
        m = model_factory()
        m.fit(X.iloc[tr], y[tr], papers.iloc[tr])
        oof[te] = m.predict_proba(X.iloc[te])
    w = weight_fn(y, papers) if weight_fn is not None else None
    return optimal_thresholds_weighted(y, oof, w)


def benchmark_interpolation(
    name: str,
    model_factory: Callable[[], Any],
    df: pd.DataFrame,
    splits: list[tuple[int, int, np.ndarray, np.ndarray]],
    weight_fn: Callable | None = None,
    base_seed: int = RNG,
) -> pd.DataFrame:
    """Repeated grouped interpolation with nested threshold tuning
    (row schema identical to fable_eval.benchmark_method)."""
    X = df
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"]
    rows = []
    for seed, fold, tr, te in splits:
        Xtr, ytr, ptr = X.iloc[tr], y[tr], papers.iloc[tr]
        Xte, yte = X.iloc[te], y[te]
        thr = inner_thresholds(model_factory, Xtr, ytr, ptr, seed=base_seed + seed,
                               weight_fn=weight_fn)
        model = model_factory()
        model.fit(Xtr, ytr, ptr)
        p = model.predict_proba(Xte)
        rows.append(
            {
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
        )
    return pd.DataFrame(rows)


def lopo_evaluate(
    name: str,
    model_factory: Callable[[], Any],
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Leave-One-Paper-Out extrapolation: per-paper AUC + pooled metrics."""
    X = df
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"]
    oof = np.full(len(y), np.nan)
    rows = []
    for pid in papers.unique():
        mask = (papers == pid).to_numpy()
        m = model_factory()
        m.fit(X.iloc[~mask], y[~mask], papers[~mask])
        p = m.predict_proba(X.iloc[mask])
        oof[mask] = p
        rows.append(
            {
                "method": name,
                "paper_id": pid,
                "paper_label": df.loc[mask, "paper_label"].iloc[0],
                "n_rows": int(mask.sum()),
                "ignition_rate": float(y[mask].mean()),
                "auc": roc_auc_score(y[mask], p) if len(np.unique(y[mask])) == 2 else np.nan,
            }
        )
    pooled = {
        "pooled_roc_auc": float(roc_auc_score(y, oof)),
        "pooled_pr_auc": float(average_precision_score(y, oof)),
    }
    return pd.DataFrame(rows), pooled, oof


def summarise_folds(folds: pd.DataFrame) -> pd.DataFrame:
    """Identical aggregation to fable_eval.summarise."""
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
        "| Model | Group ROC-AUC | PR-AUC | MCC | Balanced Accuracy |",
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


# ---------------------------------------------------------------------------
# Per-model plots (same visual language as fable_eval)
# ---------------------------------------------------------------------------

def plot_fold_distribution(folds: pd.DataFrame, title: str, path: Path) -> None:
    order = folds["method"].unique().tolist()[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.6))
    for i, mname in enumerate(order):
        sub = folds[folds["method"] == mname]["roc_auc"]
        ax.scatter(sub, np.full(len(sub), i), alpha=0.45, s=14, color="#27598e", zorder=2)
        ax.errorbar(sub.mean(), i, xerr=sub.std(), fmt="o", color="#c23b22",
                    capsize=3, markersize=6, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("held-out-paper ROC-AUC (StratifiedGroupKFold)")
    ax.set_title(title + "\n(blue = individual folds, red = mean +/- sd)")
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_lopo_per_paper(lopo: pd.DataFrame, title: str, path: Path) -> None:
    two = lopo.dropna(subset=["auc"]).sort_values("auc")
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(two) + 1.6))
    ax.barh(np.arange(len(two)), two["auc"], color="#27598e")
    ax.set_yticks(np.arange(len(two)))
    ax.set_yticklabels(two["paper_label"], fontsize=6)
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xlabel("within-paper AUC (Leave-One-Paper-Out)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared per-model entry point (train / tune / benchmark / evaluate / save)
# ---------------------------------------------------------------------------

def run_model_cli(spec: ModelSpec) -> None:
    ap = argparse.ArgumentParser(description=f"Fable pipeline for {spec.display}")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / f"{spec.key}_outputs"))
    ap.add_argument("--tune", action="store_true", help="Run group-aware random search.")
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--skip-lopo", action="store_true")
    ap.add_argument("--quick", action="store_true", help="1 seed, no LOPO (smoke test)")
    ap.add_argument("--random-state", type=int, default=RNG)
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.quick:
        out_dir = out_dir.with_name(out_dir.name + "_quick")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading {args.data} ...")
    df = load_dataset(args.data)
    X = df
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"]
    print(f"  {len(df)} rows, {papers.nunique()} canonical papers")

    params = dict(spec.champion_params)
    if args.tune:
        print(f"[2/6] Group-aware random search for {spec.display} ({args.n_iter} candidates) ...")
        params, best, _ = group_aware_search(
            lambda p: spec.make_model(p, random_state=args.random_state),
            spec.param_space, X, y, papers,
            n_iter=args.n_iter, seed=args.random_state,
            log_path=out_dir / f"{spec.key}_search_history.csv",
        )
        print(f"  best held-out-paper ROC-AUC: {best:.4f}\n  best params: {params}")
    else:
        print(f"[2/6] Using champion configuration: {params}")

    factory = lambda: spec.make_model(params, random_state=args.random_state)
    weight_fn = spec_weight_fn(spec)

    print("[3/6] Validation-fold threshold optimisation (held-out papers) ...")
    oof = grouped_oof_predictions(factory, X, y, papers, n_splits=4, seed=args.random_state)
    thresholds = optimal_thresholds_weighted(
        y, oof, weight_fn(y, papers) if weight_fn is not None else None
    )
    oof_metrics = {
        "oof_group_roc_auc": float(roc_auc_score(y, oof)),
        "oof_group_pr_auc": float(average_precision_score(y, oof)),
    }
    for tname, t in thresholds.items():
        pred = (oof >= t).astype(int)
        oof_metrics[f"oof_{tname}_threshold"] = t
        oof_metrics[f"oof_mcc_at_{tname}"] = float(matthews_corrcoef(y, pred))
        oof_metrics[f"oof_f1_at_{tname}"] = float(f1_score(y, pred))
        oof_metrics[f"oof_balacc_at_{tname}"] = float(balanced_accuracy_score(y, pred))
    print(f"  OOF grouped ROC-AUC: {oof_metrics['oof_group_roc_auc']:.4f}")

    print("[4/6] Interpolation benchmark (repeated StratifiedGroupKFold) ...")
    n_seeds = 1 if args.quick else args.n_seeds
    splits = make_outer_splits(df, n_seeds=n_seeds, base_seed=args.random_state)
    folds = benchmark_interpolation(spec.display, factory, df, splits, weight_fn,
                                    base_seed=args.random_state)
    folds.to_csv(out_dir / f"{spec.key}_benchmark_folds.csv", index=False)
    summary = summarise_folds(folds)
    summary.to_csv(out_dir / f"{spec.key}_benchmark_summary.csv")
    print(f"  group ROC-AUC = {folds['roc_auc'].mean():.4f} +/- {folds['roc_auc'].std():.4f}")
    plot_fold_distribution(folds, f"{spec.display}: unseen-paper discrimination",
                           out_dir / f"fig_{spec.key}_interpolation.png")

    lopo_pooled = None
    if not (args.quick or args.skip_lopo):
        print("[5/6] LOPO extrapolation benchmark ...")
        lopo, lopo_pooled, _ = lopo_evaluate(spec.display, factory, df)
        lopo.to_csv(out_dir / f"{spec.key}_lopo_per_paper.csv", index=False)
        print(f"  pooled LOPO ROC-AUC = {lopo_pooled['pooled_roc_auc']:.4f}")
        plot_lopo_per_paper(lopo, f"{spec.display}: LOPO per-paper AUC",
                            out_dir / f"fig_{spec.key}_lopo.png")
    else:
        print("[5/6] Skipping LOPO.")

    print("[6/6] Fitting final model on all data and saving ...")
    final = factory()
    final.fit(X, y, papers)
    joblib.dump(final, out_dir / f"{spec.key}_model.joblib")

    card = {
        "model": spec.display,
        "key": spec.key,
        "params": {k: str(v) for k, v in params.items()},
        "feature_set": spec.feature_set,
        "paper_weight": spec.paper_weight,
        "class_weight": spec.class_weight,
        "supports_sample_weight_in_fit": spec.supports_sample_weight,
        "weighting_note": (
            "combined paper x class weights applied in fit()"
            if spec.supports_sample_weight
            else "estimator cannot take fit-time weights; identical combined "
                 "weights applied during threshold optimisation instead"
        ),
        "n_rows": int(len(df)),
        "n_papers": int(papers.nunique()),
        "thresholds_from_validation_folds": thresholds,
        **oof_metrics,
        "interpolation_group_roc_auc_mean": float(folds["roc_auc"].mean()),
        "interpolation_group_roc_auc_std": float(folds["roc_auc"].std()),
        "lopo_pooled": lopo_pooled,
        "notes": spec.notes,
    }
    with open(out_dir / f"{spec.key}_model_card.json", "w") as f:
        json.dump(card, f, indent=2)
    print(f"  Done. Outputs in {out_dir}")