from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

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
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fable_common import combined_weights, feature_lists, load_clean
from fable_train import optimal_thresholds as xgb_optimal_thresholds


RNG = 42


def load_clean_any(data_path: str | Path) -> pd.DataFrame:
    path = Path(data_path)
    if path.suffix.lower() != ".xlsm":
        return load_clean(path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = Path(tmpdir) / f"{path.stem}.csv"
        raw = pd.read_excel(path, sheet_name="Sheet1", header=1)
        with open(tmp_csv, "w", encoding="utf-8", newline="") as handle:
            handle.write("# xlsm export\n")
            raw.to_csv(handle, index=False)
        return load_clean(tmp_csv)


@dataclass
class BaseAltConfig:
    feature_set: str = "all"
    paper_weight: str = "sqrt"
    class_weight: str = "balanced"
    random_state: int = RNG

    def label(self) -> str:
        return f"{self.feature_set} pw={self.paper_weight} cw={self.class_weight}"


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
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


def _thresholds_from_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    fpr, tpr, thr = roc_curve(y_true, y_score, sample_weight=sample_weight)
    if sample_weight is None:
        weights = np.ones_like(y_true, dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
    positives = float(weights[y_true == 1].sum())
    negatives = float(weights[y_true == 0].sum())
    tp = tpr * positives
    fp = fpr * negatives
    fn = positives - tp
    tn = negatives - fp

    with np.errstate(divide="ignore", invalid="ignore"):
        mcc_num = tp * tn - fp * fn
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(mcc_den > 0, mcc_num / mcc_den, 0.0)
        f1 = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        bal = 0.5 * (tpr + (1 - fpr))
        youden = tpr - fpr

    valid = slice(1, None)

    def _pick(metric: np.ndarray) -> float:
        idx = int(np.argmax(metric[valid])) + 1
        return float(min(max(thr[idx], 0.0), 1.0))

    return {
        "mcc": _pick(mcc),
        "f1": _pick(f1),
        "balanced_accuracy": _pick(bal),
        "youden_j": _pick(youden),
    }


def optimal_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    if sample_weight is None:
        return xgb_optimal_thresholds(y_true, y_score)
    return _thresholds_from_curve(y_true, y_score, sample_weight=sample_weight)


def make_repeated_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    papers: pd.Series,
    n_seeds: int = 3,
    n_splits: int = 5,
    seed_base: int = RNG,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    splits: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed_base + seed)
        for fold, (tr, te) in enumerate(sgkf.split(X, y, groups=papers)):
            splits.append((seed, fold, tr, te))
    return splits


class BaseAltSklearnModel:
    supports_sample_weight = False
    threshold_weighted = False

    def __init__(self, cfg: BaseAltConfig):
        self.cfg = cfg
        self.numeric, self.categorical = feature_lists(cfg.feature_set)
        self.features = self.numeric + self.categorical
        self.pre = make_preprocessor(self.numeric, self.categorical)
        self.model_ = None

    def _build_estimator(self, y: np.ndarray):
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: np.ndarray, papers: pd.Series):
        Xt = self.pre.fit_transform(X[self.features])
        sample_weight = None
        if self.supports_sample_weight:
            sample_weight = combined_weights(
                y,
                papers,
                paper_strategy=self.cfg.paper_weight,
                use_class=(self.cfg.class_weight == "balanced"),
            )
        estimator = self._build_estimator(y)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        estimator.fit(Xt, y, **fit_kwargs)
        self.model_ = estimator
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model has not been fit yet")
        Xt = self.pre.transform(X[self.features])
        proba = self.model_.predict_proba(Xt)
        if proba.ndim == 2:
            return proba[:, 1]
        return np.asarray(proba)


def group_aware_search(
    base_cfg: BaseAltConfig,
    model_cls,
    param_grid: dict[str, list],
    X: pd.DataFrame,
    y: np.ndarray,
    papers: pd.Series,
    n_iter: int | None = None,
    n_splits: int = 5,
    seed: int = RNG,
    log_path: Path | None = None,
) -> tuple[dict, float, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    candidates = list(ParameterGrid(param_grid))
    if n_iter is not None and n_iter < len(candidates):
        chosen = rng.choice(len(candidates), size=n_iter, replace=False)
        candidates = [candidates[i] for i in chosen]

    splits = list(
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y, groups=papers)
    )
    records = []
    best_score = -np.inf
    best_params: dict = {}

    for idx, params in enumerate(candidates, start=1):
        cfg = type(base_cfg)(**{**base_cfg.__dict__, "params": dict(params)})
        fold_scores = []
        for tr, te in splits:
            model = model_cls(cfg)
            model.fit(X.iloc[tr], y[tr], papers.iloc[tr])
            fold_scores.append(roc_auc_score(y[te], model.predict_proba(X.iloc[te])))
        score = float(np.mean(fold_scores))
        records.append({**params, "group_roc_auc": score, "std": float(np.std(fold_scores))})
        if score > best_score:
            best_score = score
            best_params = dict(params)
            marker = "  <-- new best"
        else:
            marker = ""
        print(f"  [{idx:3d}/{len(candidates)}] group ROC-AUC = {score:.4f}{marker}")

    history = pd.DataFrame(records).sort_values("group_roc_auc", ascending=False)
    if log_path is not None:
        history.to_csv(log_path, index=False)
    return best_params, best_score, history


def inner_thresholds(
    model_factory: Callable[[], BaseAltSklearnModel],
    X: pd.DataFrame,
    y: np.ndarray,
    papers: pd.Series,
    seed: int,
    threshold_weighted: bool = False,
    n_splits: int = 3,
) -> dict[str, float]:
    probe = model_factory()
    inner = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in inner.split(X, y, groups=papers):
        model = model_factory()
        model.fit(X.iloc[tr], y[tr], papers.iloc[tr])
        oof[te] = model.predict_proba(X.iloc[te])
    weights = None
    if threshold_weighted:
        weights = combined_weights(
            y,
            papers,
            paper_strategy=probe.cfg.paper_weight,
            use_class=(probe.cfg.class_weight == "balanced"),
        )
    return optimal_thresholds(y, oof, sample_weight=weights)


def benchmark_on_splits(
    name: str,
    model_factory: Callable[[], BaseAltSklearnModel],
    df: pd.DataFrame,
    outer_splits: list[tuple[int, int, np.ndarray, np.ndarray]],
    threshold_weighted: bool = False,
    inner_n_splits: int = 3,
) -> pd.DataFrame:
    X = df
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"]
    rows = []
    for seed, fold, tr, te in outer_splits:
        Xtr, ytr, ptr = X.iloc[tr], y[tr], papers.iloc[tr]
        Xte, yte = X.iloc[te], y[te]
        thr = inner_thresholds(
            model_factory,
            Xtr,
            ytr,
            ptr,
            seed=RNG + seed,
            threshold_weighted=threshold_weighted,
            n_splits=inner_n_splits,
        )
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
                "balanced_accuracy": balanced_accuracy_score(yte, (p >= thr["balanced_accuracy"]).astype(int)),
                "youden_j_balacc": balanced_accuracy_score(yte, (p >= thr["youden_j"]).astype(int)),
                "thr_mcc": thr["mcc"],
                "thr_f1": thr["f1"],
                "thr_balacc": thr["balanced_accuracy"],
                "thr_youden": thr["youden_j"],
            }
        )
    return pd.DataFrame(rows)


def summarise_folds(folds: pd.DataFrame) -> pd.DataFrame:
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
            f"| {name} | {r['group_roc_auc']:.3f} +/- {r['group_roc_auc_std']:.3f} | {r['pr_auc']:.3f} +/- {r['pr_auc_std']:.3f} | {r['mcc']:.3f} +/- {r['mcc_std']:.3f} | {r['balanced_accuracy']:.3f} +/- {r['balanced_accuracy_std']:.3f} |"
        )
    return "\n".join(lines)


def pairwise_wilcoxon_table(folds: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    methods = sorted(folds["method"].unique())
    rows = []
    for a, b in combinations(methods, 2):
        fa = folds[folds["method"] == a].set_index(["seed", "fold"])
        fb = folds[folds["method"] == b].set_index(["seed", "fold"])
        common = fa.index.intersection(fb.index)
        if len(common) == 0:
            continue
        d_auc = (fb.loc[common, "roc_auc"] - fa.loc[common, "roc_auc"]).to_numpy()
        d_mcc = (fb.loc[common, "mcc"] - fa.loc[common, "mcc"]).to_numpy()
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "delta_roc_auc": float(np.mean(d_auc)),
                "p_roc_auc": float(wilcoxon(d_auc).pvalue) if np.any(d_auc != 0) else 1.0,
                "delta_mcc": float(np.mean(d_mcc)),
                "p_mcc": float(wilcoxon(d_mcc).pvalue) if np.any(d_mcc != 0) else 1.0,
            }
        )
    return pd.DataFrame(rows)


def lopo_two_way(
    df: pd.DataFrame,
    model_factory_a: Callable[[], BaseAltSklearnModel],
    model_factory_b: Callable[[], BaseAltSklearnModel],
    names: tuple[str, str],
    out_dir: Path,
    csv_name: str = "lopo_baseline_vs_best.csv",
    json_name: str = "lopo_pooled.json",
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
        for factory, nm in zip((model_factory_a, model_factory_b), names):
            model = factory()
            model.fit(X.iloc[~mask], y[~mask], papers[~mask])
            p = model.predict_proba(X.iloc[mask])
            oof[nm][mask] = p
            rec[f"auc_{nm}"] = roc_auc_score(y[mask], p) if len(np.unique(y[mask])) == 2 else np.nan
        rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / csv_name, index=False)
    pooled = {nm: float(roc_auc_score(y, oof[nm])) for nm in names}
    with open(out_dir / json_name, "w") as handle:
        json.dump(pooled, handle, indent=2)
    return res


def lopo_n_way(
    df: pd.DataFrame,
    factories: dict[str, Callable[[], BaseAltSklearnModel]],
    out_dir: Path,
    prefix: str = "lopo_5way",
) -> pd.DataFrame:
    X = df
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"]
    rows = []
    oof = {name: np.full(len(y), np.nan) for name in factories}
    for pid in papers.unique():
        mask = (papers == pid).to_numpy()
        rec = {
            "paper_id": pid,
            "paper_label": df.loc[mask, "paper_label"].iloc[0],
            "n_rows": int(mask.sum()),
            "ignition_rate": float(y[mask].mean()),
        }
        for name, factory in factories.items():
            model = factory()
            model.fit(X.iloc[~mask], y[~mask], papers[~mask])
            p = model.predict_proba(X.iloc[mask])
            oof[name][mask] = p
            rec[f"auc_{name}"] = roc_auc_score(y[mask], p) if len(np.unique(y[mask])) == 2 else np.nan
        rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / f"{prefix}.csv", index=False)
    pooled = {name: float(roc_auc_score(y, oof[name])) for name in factories}
    with open(out_dir / f"{prefix}_pooled.json", "w") as handle:
        json.dump(pooled, handle, indent=2)
    return res