"""
fable_train.py
==============
Training script for the ignition / no-ignition classifier, rebuilt around
**unseen-paper generalisation** instead of random-split metrics.

Differences vs ``xgb_ignition_model_2.py``
------------------------------------------
1. **Canonical paper identity.** Groups are physical papers (normalised DOI,
   falling back to a normalised citation string), not raw ``DOI.fillna(
   Article)`` strings. See ``fable_common.canonical_doi``.
2. **Paper-balanced sample weighting** (requirement A). Strategies::

       none | inverse (1/N_p) | sqrt (1/sqrt(N_p)) | effective (Cui 2019) | log

3. **Combined class x paper weighting** (requirement B):
   ``w = paper_weight * class_weight``, replacing ``scale_pos_weight``.
4. **Threshold optimisation on validation folds only** (requirement C).
   Decision thresholds for MCC / F1 / balanced accuracy / Youden J are
   selected on *out-of-fold predictions of held-out papers* inside the
   training data — never on a test set.
5. **Focal loss** (requirement D): XGBoost-compatible custom objective
   (Wang et al. 2019 formulation), benchmarked against binary logistic.
6. **Group-aware hyperparameter optimisation** (requirement E): random
   search scored with StratifiedGroupKFold(5) pooled ROC-AUC on held-out
   papers, with sample weights recomputed inside every training fold.
7. **Domain-generalisation options** (requirement F):
   * ``--feature-set physics`` removes apparatus/campaign descriptors
     (chamber dimensions, facility, ignition hardware settings) that the
     audit showed act as paper fingerprints;
   * ``--monotone-o2`` enforces a physically required monotone increasing
     ignition-probability response to oxygen fraction;
   * ``--paper-bagging N`` fits an ensemble over cluster (paper-level)
     bootstrap resamples, the standard bootstrap for clustered data.

Run (final recommended configuration = the defaults)::

    python fable_train.py --data ../Microgravity_Database_reduced.csv

Group-aware hyperparameter search::

    python fable_train.py --tune --n-iter 40 ...

Artifacts -> ``<this script dir>/model_outputs/``
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from fable_common import combined_weights, feature_lists, load_clean

RNG_DEFAULT = 42
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "Microgravity_Database_reduced.csv"


# ---------------------------------------------------------------------------
# Focal loss (XGBoost custom objective)
# ---------------------------------------------------------------------------

def _robust_pow(base: np.ndarray, p: float) -> np.ndarray:
    return np.sign(base) * np.power(np.abs(base), p)


def make_focal_objective(gamma: float):
    """Binary focal loss objective for XGBoost (sklearn API).

    Closed-form gradient/Hessian following Wang, Chen & et al., "Imbalance-
    XGBoost" (Pattern Recognition Letters 2020). For ``gamma = 0`` this
    reduces to standard binary logistic loss. Class/paper re-balancing is
    handled through ``sample_weight`` (applied to grad/hess below, matching
    XGBoost's behaviour for built-in objectives), so no alpha term is needed.
    """

    def focal_obj(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None):
        y = y_true.astype(float)
        p = 1.0 / (1.0 + np.exp(-y_pred))
        sign = np.power(-1.0, y)  # +1 for y=0, -1 for y=1
        g1 = p * (1.0 - p)
        g2 = y + sign * p          # = p for y=0, 1-p for y=1
        g3 = p + y - 1.0           # = p-1 for y=0, p for y=1
        g4 = 1.0 - y - sign * p    # = 1-p for y=0, p for y=1
        g5 = g2
        grad = (
            gamma * g3 * _robust_pow(g2, gamma) * np.log(np.clip(g4, 1e-9, None))
            + sign * _robust_pow(g5, gamma + 1.0)
        )
        hess_1 = _robust_pow(g2, gamma) + gamma * sign * g3 * _robust_pow(g2, gamma - 1.0)
        hess_2 = sign * g3 * _robust_pow(g2, gamma) / np.clip(g4, 1e-9, None)
        hess = (
            (hess_1 * np.log(np.clip(g4, 1e-9, None)) - hess_2) * gamma
            + (gamma + 1.0) * _robust_pow(g5, gamma)
        ) * g1
        hess = np.maximum(hess, 1e-16)
        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
        return grad, hess

    return focal_obj


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1.0,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.0,
)

# Winner of the group-aware random search (60 candidates scored by
# StratifiedGroupKFold(5) held-out-paper ROC-AUC; full history in
# <this script dir>/model_outputs/group_search_history.csv). NOTE: the search showed *low*
# sensitivity — the top-10 configurations sit within one fold-std of each
# other (0.636-0.646) — i.e. hyperparameters are not the binding constraint
# on cross-paper transfer; domain shift is. We keep the winner anyway:
# moderately smaller capacity (400 x depth-5, subsample 0.7, lambda 2) than
# the random-split optimum.
GROUP_TUNED_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=4.0,
    subsample=0.7,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    reg_alpha=0.0,
    gamma=0.0,
)


@dataclass
class FableConfig:
    feature_set: str = "all"             # all | physics
    paper_weight: str = "none"           # none | inverse | sqrt | effective | log
    class_weight: str = "spw"            # none | balanced | spw (legacy scale_pos_weight)
    objective: str = "logistic"          # logistic | focal
    focal_gamma: float = 2.0
    monotone_o2: bool = False
    paper_bagging: int = 0               # 0 = off, else number of cluster-bootstrap models
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    random_state: int = RNG_DEFAULT

    def label(self) -> str:
        bits = [self.feature_set, f"pw={self.paper_weight}", f"cw={self.class_weight}"]
        if self.objective == "focal":
            bits.append(f"focal(g={self.focal_gamma})")
        if self.monotone_o2:
            bits.append("monoO2")
        if self.paper_bagging:
            bits.append(f"bag{self.paper_bagging}")
        return " ".join(bits)


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", "passthrough", numeric),
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


class FableModel:
    """Preprocessor + (optionally bagged) XGBoost with weighting strategies.

    Not an sklearn estimator on purpose: weights and monotone constraints
    must be recomputed *inside* every CV fold, which sklearn pipelines make
    awkward; this class keeps that logic explicit and auditable.
    """

    def __init__(self, cfg: FableConfig):
        self.cfg = cfg
        self.numeric, self.categorical = feature_lists(cfg.feature_set)
        self.features = self.numeric + self.categorical
        self.pre = make_preprocessor(self.numeric, self.categorical)
        self.models_: list[XGBClassifier] = []

    # -- internals ---------------------------------------------------------
    def _sample_weight(self, y: np.ndarray, papers: pd.Series) -> np.ndarray | None:
        cfg = self.cfg
        if cfg.paper_weight == "none" and cfg.class_weight in ("none", "spw"):
            return None
        return combined_weights(
            y,
            papers,
            paper_strategy=cfg.paper_weight,
            use_class=(cfg.class_weight == "balanced"),
        )

    def _spw(self, y: np.ndarray) -> float:
        if self.cfg.class_weight == "spw":
            return float((y == 0).sum() / max((y == 1).sum(), 1))
        return 1.0

    def _monotone_vector(self, n_features_out: int) -> str | None:
        if not self.cfg.monotone_o2:
            return None
        # oxygen_fraction is always the first numeric (passthrough) column
        cons = [0] * n_features_out
        cons[self.features.index("oxygen_fraction")] = 1
        return "(" + ",".join(str(c) for c in cons) + ")"

    def _make_xgb(self, y: np.ndarray, n_features_out: int, seed: int) -> XGBClassifier:
        cfg = self.cfg
        kwargs = dict(
            **cfg.params,
            tree_method="hist",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
        if cfg.objective == "focal":
            kwargs["objective"] = make_focal_objective(cfg.focal_gamma)
            kwargs["base_score"] = 0.5  # neutral margin start for the custom objective
        else:
            kwargs["objective"] = "binary:logistic"
            kwargs["scale_pos_weight"] = self._spw(y)
        mono = self._monotone_vector(n_features_out)
        if mono is not None:
            kwargs["monotone_constraints"] = mono
        return XGBClassifier(**kwargs)

    # -- API ----------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray, papers: pd.Series) -> "FableModel":
        Xt = self.pre.fit_transform(X[self.features]).astype(np.float32)
        w = self._sample_weight(y, papers)
        self.models_ = []
        cfg = self.cfg
        if cfg.paper_bagging <= 0:
            m = self._make_xgb(y, Xt.shape[1], cfg.random_state)
            m.fit(Xt, y, sample_weight=w)
            self.models_.append(m)
        else:
            # cluster bootstrap: resample *papers* with replacement
            rng = np.random.default_rng(cfg.random_state)
            paper_arr = papers.to_numpy()
            unique_papers = pd.unique(paper_arr)
            row_idx_of_paper = {p: np.flatnonzero(paper_arr == p) for p in unique_papers}
            for b in range(cfg.paper_bagging):
                chosen = rng.choice(unique_papers, size=len(unique_papers), replace=True)
                idx = np.concatenate([row_idx_of_paper[p] for p in chosen])
                yb = y[idx]
                if len(np.unique(yb)) < 2:
                    continue
                wb = w[idx] if w is not None else None
                m = self._make_xgb(yb, Xt.shape[1], cfg.random_state + b)
                m.fit(Xt[idx], yb, sample_weight=wb)
                self.models_.append(m)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.pre.transform(X[self.features]).astype(np.float32)
        preds = []
        for m in self.models_:
            if self.cfg.objective == "focal":
                # custom objective -> raw margins; map through sigmoid
                margins = m.predict(Xt, output_margin=True)
                preds.append(1.0 / (1.0 + np.exp(-margins)))
            else:
                preds.append(m.predict_proba(Xt)[:, 1])
        return np.mean(preds, axis=0)


# ---------------------------------------------------------------------------
# Threshold optimisation (validation folds only — requirement C)
# ---------------------------------------------------------------------------

def optimal_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Thresholds maximising MCC / F1 / balanced accuracy / Youden J.

    Vectorised over the ROC operating points. Must only ever be fed
    *validation* predictions (out-of-fold, held-out papers); applying it to
    test predictions would leak the test labels into the decision rule.
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    P = float((y_true == 1).sum())
    N = float((y_true == 0).sum())
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

    # roc_curve prepends an artificial threshold = max(score)+1; skip it
    valid = slice(1, None)

    def _pick(metric: np.ndarray) -> float:
        idx = int(np.argmax(metric[valid])) + 1
        t = float(thr[idx])
        return min(max(t, 0.0), 1.0)

    return {
        "mcc": _pick(mcc),
        "f1": _pick(f1),
        "balanced_accuracy": _pick(bal),
        "youden_j": _pick(youden),
    }


def grouped_oof_predictions(
    model_factory,
    X: pd.DataFrame,
    y: np.ndarray,
    papers: pd.Series,
    n_splits: int = 4,
    seed: int = RNG_DEFAULT,
) -> np.ndarray:
    """Out-of-fold predictions where every fold holds out whole papers."""
    oof = np.full(len(y), np.nan)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(X, y, groups=papers):
        m = model_factory()
        m.fit(X.iloc[tr], y[tr], papers.iloc[tr])
        oof[te] = m.predict_proba(X.iloc[te])
    return oof


# ---------------------------------------------------------------------------
# Group-aware hyperparameter optimisation (requirement E)
# ---------------------------------------------------------------------------

PARAM_SPACE = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
    "min_child_weight": [1, 2, 4, 6, 10, 20],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.8, 1.0],
    "gamma": [0.0, 0.1, 0.5, 1.0, 2.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
}


def group_aware_search(
    base_cfg: FableConfig,
    X: pd.DataFrame,
    y: np.ndarray,
    papers: pd.Series,
    n_iter: int = 40,
    n_splits: int = 5,
    seed: int = RNG_DEFAULT,
    log_path: Path | None = None,
) -> tuple[dict, float, pd.DataFrame]:
    """Random search scored by *held-out-paper* ROC-AUC.

    Every candidate is evaluated with StratifiedGroupKFold over canonical
    papers; sample weights are recomputed on each training fold. The score
    is the mean of per-fold pooled ROC-AUC. This replaces the previous
    RandomizedSearchCV(StratifiedKFold) which selected hyperparameters that
    memorise campaigns (deep trees, weak regularisation).
    """
    rng = np.random.default_rng(seed)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(sgkf.split(X, y, groups=papers))
    records = []
    best_score, best_params = -np.inf, dict(DEFAULT_PARAMS)
    for it in range(n_iter):
        params = {k: rng.choice(v).item() for k, v in PARAM_SPACE.items()}
        cfg = FableConfig(**{**base_cfg.__dict__, "params": params})
        fold_scores = []
        for tr, te in splits:
            m = FableModel(cfg)
            m.fit(X.iloc[tr], y[tr], papers.iloc[tr])
            p = m.predict_proba(X.iloc[te])
            fold_scores.append(roc_auc_score(y[te], p))
        score = float(np.mean(fold_scores))
        records.append({**params, "group_roc_auc": score, "std": float(np.std(fold_scores))})
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "model_outputs"))
    # Defaults = the recommended configuration from the benchmark
    # (<this script dir>/eval_outputs/comparison_table.md): combined 1/sqrt(N_paper) x class
    # weighting with group-tuned hyperparameters. No benchmarked
    # configuration separates from any other beyond fold noise on unseen
    # papers, so the default is chosen for statistical defensibility
    # (cluster-aware weighting, group-aware tuning), not for a metric win.
    ap.add_argument("--feature-set", choices=["all", "physics"], default="all")
    ap.add_argument(
        "--paper-weight",
        choices=["none", "inverse", "sqrt", "effective", "log"],
        default="sqrt",
    )
    ap.add_argument("--class-weight", choices=["none", "balanced", "spw"], default="balanced")
    ap.add_argument("--objective", choices=["logistic", "focal"], default="logistic")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--monotone-o2", action="store_true", default=False)
    ap.add_argument("--paper-bagging", type=int, default=0)
    ap.add_argument(
        "--params",
        choices=["default", "group_tuned"],
        default="group_tuned",
        help="Hyperparameter preset (group_tuned = output of the group-aware search).",
    )
    ap.add_argument("--tune", action="store_true", help="Run group-aware random search.")
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--random-state", type=int, default=RNG_DEFAULT)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading {args.data} ...")
    df = load_clean(args.data)
    X = df[[c for c in df.columns]]
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"]
    print(
        f"  {len(df)} rows, {papers.nunique()} canonical papers, "
        f"{int(y.sum())} ignition / {int((1 - y).sum())} no-ignition"
    )

    params = dict(DEFAULT_PARAMS) if args.params == "default" else dict(GROUP_TUNED_PARAMS)
    cfg = FableConfig(
        feature_set=args.feature_set,
        paper_weight=args.paper_weight,
        class_weight=args.class_weight,
        objective=args.objective,
        focal_gamma=args.focal_gamma,
        monotone_o2=args.monotone_o2,
        paper_bagging=args.paper_bagging,
        params=params,
        random_state=args.random_state,
    )
    print(f"[2/5] Configuration: {cfg.label()}")

    if args.tune:
        print(f"[3/5] Group-aware random search ({args.n_iter} candidates) ...")
        best_params, best_score, _ = group_aware_search(
            cfg,
            X,
            y,
            papers,
            n_iter=args.n_iter,
            seed=args.random_state,
            log_path=out_dir / "group_search_history.csv",
        )
        print(f"  best held-out-paper ROC-AUC: {best_score:.4f}")
        print(f"  best params: {best_params}")
        cfg.params = best_params
    else:
        print("[3/5] Skipping search (use --tune). Using preset:", args.params)

    print("[4/5] Validation-fold threshold optimisation (held-out papers, no test leakage) ...")
    oof = grouped_oof_predictions(
        lambda: FableModel(cfg), X, y, papers, n_splits=4, seed=args.random_state
    )
    thresholds = optimal_thresholds(y, oof)
    oof_metrics = {
        "oof_group_roc_auc": float(roc_auc_score(y, oof)),
        "oof_group_pr_auc": float(average_precision_score(y, oof)),
    }
    for name, t in thresholds.items():
        pred = (oof >= t).astype(int)
        oof_metrics[f"oof_{name}_threshold"] = t
        oof_metrics[f"oof_mcc_at_{name}"] = float(matthews_corrcoef(y, pred))
        oof_metrics[f"oof_f1_at_{name}"] = float(f1_score(y, pred))
        oof_metrics[f"oof_balacc_at_{name}"] = float(balanced_accuracy_score(y, pred))
    print(f"  OOF grouped ROC-AUC: {oof_metrics['oof_group_roc_auc']:.4f}")
    print(f"  thresholds: { {k: round(v, 3) for k, v in thresholds.items()} }")

    print("[5/5] Fitting final model on all data and saving ...")
    final = FableModel(cfg)
    final.fit(X, y, papers)
    joblib.dump(final, out_dir / "fable_model.joblib")

    payload = {
        "config": {**{k: v for k, v in cfg.__dict__.items() if k != "params"}, "params": cfg.params},
        "n_rows": int(len(df)),
        "n_papers": int(papers.nunique()),
        "thresholds_from_validation_folds": thresholds,
        **oof_metrics,
    }
    with open(out_dir / "fable_model_card.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  written: {out_dir / 'fable_model.joblib'}")
    print(f"  written: {out_dir / 'fable_model_card.json'}")


if __name__ == "__main__":
    main()
