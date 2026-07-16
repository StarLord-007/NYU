"""Unified model interface and registry for all classifier families."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from fable_common import combined_weights, feature_lists

RANDOM_STATE = 42


@dataclass(frozen=True)
class FocalObjective:
    """Pickle-safe stable binary focal-loss objective for XGBoost."""
    gamma: float

    def __call__(self, y_true: np.ndarray, raw: np.ndarray,
                 sample_weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        y = y_true.astype(float)
        p = np.clip(1 / (1 + np.exp(-np.clip(raw, -30, 30))), 1e-7, 1 - 1e-7)
        pt = np.where(y == 1, p, 1 - p)
        sign = np.where(y == 1, -1., 1.)
        grad = sign * (1 - pt) ** self.gamma * (
            self.gamma * pt * np.log(pt) - (1 - pt))
        hess = (1 - pt) ** self.gamma * pt * (
            (1 - pt) + self.gamma * (
                -np.log(pt) * (1 - (self.gamma + 1) * pt) + 2 * (1 - pt)))
        hess = np.maximum(hess, 1e-6)
        if sample_weight is not None:
            grad, hess = grad * sample_weight, hess * sample_weight
        return grad, hess


def focal_objective(gamma: float) -> FocalObjective:
    return FocalObjective(gamma)


@dataclass(frozen=True)
class ModelSpec:
    family: str
    estimator_factory: Callable[[dict[str, Any], int], Any]
    search_space: dict[str, list[Any]]
    supports_sample_weight: bool
    scale_numeric: bool
    native_numeric_missing: bool
    deterministic_note: str


def _xgb(params: dict[str, Any], seed: int) -> XGBClassifier:
    objective = params.pop("objective_variant", "logistic")
    focal_gamma = float(params.pop("focal_gamma", 2.0))
    kwargs = {
        "tree_method": "hist", "eval_metric": "logloss", "n_jobs": -1,
        "random_state": seed, **params,
    }
    if objective == "focal":
        kwargs.update(objective=focal_objective(focal_gamma), base_score=.5)
    else:
        kwargs["objective"] = "binary:logistic"
    return XGBClassifier(**kwargs)


def _knn(params: dict[str, Any], _: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_jobs=-1, **params)


def _tree(params: dict[str, Any], seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=seed, **params)


def _mlp(params: dict[str, Any], seed: int) -> MLPClassifier:
    return MLPClassifier(max_iter=600, early_stopping=True, n_iter_no_change=25,
                         random_state=seed, **params)


def _svm(params: dict[str, Any], seed: int) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        SVC(kernel="rbf", cache_size=1000, random_state=seed, **params),
        method="sigmoid", cv=3,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "xgboost": ModelSpec(
        "xgboost", _xgb, {
            "n_estimators": [200, 400, 600, 800], "learning_rate": [.01, .03, .05, .1],
            "max_depth": [2, 3, 4, 5, 6], "min_child_weight": [1, 2, 4, 8],
            "subsample": [.6, .8, 1.], "colsample_bytree": [.6, .8, 1.],
            "reg_lambda": [.5, 1., 2., 5.], "reg_alpha": [0., .1, .5],
            "gamma": [0., .1, .5],
        }, True, False, True, "Deterministic seed; histogram tree method."
    ),
    "knn": ModelSpec(
        "knn", _knn, {"n_neighbors": [5, 9, 15, 25, 40, 60],
                      "weights": ["uniform", "distance"], "p": [1, 2],
                      "leaf_size": [15, 30, 60]},
        False, True, False, "Deterministic; seed retained by common interface."
    ),
    "decision_tree": ModelSpec(
        "decision_tree", _tree, {"max_depth": [3, 5, 7, 10, None],
                                 "min_samples_leaf": [1, 2, 5, 10, 20],
                                 "criterion": ["gini", "entropy", "log_loss"]},
        True, False, False, "Random state fixed for tie resolution."
    ),
    "mlp": ModelSpec(
        "mlp", _mlp, {"hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                      "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                      "learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3],
                      "activation": ["relu", "tanh"], "batch_size": [32, 64, 128]},
        False, True, False, "Random state fixes initialization and early stopping."
    ),
    "svm": ModelSpec(
        "svm", _svm, {"C": [.1, .3, 1., 3., 10., 30.],
                      "gamma": ["scale", .01, .03, .1, .3], "shrinking": [True, False]},
        True, True, False, "Random state fixes SVC and calibrated probabilities."
    ),
}


class FableModel:
    """Model wrapper exposing fit(X,y,papers), predict_proba and feature names."""

    def __init__(self, family: str, feature_set: str, params: dict[str, Any],
                 paper_weight: str = "none", class_weight: bool = False,
                 random_state: int = RANDOM_STATE, monotone_oxygen: bool = False,
                 paper_bagging: int = 0):
        if family not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model family {family!r}")
        self.family, self.spec, self.feature_set = family, MODEL_REGISTRY[family], feature_set
        self.params = dict(params)
        self.paper_weight, self.class_weight = paper_weight, class_weight
        self.random_state, self.monotone_oxygen = random_state, monotone_oxygen
        self.paper_bagging = int(paper_bagging)
        if self.paper_bagging and family != "xgboost":
            raise ValueError("Paper bagging is restricted to scientifically justified XGBoost candidates")
        self.numeric, self.categorical = feature_lists(feature_set)
        self.features = self.numeric + self.categorical
        self.preprocessor_: ColumnTransformer | None = None
        self.estimators_: list[Any] = []

    def _preprocessor(self) -> ColumnTransformer:
        numeric_steps: list[tuple[str, Any]] = []
        if not self.spec.native_numeric_missing:
            numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
        if self.spec.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_transformer: Any = Pipeline(numeric_steps) if numeric_steps else "passthrough"
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        return ColumnTransformer([
            ("numeric", numeric_transformer, self.numeric),
            ("categorical", categorical_transformer, self.categorical),
        ], remainder="drop")

    def _weights(self, y: np.ndarray, papers: pd.Series) -> np.ndarray:
        return combined_weights(y, papers, self.paper_weight, self.class_weight)

    def _resample(self, y: np.ndarray, weights: np.ndarray, seed: int) -> np.ndarray:
        """Deterministic probability-proportional resampling for estimators lacking weights."""
        rng = np.random.default_rng(seed)
        probabilities = weights / weights.sum()
        indices = rng.choice(np.arange(len(y)), size=len(y), replace=True, p=probabilities)
        if len(np.unique(y[indices])) < 2:
            raise ValueError("Weighted resampling produced one class; training fold is invalid")
        return indices

    def _make_estimator(self, y: np.ndarray, width: int, seed: int) -> Any:
        params = dict(self.params)
        if self.family == "xgboost" and self.monotone_oxygen:
            constraints = [0] * width
            constraints[self.numeric.index("oxygen_fraction")] = 1
            params["monotone_constraints"] = "(" + ",".join(map(str, constraints)) + ")"
        return self.spec.estimator_factory(params, seed)

    def fit(self, X: pd.DataFrame, y: np.ndarray, papers: pd.Series) -> "FableModel":
        y = np.asarray(y, dtype=int)
        papers = pd.Series(papers).reset_index(drop=True)
        self.preprocessor_ = self._preprocessor()
        transformed = np.asarray(self.preprocessor_.fit_transform(X[self.features]), dtype=np.float32)
        weights = self._weights(y, papers)
        self.estimators_ = []
        if self.paper_bagging:
            rng = np.random.default_rng(self.random_state)
            unique = papers.unique()
            rows = {paper: np.flatnonzero(papers.to_numpy() == paper) for paper in unique}
            training_indices = [
                np.concatenate([rows[p] for p in rng.choice(unique, len(unique), replace=True)])
                for _ in range(self.paper_bagging)
            ]
        else:
            training_indices = [np.arange(len(y))]
        for number, indices in enumerate(training_indices):
            estimator = self._make_estimator(y[indices], transformed.shape[1],
                                             self.random_state + number)
            if self.spec.supports_sample_weight:
                estimator.fit(transformed[indices], y[indices], sample_weight=weights[indices])
            else:
                sampled = self._resample(y[indices], weights[indices], self.random_state + number)
                estimator.fit(transformed[indices][sampled], y[indices][sampled])
            self.estimators_.append(estimator)
        if not self.estimators_:
            raise RuntimeError("No estimator was fitted")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.preprocessor_ is None or not self.estimators_:
            raise RuntimeError("Model is not fitted")
        transformed = np.asarray(self.preprocessor_.transform(X[self.features]), dtype=np.float32)
        predictions = []
        for estimator in self.estimators_:
            if self.family == "xgboost" and self.params.get("objective_variant") == "focal":
                margin = estimator.predict(transformed, output_margin=True)
                probability = 1 / (1 + np.exp(-np.clip(margin, -30, 30)))
            else:
                probability = estimator.predict_proba(transformed)[:, 1]
            predictions.append(probability)
        result = np.mean(predictions, axis=0)
        if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
            raise ValueError("Estimator produced invalid class-1 probabilities")
        return result

    def get_feature_names_out(self) -> np.ndarray:
        if self.preprocessor_ is None:
            raise RuntimeError("Model is not fitted")
        return self.preprocessor_.get_feature_names_out()


def make_model(candidate: dict[str, Any], params: dict[str, Any] | None = None,
               seed: int = RANDOM_STATE) -> FableModel:
    merged = dict(candidate.get("fixed_params", {}))
    merged.update(params or {})
    return FableModel(
        family=candidate["model_family"], feature_set=candidate["feature_set"],
        params=merged, paper_weight=candidate.get("paper_weight", "none"),
        class_weight=bool(candidate.get("class_weight", False)), random_state=seed,
        monotone_oxygen=bool(candidate.get("monotone_oxygen", False)),
        paper_bagging=int(candidate.get("paper_bagging", 0)),
    )
