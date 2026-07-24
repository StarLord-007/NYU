"""Unified model interface and registry for all classifier families."""
from __future__ import annotations

import os
import socket
import tempfile
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from xgboost import XGBClassifier

from fable_common import (combined_weights, configure_torch, empty_cuda_cache,
                          feature_lists, slurm_num_workers, torch_device)

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


class _TorchNetwork(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: tuple[int, ...], activation: str):
        super().__init__()
        activation_factory: type[torch.nn.Module] = (
            torch.nn.ReLU if activation == "relu" else torch.nn.Tanh)
        layers: list[torch.nn.Module] = []
        width = input_size
        for hidden in hidden_sizes:
            layers.extend((torch.nn.Linear(width, hidden), activation_factory()))
            width = hidden
        layers.append(torch.nn.Linear(width, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(1)


def _train_network(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                   learning_rate: float, weight_decay: float, max_iter: int,
                   sampler: DistributedSampler | None = None) -> None:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_loss, stale_epochs = float("inf"), 0
    for epoch in range(max_iter):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        total_loss = torch.zeros(2, device=device)
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            total_loss[0] += loss.detach() * len(targets)
            total_loss[1] += len(targets)
        if dist.is_initialized():
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        epoch_loss = float((total_loss[0] / total_loss[1].clamp_min(1)).item())
        if epoch_loss < best_loss - 1e-6:
            best_loss, stale_epochs = epoch_loss, 0
        else:
            stale_epochs += 1
            if stale_epochs >= 25:
                break


def _ddp_worker(rank: int, world_size: int, port: int, features: np.ndarray,
                targets: np.ndarray, settings: dict[str, Any], state_path: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        device = configure_torch(rank)
        dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=RANDOM_STATE)
        loader = DataLoader(
            dataset, batch_size=settings["batch_size"], sampler=sampler,
            pin_memory=True, num_workers=slurm_num_workers())
        network = _TorchNetwork(
            features.shape[1], settings["hidden_layer_sizes"], settings["activation"]).to(device)
        model = DistributedDataParallel(network, device_ids=[rank], output_device=rank)
        _train_network(
            model, loader, device, settings["learning_rate_init"], settings["alpha"],
            settings["max_iter"], sampler)
        if rank == 0:
            torch.save({key: value.detach().cpu()
                        for key, value in model.module.state_dict().items()}, state_path)
        dist.barrier()
    finally:
        dist.destroy_process_group()
        empty_cuda_cache()


class TorchMLPClassifier:
    """Small sklearn-compatible PyTorch MLP with GPU and conditional DDP training."""

    def __init__(self, hidden_layer_sizes: tuple[int, ...] = (100,), alpha: float = .0001,
                 learning_rate_init: float = .001, activation: str = "relu",
                 batch_size: int = 32, max_iter: int = 600, random_state: int = RANDOM_STATE,
                 **_: Any):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.activation = activation
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = RANDOM_STATE
        self.input_size_: int | None = None
        self.state_dict_: dict[str, torch.Tensor] | None = None
        self.classes_ = np.array([0, 1])

    def _settings(self) -> dict[str, Any]:
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes, "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init, "activation": self.activation,
            "batch_size": self.batch_size, "max_iter": self.max_iter,
        }

    @staticmethod
    def _open_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPClassifier":
        torch.manual_seed(42)
        features = np.ascontiguousarray(X, dtype=np.float32)
        targets = np.ascontiguousarray(y, dtype=np.float32)
        self.input_size_ = features.shape[1]
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if gpu_count > 1:
            with tempfile.TemporaryDirectory(prefix="fable-ddp-") as directory:
                state_path = os.path.join(directory, "state.pt")
                mp.spawn(
                    _ddp_worker,
                    args=(gpu_count, self._open_port(), features, targets,
                          self._settings(), state_path),
                    nprocs=gpu_count,
                    join=True,
                )
                self.state_dict_ = torch.load(state_path, map_location="cpu", weights_only=True)
        else:
            device = configure_torch()
            dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                num_workers=slurm_num_workers())
            model = _TorchNetwork(
                self.input_size_, self.hidden_layer_sizes, self.activation).to(device)
            _train_network(
                model, loader, device, self.learning_rate_init, self.alpha, self.max_iter)
            self.state_dict_ = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            empty_cuda_cache()
        return self

    def _model(self, device: torch.device) -> _TorchNetwork:
        if self.input_size_ is None or self.state_dict_ is None:
            raise RuntimeError("MLP is not fitted")
        model = _TorchNetwork(self.input_size_, self.hidden_layer_sizes, self.activation)
        model.load_state_dict(self.state_dict_)
        return model.to(device).eval()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        device = torch_device()
        features = torch.as_tensor(np.asarray(X, dtype=np.float32))
        loader = DataLoader(
            TensorDataset(features), batch_size=self.batch_size, shuffle=False,
            pin_memory=True, num_workers=slurm_num_workers())
        model = self._model(device)
        probabilities = []
        with torch.no_grad():
            for (batch,) in loader:
                logits = model(batch.to(device, non_blocking=True))
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        positive = np.concatenate(probabilities)
        empty_cuda_cache()
        return np.column_stack((1 - positive, positive))


class CuMLSVCWithFallback:
    """Prefer a GPU cuML SVC and retain the calibrated sklearn implementation."""

    def __init__(self, params: dict[str, Any], random_state: int = RANDOM_STATE):
        self.params = dict(params)
        self.random_state = RANDOM_STATE
        self.estimator_: Any | None = None
        self.using_cuml_ = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "CuMLSVCWithFallback":
        if torch.cuda.is_available():
            try:
                from cuml.svm import SVC as CuMLSVC
                gpu_params = {key: value for key, value in self.params.items()
                              if key != "shrinking"}
                estimator = CuMLSVC(
                    kernel="rbf", probability=True, random_state=RANDOM_STATE, **gpu_params)
                estimator.fit(X, y, sample_weight=sample_weight)
                self.estimator_, self.using_cuml_ = estimator, True
                return self
            except (ImportError, ModuleNotFoundError, TypeError, ValueError, RuntimeError) as exc:
                warnings.warn(f"cuML SVC unavailable; using sklearn SVC: {exc}", RuntimeWarning)
        estimator = CalibratedClassifierCV(
            SVC(kernel="rbf", cache_size=1000, random_state=RANDOM_STATE, **self.params),
            method="sigmoid", cv=3, n_jobs=-1)
        estimator.fit(X, y, sample_weight=sample_weight)
        self.estimator_ = estimator
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("SVC is not fitted")
        probability = self.estimator_.predict_proba(X)
        if hasattr(probability, "to_numpy"):
            probability = probability.to_numpy()
        elif hasattr(probability, "get"):
            probability = probability.get()
        return np.asarray(probability)


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
        "tree_method": "gpu_hist", "device": "cuda", "eval_metric": "logloss",
        "n_jobs": -1, "random_state": RANDOM_STATE, **params,
    }
    if objective == "focal":
        kwargs.update(objective=focal_objective(focal_gamma), base_score=.5)
    else:
        kwargs["objective"] = "binary:logistic"
    return XGBClassifier(**kwargs)


def _knn(params: dict[str, Any], _: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_jobs=-1, **params)


def _tree(params: dict[str, Any], seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)


def _mlp(params: dict[str, Any], seed: int) -> TorchMLPClassifier:
    return TorchMLPClassifier(max_iter=600, random_state=RANDOM_STATE, **params)


def _svm(params: dict[str, Any], seed: int) -> CuMLSVCWithFallback:
    return CuMLSVCWithFallback(params, random_state=RANDOM_STATE)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "xgboost": ModelSpec(
        "xgboost", _xgb, {
            "n_estimators": [200, 400, 600, 800], "learning_rate": [.01, .03, .05, .1],
            "max_depth": [2, 3, 4, 5, 6], "min_child_weight": [1, 2, 4, 8],
            "subsample": [.6, .8, 1.], "colsample_bytree": [.6, .8, 1.],
            "reg_lambda": [.5, 1., 2., 5.], "reg_alpha": [0., .1, .5],
            "gamma": [0., .1, .5],
        }, True, False, True, "Random state 42; GPU histogram tree method."
    ),
    "knn": ModelSpec(
        "knn", _knn, {"n_neighbors": [5, 9, 15, 25, 40, 60],
                      "weights": ["uniform", "distance"], "p": [1, 2],
                      "leaf_size": [15, 30, 60]},
        False, True, False, "Deterministic; all available CPU cores are used."
    ),
    "decision_tree": ModelSpec(
        "decision_tree", _tree, {"max_depth": [3, 5, 7, 10, None],
                                 "min_samples_leaf": [1, 2, 5, 10, 20],
                                 "criterion": ["gini", "entropy", "log_loss"]},
        True, False, False, "Random state fixed at 42 for tie resolution."
    ),
    "mlp": ModelSpec(
        "mlp", _mlp, {"hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                      "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                      "learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3],
                      "activation": ["relu", "tanh"], "batch_size": [32, 64, 128]},
        False, True, False, "PyTorch seed 42; one GPU or NCCL DDP on multiple GPUs."
    ),
    "svm": ModelSpec(
        "svm", _svm, {"C": [.1, .3, 1., 3., 10., 30.],
                      "gamma": ["scale", .01, .03, .1, .3], "shrinking": [True, False]},
        True, True, False, "cuML GPU SVC preferred; seeded sklearn fallback."
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
        self.random_state, self.monotone_oxygen = RANDOM_STATE, monotone_oxygen
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
        ], remainder="drop", n_jobs=-1)

    def _weights(self, y: np.ndarray, papers: pd.Series) -> np.ndarray:
        return combined_weights(y, papers, self.paper_weight, self.class_weight)

    def _resample(self, y: np.ndarray, weights: np.ndarray, seed: int) -> np.ndarray:
        """Deterministic probability-proportional resampling for estimators lacking weights."""
        rng = np.random.default_rng(RANDOM_STATE)
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
        return self.spec.estimator_factory(params, RANDOM_STATE)

    def fit(self, X: pd.DataFrame, y: np.ndarray, papers: pd.Series) -> "FableModel":
        y = np.asarray(y, dtype=int)
        papers = pd.Series(papers).reset_index(drop=True)
        self.preprocessor_ = self._preprocessor()
        transformed = np.asarray(self.preprocessor_.fit_transform(X[self.features]), dtype=np.float32)
        weights = self._weights(y, papers)
        self.estimators_ = []
        if self.paper_bagging:
            rng = np.random.default_rng(RANDOM_STATE)
            unique = papers.unique()
            rows = {paper: np.flatnonzero(papers.to_numpy() == paper) for paper in unique}
            training_indices = [
                np.concatenate([rows[p] for p in rng.choice(unique, len(unique), replace=True)])
                for _ in range(self.paper_bagging)
            ]
        else:
            training_indices = [np.arange(len(y))]
        for number, indices in enumerate(training_indices):
            estimator = self._make_estimator(y[indices], transformed.shape[1], RANDOM_STATE)
            if self.spec.supports_sample_weight:
                estimator.fit(transformed[indices], y[indices], sample_weight=weights[indices])
            else:
                sampled = self._resample(y[indices], weights[indices], RANDOM_STATE)
                estimator.fit(transformed[indices][sampled], y[indices][sampled])
            self.estimators_.append(estimator)
            empty_cuda_cache()
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
        empty_cuda_cache()
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
        class_weight=bool(candidate.get("class_weight", False)), random_state=RANDOM_STATE,
        monotone_oxygen=bool(candidate.get("monotone_oxygen", False)),
        paper_bagging=int(candidate.get("paper_bagging", 0)),
    )
