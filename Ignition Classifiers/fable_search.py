"""Leakage-safe nested hyperparameter search and threshold optimization."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, matthews_corrcoef,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from fable_models import MODEL_REGISTRY, make_model

THRESHOLD_NAMES = ("mcc", "f1", "balanced_accuracy", "youden_j")


@dataclass
class SearchResult:
    selected_params: dict[str, Any]
    thresholds: dict[str, float]
    history: pd.DataFrame
    inner_predictions: pd.DataFrame


def classification_metrics(y: np.ndarray, probability: np.ndarray,
                           threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = _confusion(y, prediction)
    two_classes = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, probability)) if two_classes else np.nan,
        "pr_auc": float(average_precision_score(y, probability)) if two_classes else np.nan,
        "brier": float(brier_score_loss(y, probability)),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "precision": float(tp / (tp + fp)) if tp + fp else np.nan,
    }


def _confusion(y: np.ndarray, prediction: np.ndarray) -> tuple[int, int, int, int]:
    y, prediction = np.asarray(y), np.asarray(prediction)
    tn = int(((y == 0) & (prediction == 0)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    tp = int(((y == 1) & (prediction == 1)).sum())
    return tn, fp, fn, tp


def optimize_thresholds(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Select four distinct operating policies from validation predictions only."""
    y, probability = np.asarray(y), np.asarray(probability)
    candidates = np.unique(np.r_[0., probability, 1.])
    best = {name: (-np.inf, .5) for name in THRESHOLD_NAMES}
    for threshold in candidates:
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = _confusion(y, prediction)
        sensitivity = tp / (tp + fn) if tp + fn else 0.
        specificity = tn / (tn + fp) if tn + fp else 0.
        scores = {
            "mcc": matthews_corrcoef(y, prediction),
            "f1": f1_score(y, prediction, zero_division=0),
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "youden_j": sensitivity + specificity - 1,
        }
        for name, score in scores.items():
            if score > best[name][0] or (score == best[name][0] and abs(threshold - .5) <
                                         abs(best[name][1] - .5)):
                best[name] = (float(score), float(threshold))
    return {name: value[1] for name, value in best.items()}


def _sample_configs(candidate: dict[str, Any], iterations: int, seed: int) -> list[dict[str, Any]]:
    spec = MODEL_REGISTRY[candidate["model_family"]]
    space = candidate["search_space"] if "search_space" in candidate else spec.search_space
    if not space:
        return [{}]
    rng = np.random.default_rng(seed)
    configurations: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(configurations) < iterations and attempts < iterations * 100:
        configuration = {key: values[int(rng.integers(len(values)))]
                         for key, values in sorted(space.items())}
        token = json.dumps(configuration, sort_keys=True)
        if token not in seen:
            seen.add(token)
            configurations.append(configuration)
        attempts += 1
    if not configurations:
        configurations.append({})
    return configurations


def _inner_splits(X: pd.DataFrame, y: np.ndarray, papers: pd.Series,
                  protocol: str, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if protocol in {"extrapolation_grouped", "lopo"}:
        maximum = min(folds, int(papers.nunique()))
        for n_splits in range(maximum, 1, -1):
            for offset in range(100):
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits, shuffle=True, random_state=seed + offset)
                splits = list(splitter.split(X, y, groups=papers))
                valid = all(
                    not (set(papers.iloc[train]) & set(papers.iloc[validation])) and
                    len(np.unique(y[train])) == 2 and len(np.unique(y[validation])) == 2
                    for train, validation in splits
                )
                if valid:
                    return splits
        raise ValueError(
            "No deterministic paper-disjoint inner split with both classes in every fold "
            f"could be generated for {protocol}")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(X, y))


def _fit_inner_fold(candidate: dict[str, Any], params: dict[str, Any],
                    X: pd.DataFrame, y: np.ndarray, papers: pd.Series,
                    protocol: str, seed: int, config_number: int, fold: int,
                    train: np.ndarray, validation: np.ndarray
                    ) -> tuple[int, np.ndarray, dict[str, float]]:
    model = make_model(candidate, params, seed + config_number * 1000 + fold)
    model.fit(X.iloc[train], y[train], papers.iloc[train])
    probability = model.predict_proba(X.iloc[validation])
    if len(np.unique(y[validation])) < 2:
        raise ValueError(f"Inner fold {fold} for {protocol} contains one target class")
    return fold, probability, classification_metrics(y[validation], probability, .5)


def nested_search(candidate: dict[str, Any], X: pd.DataFrame, y: np.ndarray,
                  papers: pd.Series, protocol: str, iterations: int,
                  inner_folds: int, seed: int, n_jobs: int = -1) -> SearchResult:
    """Tune only on an outer training partition and freeze validation thresholds."""
    splits = _inner_splits(X, y, papers, protocol, inner_folds, seed)
    configs = _sample_configs(candidate, iterations, seed)
    histories, predictions_by_config = [], {}
    # CPU families parallelize folds; GPU families already saturate their assigned device(s).
    parallel_jobs = n_jobs if candidate["model_family"] in {"knn", "decision_tree"} else 1
    for config_number, params in enumerate(configs):
        oof = np.full(len(y), np.nan)
        fold_records = []
        results = Parallel(n_jobs=parallel_jobs, prefer="threads")(
            delayed(_fit_inner_fold)(
                candidate, params, X, y, papers, protocol, seed, config_number,
                fold, train, validation)
            for fold, (train, validation) in enumerate(splits)
        )
        for fold, probability, metrics in results:
            validation = splits[fold][1]
            oof[validation] = probability
            fold_records.append(metrics)
            histories.append({
                "configuration_id": config_number, "inner_fold": fold,
                "parameters": json.dumps(params, sort_keys=True), **metrics,
            })
        if np.isnan(oof).any():
            raise ValueError("Inner OOF predictions are incomplete")
        predictions_by_config[config_number] = oof
        means = pd.DataFrame(fold_records).mean(numeric_only=True)
        histories.append({
            "configuration_id": config_number, "inner_fold": "mean",
            "parameters": json.dumps(params, sort_keys=True), **means.to_dict(),
        })
    history = pd.DataFrame(histories)
    means = history[history["inner_fold"] == "mean"].copy()
    means = means.sort_values(
        ["roc_auc", "pr_auc", "brier"], ascending=[False, False, True], kind="stable")
    best_id = int(means.iloc[0]["configuration_id"])
    selected = configs[best_id]
    selected_oof = predictions_by_config[best_id]
    thresholds = optimize_thresholds(y, selected_oof)
    inner_predictions = pd.DataFrame({
        "row_id": X["row_id"].to_numpy(), "true_label": y,
        "predicted_probability": selected_oof,
    })
    return SearchResult(selected, thresholds, history, inner_predictions)
