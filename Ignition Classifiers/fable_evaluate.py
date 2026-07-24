"""The sole benchmark runner: nested tuning on immutable outer splits."""
from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from fable_common import DATA_VERSION, configure_torch, empty_cuda_cache, load_data
from fable_models import make_model
from fable_search import THRESHOLD_NAMES, classification_metrics, nested_search

MODEL_ID_TO_FAMILY = ("xgboost", "svm", "knn", "decision_tree", "mlp")


def _load_candidates(path: str | Path) -> tuple[list[dict[str, Any]], int]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    ids = [candidate.get("candidate_id") for candidate in candidates]
    if not candidates or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Candidate config must contain unique non-empty candidate_id values")
    return candidates, int(payload.get("random_state", 42))


def _split_indices(split: pd.DataFrame, row_lookup: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    train_ids = split.loc[split["partition"] == "train", "row_id"]
    test_ids = split.loc[split["partition"] == "test", "row_id"]
    unknown = (set(train_ids) | set(test_ids)) - set(row_lookup)
    if unknown:
        raise ValueError(f"Split contains {len(unknown)} row IDs absent from loaded data")
    return (np.array([row_lookup[x] for x in train_ids], dtype=int),
            np.array([row_lookup[x] for x in test_ids], dtype=int))


def _fold_metrics(y: np.ndarray, probability: np.ndarray,
                  thresholds: dict[str, float]) -> dict[str, float]:
    two_classes = len(np.unique(y)) == 2
    record = {
        "roc_auc": float(roc_auc_score(y, probability)) if two_classes else np.nan,
        "pr_auc": float(average_precision_score(y, probability)) if two_classes else np.nan,
        "brier": float(brier_score_loss(y, probability)),
    }
    for policy, threshold in thresholds.items():
        metrics = classification_metrics(y, probability, threshold)
        for metric in ("mcc", "f1", "balanced_accuracy", "sensitivity",
                       "specificity", "precision"):
            record[f"{metric}_at_{policy}"] = metrics[metric]
    return record


def _bootstrap(predictions: pd.DataFrame, grouped: bool, iterations: int,
               seed: int) -> list[dict[str, Any]]:
    collapsed = (predictions.groupby(["row_id", "paper_id", "true_label"], as_index=False)
                 ["predicted_probability"].mean())
    rng = np.random.default_rng(seed)
    rows = []
    units = collapsed["paper_id"].unique() if grouped else np.arange(len(collapsed))
    for metric in ("roc_auc", "pr_auc", "brier"):
        values = []
        for _ in range(iterations):
            sampled = rng.choice(units, len(units), replace=True)
            if grouped:
                parts = [collapsed[collapsed["paper_id"] == unit] for unit in sampled]
                sample = pd.concat(parts, ignore_index=True)
            else:
                sample = collapsed.iloc[sampled]
            y = sample["true_label"].to_numpy()
            p = sample["predicted_probability"].to_numpy()
            if len(np.unique(y)) < 2:
                continue
            values.append(
                roc_auc_score(y, p) if metric == "roc_auc" else
                average_precision_score(y, p) if metric == "pr_auc" else
                brier_score_loss(y, p)
            )
        if not values:
            low = high = np.nan
        else:
            low, high = np.quantile(values, [.025, .975])
        rows.append({
            "metric": metric, "estimate": (
                roc_auc_score(collapsed["true_label"], collapsed["predicted_probability"])
                if metric == "roc_auc" else
                average_precision_score(collapsed["true_label"], collapsed["predicted_probability"])
                if metric == "pr_auc" else
                brier_score_loss(collapsed["true_label"], collapsed["predicted_probability"])),
            "ci_low": float(low), "ci_high": float(high), "confidence_level": .95,
            "iterations": iterations,
            "resampling_unit": "canonical_paper" if grouped else "unique_row_id",
            "repeat_aggregation": "mean probability per row_id before resampling",
        })
    return rows


def _summaries(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_id", "model_family", "feature_set", "protocol"]
    numeric = [c for c in fold_metrics.select_dtypes(include="number").columns
               if c not in {"seed", "fold", "n_test_rows", "n_test_papers"}]
    rows = []
    for values, group in fold_metrics.groupby(keys):
        record = dict(zip(keys, values))
        record.update({
            "fold_count": len(group), "sample_count": int(group["n_test_rows"].sum()),
            "paper_count_sum": int(group["n_test_papers"].sum()),
        })
        for metric in numeric:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(record)
    return pd.DataFrame(rows)


def _paired(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ("roc_auc", "pr_auc", "brier", "mcc_at_mcc", "f1_at_f1",
               "balanced_accuracy_at_balanced_accuracy")
    for protocol, protocol_frame in fold_metrics.groupby("protocol"):
        models = sorted(protocol_frame["model_id"].unique())
        indexed = {model: protocol_frame[protocol_frame["model_id"] == model]
                   .set_index("split_id") for model in models}
        for left, right in combinations(models, 2):
            common = indexed[left].index.intersection(indexed[right].index)
            record = {"protocol": protocol, "model_a": left, "model_b": right,
                      "paired_fold_count": len(common)}
            for metric in metrics:
                differences = (indexed[left].loc[common, metric] -
                               indexed[right].loc[common, metric]).to_numpy()
                differences = differences[np.isfinite(differences)]
                record[f"delta_{metric}"] = float(np.mean(differences)) if len(differences) else np.nan
                try:
                    record[f"wilcoxon_p_{metric}"] = (
                        float(wilcoxon(differences).pvalue)
                        if len(differences) and np.any(differences != 0) else 1.)
                except ValueError:
                    record[f"wilcoxon_p_{metric}"] = np.nan
            record["caveat"] = (
                "Exploratory paired test over identical outer folds; multiplicity is not adjusted.")
            rows.append(record)
    return pd.DataFrame(rows)


def _per_paper(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model_id", "model_family", "feature_set", "protocol", "paper_id", "paper_label"]
    for values, group in predictions.groupby(keys, dropna=False):
        y, p = group["true_label"].to_numpy(), group["predicted_probability"].to_numpy()
        rows.append({
            **dict(zip(keys, values)), "n_predictions": len(group),
            "n_unique_rows": group["row_id"].nunique(), "ignition_prevalence": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
            "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
            "brier": float(brier_score_loss(y, p)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--search-iterations", type=int, default=40)
    parser.add_argument("--inner-group-folds", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--model-id", type=int, choices=range(len(MODEL_ID_TO_FAMILY)),
        help="Run one model family: 0=XGBoost, 1=SVM, 2=KNN, 3=Decision Tree, 4=MLP",
    )
    args = parser.parse_args()
    configure_torch()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    candidates, base_seed = _load_candidates(args.config)
    indexed_candidates = list(enumerate(candidates))
    if args.model_id is not None:
        selected_family = MODEL_ID_TO_FAMILY[args.model_id]
        indexed_candidates = [
            (number, candidate) for number, candidate in indexed_candidates
            if candidate["model_family"] == selected_family
        ]
        if not indexed_candidates:
            raise ValueError(f"No configured candidates found for {selected_family}")
    df, data_report = load_data(args.data, require_target=True)
    split_dir = Path(args.splits)
    assignments = pd.read_parquet(split_dir / "split_assignments.parquet")
    split_metadata = json.loads((split_dir / "splits_metadata.json").read_text(encoding="utf-8"))
    if split_metadata["dataset_sha256"] != data_report["dataset_sha256"]:
        raise ValueError("Dataset fingerprint differs from the dataset used to generate splits")
    row_lookup = {row_id: i for i, row_id in enumerate(df["row_id"])}
    expected_protocols = {
        "interpolation_holdout", "interpolation_stratified", "extrapolation_grouped", "lopo"}
    if set(assignments["protocol"].unique()) != expected_protocols:
        raise ValueError("Persisted splits do not contain every required protocol")

    manifest_rows, prediction_rows, metric_rows = [], [], []
    threshold_rows, parameter_rows, history_frames, inner_prediction_frames, failures = [], [], [], [], []
    explainability_rows, explained_candidates = [], set()
    protocol_status: dict[str, dict[str, Any]] = {}
    split_groups = list(assignments.groupby("split_id", sort=False))
    for selected_number, (candidate_number, candidate) in enumerate(indexed_candidates):
        candidate_id = candidate["candidate_id"]
        grouped_params_by_paper: dict[str, list[str]] = {}
        manifest_rows.append({
            **{k: v for k, v in candidate.items()
               if k not in {"fixed_params", "search_space"}},
            "fixed_params": json.dumps(candidate.get("fixed_params", {}), sort_keys=True),
            "weighting_policy": json.dumps({
                "paper_weight": candidate.get("paper_weight", "none"),
                "class_weight": candidate.get("class_weight", False),
                "unsupported_fit_weight_strategy": "deterministic_weighted_resampling",
            }, sort_keys=True),
        })
        for split_number, (split_id, split) in enumerate(split_groups):
            protocol, seed, fold = (split.iloc[0][x] for x in ("protocol", "seed", "fold"))
            status_key = f"{candidate_id}::{protocol}"
            protocol_status.setdefault(status_key, {
                "candidate_id": candidate_id, "protocol": protocol,
                "expected_folds": int(assignments[assignments["protocol"] == protocol]
                                      ["split_id"].nunique()),
                "completed_folds": 0, "valid_probabilities": True, "passed": True,
                "errors": [],
            })
            try:
                train, test = _split_indices(split, row_lookup)
                train_papers, test_papers = set(df.iloc[train]["paper_id"]), set(df.iloc[test]["paper_id"])
                if protocol in {"extrapolation_grouped", "lopo"} and train_papers & test_papers:
                    raise ValueError("Outer grouped split contains paper leakage")
                y_train = df.iloc[train]["ignition_binary"].to_numpy()
                y_test = df.iloc[test]["ignition_binary"].to_numpy()
                if len(np.unique(y_train)) < 2:
                    raise ValueError("Outer training split does not contain both target classes")
                search_seed = base_seed + candidate_number * 100000 + split_number
                search_candidate = candidate
                search_iterations = args.search_iterations
                frozen_lopo_params: dict[str, Any] | None = None
                if protocol == "lopo":
                    held_out_paper = next(iter(test_papers))
                    eligible_params = grouped_params_by_paper.get(held_out_paper, [])
                    if not eligible_params:
                        raise ValueError(
                            "No grouped-CV hyperparameter selection excluded the LOPO paper")
                    counts = Counter(eligible_params)
                    modal_json = min(
                        value for value, count in counts.items() if count == max(counts.values()))
                    frozen_lopo_params = json.loads(modal_json)
                    search_candidate = {
                        **candidate,
                        "fixed_params": {
                            **candidate.get("fixed_params", {}), **frozen_lopo_params},
                        "search_space": {},
                    }
                    search_iterations = 1
                search = nested_search(
                    search_candidate, df.iloc[train].reset_index(drop=True), y_train,
                    df.iloc[train]["paper_id"].reset_index(drop=True), protocol,
                    search_iterations, args.inner_group_folds, search_seed, n_jobs=-1)
                selected_params = (
                    frozen_lopo_params if frozen_lopo_params is not None
                    else search.selected_params)
                model = make_model(
                    search_candidate,
                    {} if frozen_lopo_params is not None else search.selected_params,
                    search_seed)
                model.fit(df.iloc[train], y_train, df.iloc[train]["paper_id"].reset_index(drop=True))
                probability = model.predict_proba(df.iloc[test])
                if not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
                    protocol_status[status_key]["valid_probabilities"] = False
                    raise ValueError("Invalid probability output")
                selected_json = json.dumps(selected_params, sort_keys=True)
                if protocol == "extrapolation_grouped":
                    for paper in test_papers:
                        grouped_params_by_paper.setdefault(paper, []).append(selected_json)
                weighting_json = manifest_rows[-1]["weighting_policy"]
                if (protocol == "extrapolation_grouped" and
                        candidate["model_family"] == "xgboost" and
                        candidate_id not in explained_candidates):
                    model_dir = out / "explainability_models"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    model_path = model_dir / f"{candidate_id}.joblib"
                    joblib.dump(model, model_path, compress=3)
                    explainability_rows.append({
                        "model_id": candidate_id, "split_id": split_id,
                        "model_path": str(model_path.relative_to(out)),
                        "held_out_row_ids": json.dumps(df.iloc[test]["row_id"].tolist()),
                        "held_out_paper_ids": json.dumps(sorted(test_papers)),
                    })
                    explained_candidates.add(candidate_id)
                for local, (_, row) in enumerate(df.iloc[test].iterrows()):
                    record = {
                        "row_id": row["row_id"], "paper_id": row["paper_id"],
                        "paper_label": row["paper_label"], "protocol": protocol,
                        "seed": seed, "fold": fold, "split_id": split_id,
                        "model_id": candidate_id, "model_family": candidate["model_family"],
                        "feature_set": candidate["feature_set"],
                        "true_label": int(y_test[local]),
                        "predicted_probability": float(probability[local]),
                        "selected_hyperparameters": selected_json,
                        "weighting_policy": weighting_json,
                    }
                    for name, threshold in search.thresholds.items():
                        record[f"threshold_{name}"] = threshold
                        record[f"prediction_{name}"] = int(probability[local] >= threshold)
                    prediction_rows.append(record)
                metrics = _fold_metrics(y_test, probability, search.thresholds)
                metric_rows.append({
                    "split_id": split_id, "protocol": protocol, "seed": seed, "fold": fold,
                    "model_id": candidate_id, "model_family": candidate["model_family"],
                    "feature_set": candidate["feature_set"], "n_test_rows": len(test),
                    "n_test_papers": len(test_papers), **metrics,
                })
                threshold_rows.append({
                    "split_id": split_id, "protocol": protocol, "seed": seed, "fold": fold,
                    "model_id": candidate_id, **search.thresholds,
                })
                parameter_rows.append({
                    "split_id": split_id, "protocol": protocol, "seed": seed, "fold": fold,
                    "model_id": candidate_id, "selected_hyperparameters": selected_json,
                })
                history = search.history.assign(
                    split_id=split_id, protocol=protocol, seed=seed, fold=fold,
                    model_id=candidate_id)
                if frozen_lopo_params is not None:
                    history["parameters"] = selected_json
                history_frames.append(history)
                inner_prediction_frames.append(search.inner_predictions.assign(
                    split_id=split_id, protocol=protocol, seed=seed, fold=fold,
                    model_id=candidate_id))
                protocol_status[status_key]["completed_folds"] += 1
                print(
                    f"[{selected_number + 1}/{len(indexed_candidates)}] {candidate_id} "
                    f"[{split_number + 1}/{len(split_groups)}] {split_id}: complete",
                    flush=True,
                )
            except Exception as exc:
                protocol_status[status_key]["passed"] = False
                protocol_status[status_key]["errors"].append(f"{split_id}: {exc}")
                failures.append({
                    "candidate_id": candidate_id, "protocol": protocol, "split_id": split_id,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(
                    f"[{selected_number + 1}/{len(indexed_candidates)}] {candidate_id} "
                    f"[{split_number + 1}/{len(split_groups)}] {split_id}: FAILED: {exc}",
                    flush=True,
                )
            finally:
                empty_cuda_cache()

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(metric_rows)
    for status in protocol_status.values():
        status["complete"] = status["completed_folds"] == status["expected_folds"]
        status["passed"] = bool(status["passed"] and status["complete"] and
                                status["valid_probabilities"])
    eligible = {(x["candidate_id"], x["protocol"]) for x in protocol_status.values() if x["passed"]}
    if len(predictions):
        mask = predictions.apply(lambda x: (x["model_id"], x["protocol"]) in eligible, axis=1)
        predictions = predictions[mask].reset_index(drop=True)
    if len(fold_metrics):
        mask = fold_metrics.apply(lambda x: (x["model_id"], x["protocol"]) in eligible, axis=1)
        fold_metrics = fold_metrics[mask].reset_index(drop=True)

    predictions.to_csv(out / "outer_fold_predictions.csv", index=False)
    predictions.to_parquet(out / "outer_fold_predictions.parquet", index=False)
    fold_metrics.to_csv(out / "fold_metrics.csv", index=False)
    summary = _summaries(fold_metrics)
    summary.to_csv(out / "summary_metrics.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(out / "candidate_manifest.csv", index=False)
    thresholds_frame = pd.DataFrame(threshold_rows)
    parameters_frame = pd.DataFrame(parameter_rows)
    for frame in (thresholds_frame, parameters_frame):
        if len(frame):
            frame.drop(frame.index[~frame.apply(
                lambda x: (x["model_id"], x["protocol"]) in eligible, axis=1)], inplace=True)
    thresholds_frame.to_csv(out / "thresholds_by_fold.csv", index=False)
    parameters_frame.to_csv(out / "selected_hyperparameters.csv", index=False)
    pd.DataFrame(explainability_rows).to_csv(out / "explainability_index.csv", index=False)
    history_frame = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    if len(history_frame):
        history_frame = history_frame[history_frame.apply(
            lambda x: (x["model_id"], x["protocol"]) in eligible, axis=1)]
    history_frame.to_csv(out / "search_history.csv", index=False)
    inner_predictions = (pd.concat(inner_prediction_frames, ignore_index=True)
                         if inner_prediction_frames else pd.DataFrame())
    if len(inner_predictions):
        inner_predictions = inner_predictions[inner_predictions.apply(
            lambda x: (x["model_id"], x["protocol"]) in eligible, axis=1)]
    inner_predictions.to_csv(out / "inner_oof_predictions.csv", index=False)
    inner_predictions.to_parquet(out / "inner_oof_predictions.parquet", index=False)
    _paired(fold_metrics).to_csv(out / "paired_model_comparisons.csv", index=False)
    _per_paper(predictions).to_csv(out / "per_paper_metrics.csv", index=False)

    intervals = []
    for (model_id, protocol), group in predictions.groupby(["model_id", "protocol"]):
        grouped = protocol in {"extrapolation_grouped", "lopo"}
        for row in _bootstrap(group, grouped, args.bootstrap_iterations,
                              base_seed + sum(map(ord, model_id + protocol))):
            intervals.append({"model_id": model_id, "protocol": protocol, **row})
    pd.DataFrame(intervals).to_csv(out / "bootstrap_intervals.csv", index=False)
    integrity = {
        "passed": not failures, "dataset_hash_matches_splits": True,
        "protocol_status": list(protocol_status.values()), "failures": failures,
        "failed_result_policy": "Failed candidate/protocol results are excluded from result tables.",
    }
    (out / "integrity_checks.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8")
    if failures:
        (out / "evaluation_failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8")
    metadata = {
        "data_version": DATA_VERSION, "dataset_sha256": data_report["dataset_sha256"],
        "candidate_count": len(indexed_candidates), "search_iterations": args.search_iterations,
        "model_id": args.model_id,
        "model_family_filter": (MODEL_ID_TO_FAMILY[args.model_id]
                                if args.model_id is not None else None),
        "inner_folds": args.inner_group_folds,
        "threshold_policy": {
            name: "selected exclusively from inner OOF predictions" for name in THRESHOLD_NAMES},
        "lopo_hyperparameter_policy": (
            "For each held-out paper, freeze the modal selected hyperparameters from repeated "
            "grouped outer folds whose training partitions excluded that paper; resolve modal "
            "ties lexically. LOPO thresholds remain fitted from grouped inner OOF predictions "
            "using only the LOPO training partition."),
        "bootstrap": {
            "iterations": args.bootstrap_iterations,
            "interpolation_unit": "unique row_id",
            "extrapolation_unit": "canonical paper cluster",
            "note": "Repeated outer predictions are averaged per row before bootstrap; "
                    "grouped protocols resample complete canonical-paper clusters.",
        },
    }
    (out / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "integrity_passed": not failures,
                      "eligible_candidate_protocols": len(eligible)}, indent=2))


if __name__ == "__main__":
    main()
