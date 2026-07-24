"""Refit a selected champion and package a self-contained deployable artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from fable_common import DATA_VERSION, feature_manifest, load_data
from fable_models import make_model
from fable_search import optimize_thresholds


def _candidate(selection: dict, champion: str) -> tuple[dict, dict]:
    key = f"{champion}_champion"
    if key not in selection:
        raise ValueError(f"Selection file has no {key}")
    chosen = selection[key]
    config = chosen["candidate_configuration"]
    candidate = {
        "candidate_id": chosen["candidate_id"],
        "model_family": chosen["model_family"],
        "feature_set": chosen["feature_set"],
        "paper_weight": config.get("paper_weight", "none"),
        "class_weight": bool(config.get("class_weight", False)),
        "monotone_oxygen": bool(config.get("monotone_oxygen", False)),
        "paper_bagging": int(config.get("paper_bagging", 0) or 0),
        "fixed_params": {},
    }
    return candidate, chosen


def _oof(df: pd.DataFrame, candidate: dict, params: dict, champion: str,
         seed: int = 42) -> tuple[np.ndarray, pd.DataFrame]:
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"].reset_index(drop=True)
    sums, counts, records = np.zeros(len(df)), np.zeros(len(df), dtype=int), []
    if champion == "interpolation":
        splitters = [
            (seed + offset, StratifiedKFold(5, shuffle=True, random_state=seed + offset)
             .split(df, y))
            for offset in range(3)
        ]
    else:
        splitters = [
            (seed + offset, StratifiedGroupKFold(5, shuffle=True, random_state=seed + offset)
             .split(df, y, groups=papers))
            for offset in range(3)
        ]
    for split_seed, splits in splitters:
        for fold, (train, validation) in enumerate(splits):
            if champion == "extrapolation" and (
                    set(papers.iloc[train]) & set(papers.iloc[validation])):
                raise ValueError("Paper leakage in extrapolation refit OOF split")
            model = make_model(candidate, params, split_seed * 100 + fold)
            model.fit(df.iloc[train], y[train], papers.iloc[train].reset_index(drop=True))
            probability = model.predict_proba(df.iloc[validation])
            sums[validation] += probability
            counts[validation] += 1
            records.extend({
                "row_id": df.iloc[index]["row_id"], "paper_id": papers.iloc[index],
                "seed": split_seed, "fold": fold, "true_label": int(y[index]),
                "predicted_probability": float(probability[local]),
            } for local, index in enumerate(validation))
    if np.any(counts == 0):
        raise RuntimeError("Refit OOF predictions do not cover every training row")
    return sums / counts, pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--champion", choices=["interpolation", "extrapolation"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    candidate, chosen = _candidate(selection, args.champion)
    params = chosen["exact_hyperparameters"]
    df, report = load_data(args.data, require_target=True)
    oof_average, oof_records = _oof(df, candidate, params, args.champion, args.random_state)
    thresholds = optimize_thresholds(df["ignition_binary"].to_numpy(), oof_average)
    oof_records.to_csv(out / "out_of_fold_predictions.csv", index=False)
    final_model = make_model(candidate, params, args.random_state)
    final_model.fit(
        df, df["ignition_binary"].to_numpy(), df["paper_id"].reset_index(drop=True))
    joblib.dump(final_model, out / "model.joblib", compress=3)
    (out / "feature_manifest.json").write_text(
        json.dumps({"data_version": DATA_VERSION, "features": feature_manifest(),
                    "selected_feature_set": candidate["feature_set"]}, indent=2), encoding="utf-8")
    (out / "thresholds.json").write_text(
        json.dumps({
            "thresholds": thresholds,
            "source": f"{args.champion} full-data out-of-fold predictions",
            "warning": "Each threshold optimizes a different operational objective.",
        }, indent=2), encoding="utf-8")
    fingerprint = {
        "dataset_sha256": report["dataset_sha256"], "data_version": DATA_VERSION,
        "training_row_ids_sha256": __import__("hashlib").sha256(
            "\n".join(df["row_id"]).encode()).hexdigest(),
        "training_rows": len(df), "training_papers": int(df["paper_id"].nunique()),
    }
    (out / "training_data_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2), encoding="utf-8")
    card = {
        "artifact_version": "fable-model-v1", "champion": args.champion,
        "model_id": chosen["candidate_id"], "model_family": chosen["model_family"],
        "feature_set": chosen["feature_set"], "hyperparameters": params,
        "weighting_policy": {
            "paper_weight": candidate["paper_weight"],
            "class_weight": candidate["class_weight"],
            "unsupported_fit_weight_strategy": "deterministic weighted resampling",
        },
        "training_rows": len(df), "training_papers": int(df["paper_id"].nunique()),
        "data_fingerprint": fingerprint, "thresholds": thresholds,
        "evaluation_summary": {
            "selected_champion": chosen["metrics"],
            "interpolation": selection["interpolation_champion"]["metrics"],
            "extrapolation": selection["extrapolation_champion"]["metrics"],
        },
        "intended_use": "Binary ignition probability for data matching the documented schema; "
                        "use the champion matching the scientific question.",
        "known_limitations": [
            "Observational database heterogeneity and missingness limit causal interpretation.",
            "Interpolation scores do not establish unseen-paper generalization.",
            "Extrapolation estimates apply to campaigns represented by the source database.",
            "Thresholds are objective-specific and are not universally optimal.",
        ],
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    metadata = {
        "champion": args.champion, "candidate": candidate, "hyperparameters": params,
        "random_state": args.random_state,
        "oof_protocol": ("stratified 5-fold repeated over 3 seeds"
                         if args.champion == "interpolation"
                         else "StratifiedGroupKFold(5) repeated over 3 seeds"),
        "thresholds_derived_without_training_label_prediction": True,
    }
    (out / "refit_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(out), "model_id": chosen["candidate_id"]}, indent=2))


if __name__ == "__main__":
    main()
