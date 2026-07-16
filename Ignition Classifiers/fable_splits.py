"""Generate immutable row-level, grouped-paper, and LOPO split assignments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from fable_common import DATA_VERSION, load_data, write_validation_report

DEFAULT_SEED = 42


def _record(df: pd.DataFrame, protocol: str, seed: int, fold: int,
            train: np.ndarray, test: np.ndarray) -> list[dict]:
    split_id = f"{protocol}__seed-{seed}__fold-{fold}"
    rows = []
    for partition, indices in (("train", train), ("test", test)):
        for idx in indices:
            rows.append({
                "split_id": split_id, "protocol": protocol, "seed": seed, "fold": fold,
                "partition": partition, "row_id": df.iloc[idx]["row_id"],
                "paper_id": df.iloc[idx]["paper_id"],
            })
    return rows


def _validate(assignments: pd.DataFrame, df: pd.DataFrame) -> dict:
    expected = set(df["row_id"])
    failures, checked = [], 0
    for split_id, split in assignments.groupby("split_id", sort=False):
        train = split[split["partition"] == "train"]
        test = split[split["partition"] == "test"]
        if set(train["row_id"]) & set(test["row_id"]):
            failures.append(f"{split_id}: row overlap")
        if set(train["row_id"]) | set(test["row_id"]) != expected:
            failures.append(f"{split_id}: incomplete row coverage")
        protocol = split["protocol"].iloc[0]
        if protocol in {"extrapolation_grouped", "lopo"}:
            overlap = set(train["paper_id"]) & set(test["paper_id"])
            if overlap:
                failures.append(f"{split_id}: paper leakage ({len(overlap)} papers)")
        checked += 1
    if failures:
        raise ValueError("Split integrity validation failed:\n" + "\n".join(failures))
    return {"passed": True, "splits_checked": checked, "failures": []}


def generate(df: pd.DataFrame, n_seeds: int, n_group_folds: int,
             n_row_folds: int, base_seed: int) -> pd.DataFrame:
    y = df["ignition_binary"].to_numpy()
    groups = df["paper_id"].to_numpy()
    rows: list[dict] = []
    indices = np.arange(len(df))
    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        train, test = train_test_split(indices, test_size=.2, stratify=y, random_state=seed)
        rows.extend(_record(df, "interpolation_holdout", seed, 0, train, test))
        row_cv = StratifiedKFold(n_splits=n_row_folds, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(row_cv.split(indices, y)):
            rows.extend(_record(df, "interpolation_stratified", seed, fold, train, test))
        group_cv = StratifiedGroupKFold(
            n_splits=n_group_folds, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(group_cv.split(indices, y, groups)):
            rows.extend(_record(df, "extrapolation_grouped", seed, fold, train, test))
    for fold, paper in enumerate(pd.unique(groups)):
        test = np.flatnonzero(groups == paper)
        train = np.flatnonzero(groups != paper)
        rows.extend(_record(df, "lopo", base_seed, fold, train, test))
    assignments = pd.DataFrame(rows)
    _validate(assignments, df)
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-group-folds", type=int, default=5)
    parser.add_argument("--n-row-folds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if min(args.n_seeds, args.n_group_folds, args.n_row_folds) < 1:
        raise ValueError("Split counts must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, report = load_data(args.data, require_target=True)
    assignments = generate(
        df, args.n_seeds, args.n_group_folds, args.n_row_folds, args.base_seed)
    assignments.to_csv(out / "split_assignments.csv", index=False)
    assignments.to_parquet(out / "split_assignments.parquet", index=False)
    joblib.dump(assignments, out / "split_assignments.joblib", compress=3)
    integrity = _validate(assignments, df)
    protocols = assignments[["split_id", "protocol", "seed", "fold"]].drop_duplicates()
    metadata = {
        "data_version": DATA_VERSION, "dataset_sha256": report["dataset_sha256"],
        "row_count": len(df), "paper_count": int(df["paper_id"].nunique()),
        "label_prevalence": float(df["ignition_binary"].mean()),
        "label_counts": report["label_counts"], "base_seed": args.base_seed,
        "n_seeds": args.n_seeds, "n_group_folds": args.n_group_folds,
        "n_row_folds": args.n_row_folds,
        "protocol_split_counts": protocols["protocol"].value_counts().to_dict(),
        "integrity": integrity,
    }
    (out / "splits_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    write_validation_report(report, out / "data_validation_report.json")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
