"""Run inference with a refitted interpolation or extrapolation champion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from fable_common import load_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CSV using the database schema.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--champion", choices=["interpolation", "extrapolation"],
                        default="extrapolation")
    parser.add_argument("--artifact", help="Override the champion artifact directory.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    artifact = (Path(args.artifact) if args.artifact else
                root / "artifacts" / f"{args.champion}_champion")
    required = ["model.joblib", "model_card.json", "thresholds.json"]
    missing = [name for name in required if not (artifact / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Artifact {artifact} is missing required files: {missing}")
    data, _ = load_data(args.input, require_target=False, deduplicate=False)
    model = joblib.load(artifact / "model.joblib")
    card = json.loads((artifact / "model_card.json").read_text(encoding="utf-8"))
    thresholds = json.loads((artifact / "thresholds.json").read_text(
        encoding="utf-8"))["thresholds"]
    probability = model.predict_proba(data)
    output = pd.DataFrame({
        "row_id": data["row_id"], "paper_id": data["paper_id"],
        "paper_label": data["paper_label"],
        "ignition_probability": probability,
    })
    for name, threshold in thresholds.items():
        output[f"prediction_{name}"] = (probability >= threshold).astype(int)
        output[f"threshold_{name}"] = threshold
    output["champion"] = args.champion
    output["model_id"] = card["model_id"]
    output["model_family"] = card["model_family"]
    output["artifact_version"] = card["artifact_version"]
    output["training_dataset_sha256"] = card["data_fingerprint"]["dataset_sha256"]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(json.dumps({"rows": len(output), "output": str(path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
