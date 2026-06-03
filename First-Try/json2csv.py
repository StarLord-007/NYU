from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List


def load_conditions_from_file(path: Path) -> List[Dict[str, Any]]:
    """
        Load all row objects from an experiments JSON file.

        Supported structures:
    {
      "src\\test_v2.chunks.json": {
                "conditions": [ {...}, {...}, ... ]
            },
            "src\\test_v3.chunks.json": {
                "rows": [ {...}, {...}, ... ]
      },
      "another_file.chunks.json": {
                "conditions": [ ... ]
      }
    }
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Dict[str, Any]] = []
    for source_chunks, payload in data.items():
        if not isinstance(payload, dict):
            continue

        conds = payload.get("conditions")
        if conds is None:
            conds = payload.get("rows", [])

        if not isinstance(conds, list):
            continue

        for cond in conds:
            if not isinstance(cond, dict):
                continue
            row = dict(cond)  # copy
            row["source_chunks"] = source_chunks
            rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write to CSV.")

    # Collect all field names used across rows (keys of ExperimentCondition + source_chunks)
    fieldnames = sorted({key for row in rows for key in row.keys()})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert *_experiments.json (LLM output) into a flat CSV for Excel.\n\n"
            "Example:\n"
            "  python json_to_csv.py huang_2019_experiments.json "
            "--out huang_2019_experiments.csv\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        help="One or more *_experiments.json files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output CSV path. If not set and exactly one JSON file is given, "
            "uses the same basename with .csv."
        ),
    )

    args = parser.parse_args()

    json_paths = [Path(p) for p in args.json_files]
    for p in json_paths:
        if not p.exists():
            raise SystemExit(f"JSON file not found: {p}")

    all_rows: List[Dict[str, Any]] = []
    for p in json_paths:
        rows = load_conditions_from_file(p)
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No rows found in any input JSON file.")

    if args.out:
        out_path = Path(args.out)
    else:
        if len(json_paths) != 1:
            raise SystemExit("When using multiple input files you must specify --out.")
        out_path = json_paths[0].with_suffix(".csv")

    write_csv(all_rows, out_path)
    print(f"Wrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()