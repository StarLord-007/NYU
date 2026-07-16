"""Select interpolation and extrapolation champions from evaluation artifacts only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def _rank(summary: pd.DataFrame, intervals: pd.DataFrame, manifest: pd.DataFrame,
          status: dict[tuple[str, str], dict], policy: dict[str, Any],
          selection: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    protocol = selection["primary_protocol"]
    table = summary[summary["protocol"] == protocol].copy()
    rejected = []
    required_lopo = selection.get("mandatory_robustness_protocol")
    valid_ids = []
    for candidate_id in table["model_id"]:
        reasons = []
        primary_status = status.get((candidate_id, protocol), {})
        if policy.get("required_integrity_pass", True) and primary_status.get("errors"):
            reasons.append("failed protocol integrity checks")
        if policy.get("required_complete_folds", True) and not primary_status.get("complete", False):
            reasons.append("could not be evaluated on every required fold")
        if policy.get("required_valid_probabilities", True) and not primary_status.get(
                "valid_probabilities", False):
            reasons.append("produced invalid probabilities")
        if required_lopo:
            robustness = status.get((candidate_id, required_lopo), {})
            if (policy.get("required_integrity_pass", True) and robustness.get("errors")) or (
                    policy.get("required_complete_folds", True) and
                    not robustness.get("complete", False)) or (
                    policy.get("required_valid_probabilities", True) and
                    not robustness.get("valid_probabilities", False)):
                reasons.append(f"missing successful mandatory {required_lopo} robustness evaluation")
        std = table.loc[table["model_id"] == candidate_id, "roc_auc_std"].iloc[0]
        if pd.isna(std) or std > policy["instability"]["maximum_fold_standard_deviation"]:
            reasons.append(
                f"ROC-AUC fold SD {std!r} exceeds "
                f"{policy['instability']['maximum_fold_standard_deviation']}")
        if reasons:
            rejected.append(f"{candidate_id}: " + "; ".join(reasons))
        else:
            valid_ids.append(candidate_id)
    table = table[table["model_id"].isin(valid_ids)].copy()
    if table.empty:
        raise RuntimeError(f"No eligible candidates for {protocol}. Rejections: {rejected}")
    widths = intervals[(intervals["protocol"] == protocol) & (intervals["metric"] == "roc_auc")]
    widths = widths.assign(uncertainty=widths["ci_high"] - widths["ci_low"])
    table = table.merge(widths[["model_id", "uncertainty"]], on="model_id", how="left")
    table = table.merge(
        manifest[["candidate_id", "fixed_params", "paper_weight", "class_weight"]]
        .rename(columns={"candidate_id": "model_id"}), on="model_id", how="left")
    simplicity = {name: index for index, name in enumerate(policy["simplicity_order"])}
    table["simplicity_rank"] = table["model_family"].map(simplicity).fillna(999)
    table["feature_rank"] = table["feature_set"].map({"physics": 0, "all": 1})
    best_auc = table["roc_auc_mean"].max()
    table["within_primary_tie"] = table["roc_auc_mean"] >= best_auc - policy["tie_tolerance"]
    tied = table[table["within_primary_tie"]].copy()
    tie_columns = {
        "higher_pr_auc": ("pr_auc_mean", False),
        "lower_uncertainty": ("uncertainty", True),
        "simpler_model": ("simplicity_rank", True),
        "fewer_features": ("feature_rank", True),
        "physics_only": ("feature_rank", True),
    }
    unknown = set(selection["tie_breakers"]) - set(tie_columns)
    if unknown:
        raise ValueError(f"Unknown selection tie-breakers: {sorted(unknown)}")
    sort_columns = [tie_columns[name][0] for name in selection["tie_breakers"]] + ["model_id"]
    ascending = [tie_columns[name][1] for name in selection["tie_breakers"]] + [True]
    tied = tied.sort_values(sort_columns, ascending=ascending, kind="stable")
    champion = tied.iloc[0]["model_id"]
    table["selected"] = table["model_id"] == champion
    table = table.sort_values(
        ["selected", "roc_auc_mean", "pr_auc_mean"], ascending=[False, False, False])
    return table.where(pd.notna(table), None).to_dict(orient="records"), rejected


def _modal_parameters(parameters: pd.DataFrame, model_id: str, protocol: str) -> dict:
    values = parameters[(parameters["model_id"] == model_id) &
                        (parameters["protocol"] == protocol)]["selected_hyperparameters"]
    if values.empty:
        raise RuntimeError(f"No selected hyperparameters for {model_id}/{protocol}")
    return json.loads(values.value_counts().sort_index().idxmax())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evaluation, out = Path(args.evaluation), Path(args.out)
    policy = yaml.safe_load(Path(args.policy).read_text(encoding="utf-8"))
    summary = pd.read_csv(evaluation / "summary_metrics.csv")
    intervals = pd.read_csv(evaluation / "bootstrap_intervals.csv")
    paired = pd.read_csv(evaluation / "paired_model_comparisons.csv")
    manifest = pd.read_csv(evaluation / "candidate_manifest.csv")
    parameters = pd.read_csv(evaluation / "selected_hyperparameters.csv")
    integrity = json.loads((evaluation / "integrity_checks.json").read_text(encoding="utf-8"))
    status = {(item["candidate_id"], item["protocol"]): item
              for item in integrity["protocol_status"]}
    selections = {}
    markdown = [
        "# Reproducible champion selection", "",
        "> Interpolation and extrapolation champions answer different scientific questions; "
        "neither result substitutes for the other.", "",
    ]
    for name in ("interpolation_champion", "extrapolation_champion"):
        rule = policy["selections"][name]
        ranking, rejected = _rank(summary, intervals, manifest, status, policy, rule)
        winner = next(row for row in ranking if row["selected"])
        manifest_row = manifest[manifest["candidate_id"] == winner["model_id"]].iloc[0].to_dict()
        selected_params = _modal_parameters(parameters, winner["model_id"], rule["primary_protocol"])
        fixed = json.loads(manifest_row.get("fixed_params") or "{}")
        exact_params = {**fixed, **selected_params}
        pair_rows = paired[(paired["protocol"] == rule["primary_protocol"]) &
                           ((paired["model_a"] == winner["model_id"]) |
                            (paired["model_b"] == winner["model_id"]))]
        reason = (
            f"Highest eligible {rule['primary_metric']} within tolerance "
            f"{policy['tie_tolerance']}; tie-breakers applied in declared order: "
            f"{', '.join(rule['tie_breakers'])}. Refit hyperparameters use: "
            f"{policy['refit_hyperparameter_rule']}.")
        selections[name] = {
            "candidate_id": winner["model_id"], "model_family": winner["model_family"],
            "feature_set": winner["feature_set"], "primary_protocol": rule["primary_protocol"],
            "metrics": winner, "exact_hyperparameters": exact_params,
            "candidate_configuration": manifest_row, "reason": reason,
            "ranking": ranking, "rejected_candidates": rejected,
            "paired_comparisons": pair_rows.where(pd.notna(pair_rows), None).to_dict("records"),
        }
        markdown += [
            f"## {name.replace('_', ' ').title()}", "",
            f"Selected `{winner['model_id']}`. {reason}", "",
            f"- ROC-AUC: {winner['roc_auc_mean']:.4f} ± {winner['roc_auc_std']:.4f}",
            f"- PR-AUC: {winner['pr_auc_mean']:.4f} ± {winner['pr_auc_std']:.4f}",
            f"- Protocol: `{rule['primary_protocol']}`",
            f"- Exact hyperparameters: `{json.dumps(exact_params, sort_keys=True)}`", "",
            "| Rank | Candidate | ROC-AUC | PR-AUC | SD | Features |",
            "|---:|---|---:|---:|---:|---|",
        ]
        for rank, row in enumerate(ranking, 1):
            markdown.append(
                f"| {rank} | {row['model_id']} | {row['roc_auc_mean']:.4f} | "
                f"{row['pr_auc_mean']:.4f} | {row['roc_auc_std']:.4f} | "
                f"{row['feature_set']} |")
        if rejected:
            markdown += ["", "Rejected:"] + [f"- {reason}" for reason in rejected]
        markdown.append("")
    payload = _clean_json({
        "policy": policy, **selections,
        "operational_threshold_policy": policy["operational_threshold_policy"],
        "warning": "Interpolation and extrapolation champions answer different scientific questions.",
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({name: selections[name]["candidate_id"] for name in selections}, indent=2))


if __name__ == "__main__":
    main()
