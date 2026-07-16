"""Build publication tables and figures exclusively from persisted artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from fable_common import load_data

sns.set_theme(style="whitegrid", context="paper")
COLORS = {"interpolation_stratified": "#0072B2", "extrapolation_grouped": "#D55E00"}
FAMILIES = ["xgboost", "knn", "decision_tree", "mlp", "svm"]


def _save_table(frame: pd.DataFrame, name: str, out: Path) -> None:
    frame.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.md").write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")


def _leaderboard(summary: pd.DataFrame, protocol: str) -> pd.DataFrame:
    columns = ["model_id", "model_family", "feature_set", "roc_auc_mean", "roc_auc_std",
               "pr_auc_mean", "pr_auc_std", "brier_mean", "fold_count"]
    return summary[summary["protocol"] == protocol][columns].sort_values(
        ["roc_auc_mean", "pr_auc_mean"], ascending=False)


def _metric_plot(leaderboards: pd.DataFrame, metric: str, path: Path) -> None:
    order = leaderboards.groupby("model_id")[f"{metric}_mean"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(9, max(4, .28 * len(order))))
    for protocol, frame in leaderboards.groupby("protocol"):
        positions = np.arange(len(order)) + (-.12 if protocol.startswith("interpolation") else .12)
        aligned = frame.set_index("model_id").reindex(order)
        ax.errorbar(aligned[f"{metric}_mean"], positions,
                    xerr=aligned[f"{metric}_std"], fmt="o", capsize=2,
                    label=protocol.replace("_", " "), color=COLORS[protocol])
    ax.set_yticks(np.arange(len(order)), order, fontsize=7)
    ax.set_xlabel(metric.upper().replace("_", "-") + " (fold mean ± SD)")
    ax.legend(title="Protocol")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _family_plot(summary: pd.DataFrame, path: Path) -> None:
    frame = summary[summary["protocol"].isin(COLORS)].copy()
    best = (frame.sort_values("roc_auc_mean", ascending=False)
            .groupby(["protocol", "model_family"], as_index=False).first())
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), sharey=True)
    for axis, family in zip(axes, FAMILIES):
        part = best[best["model_family"] == family].set_index("protocol").reindex(COLORS)
        axis.bar([0, 1], part["roc_auc_mean"], yerr=part["roc_auc_std"],
                 color=[COLORS[x] for x in COLORS], capsize=3)
        axis.set_xticks([0, 1], ["Interpolation", "Extrapolation"], rotation=30, ha="right")
        axis.set_title(family.replace("_", " ").title())
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("Best candidate ROC-AUC (mean ± SD)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _drop_plot(summary: pd.DataFrame, path: Path) -> pd.DataFrame:
    pivot = summary[summary["protocol"].isin(COLORS)].pivot(
        index=["model_id", "model_family", "feature_set"], columns="protocol",
        values="roc_auc_mean").dropna().reset_index()
    pivot["roc_auc_drop"] = (pivot["interpolation_stratified"] -
                             pivot["extrapolation_grouped"])
    pivot = pivot.sort_values("roc_auc_drop")
    fig, ax = plt.subplots(figsize=(8, max(4, .28 * len(pivot))))
    colors = np.where(pivot["roc_auc_drop"] > 0, "#D55E00", "#0072B2")
    ax.barh(pivot["model_id"], pivot["roc_auc_drop"], color=colors)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_xlabel("ROC-AUC interpolation minus extrapolation")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return pivot


def _paired_plot(paired: pd.DataFrame, champion: str, path: Path) -> None:
    frame = paired[(paired["protocol"] == "extrapolation_grouped") &
                   ((paired["model_a"] == champion) | (paired["model_b"] == champion))].copy()
    frame["comparator"] = np.where(frame["model_a"] == champion, frame["model_b"], frame["model_a"])
    frame["champion_delta"] = np.where(
        frame["model_a"] == champion, frame["delta_roc_auc"], -frame["delta_roc_auc"])
    frame = frame.sort_values("champion_delta")
    fig, ax = plt.subplots(figsize=(8, max(3.5, .28 * len(frame))))
    ax.barh(frame["comparator"], frame["champion_delta"],
            color=np.where(frame["champion_delta"] < 0, "#D62728", "#0072B2"))
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_xlabel(f"Paired ROC-AUC delta: {champion} minus comparator")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _calibration(predictions: pd.DataFrame, selections: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for champion, protocol, color in (
        ("interpolation_champion", "interpolation_stratified", "#0072B2"),
        ("extrapolation_champion", "extrapolation_grouped", "#D55E00"),
    ):
        model_id = selections[champion]["candidate_id"]
        frame = predictions[(predictions["model_id"] == model_id) &
                            (predictions["protocol"] == protocol)]
        observed, predicted = calibration_curve(
            frame["true_label"], frame["predicted_probability"], n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, "o-", color=color,
                label=f"{champion.replace('_champion', '')}: {model_id}")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed ignition fraction",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _threshold_table(predictions: pd.DataFrame, selections: dict) -> pd.DataFrame:
    rows = []
    for champion, protocol in (
        ("interpolation_champion", "interpolation_stratified"),
        ("extrapolation_champion", "extrapolation_grouped"),
    ):
        model_id = selections[champion]["candidate_id"]
        frame = predictions[(predictions["model_id"] == model_id) &
                            (predictions["protocol"] == protocol)]
        for policy in ("mcc", "f1", "balanced_accuracy", "youden_j"):
            y, pred = frame["true_label"].to_numpy(), frame[f"prediction_{policy}"].to_numpy()
            tn = int(((y == 0) & (pred == 0)).sum())
            fp = int(((y == 0) & (pred == 1)).sum())
            fn = int(((y == 1) & (pred == 0)).sum())
            tp = int(((y == 1) & (pred == 1)).sum())
            rows.append({
                "champion": champion, "model_id": model_id, "protocol": protocol,
                "threshold_policy": policy, "threshold_mean": frame[f"threshold_{policy}"].mean(),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                "specificity": tn / (tn + fp) if tn + fp else np.nan,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
            })
    return pd.DataFrame(rows)


def _threshold_plot(table: pd.DataFrame, path: Path) -> None:
    plot = table.melt(
        id_vars=["champion", "threshold_policy"], value_vars=["sensitivity", "specificity", "precision"],
        var_name="metric", value_name="value")
    chart = sns.catplot(data=plot, x="threshold_policy", y="value", hue="metric",
                        col="champion", kind="bar", height=4, aspect=1.2,
                        palette=["#0072B2", "#E69F00", "#009E73"])
    chart.set_xticklabels(rotation=25, ha="right")
    chart.set_axis_labels("Threshold objective", "Pooled outer-fold metric")
    chart.figure.tight_layout()
    chart.figure.savefig(path, dpi=300)
    plt.close(chart.figure)


def _explainability(data: pd.DataFrame, evaluation: Path, xgb_id: str, out: Path,
                    random_state: int = 42) -> None:
    index = pd.read_csv(evaluation / "explainability_index.csv")
    row = index[index["model_id"] == xgb_id]
    if row.empty:
        raise RuntimeError(f"No persisted held-out-paper XGBoost model for {xgb_id}")
    row = row.iloc[0]
    model = joblib.load(evaluation / row["model_path"])
    held_out_ids = set(json.loads(row["held_out_row_ids"]))
    held_out = data[data["row_id"].isin(held_out_ids)].copy()
    y = held_out["ignition_binary"].to_numpy()
    baseline = roc_auc_score(y, model.predict_proba(held_out))
    rng, importance = np.random.default_rng(random_state), []
    for feature in model.features:
        drops = []
        for _ in range(20):
            shuffled = held_out.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            drops.append(baseline - roc_auc_score(y, model.predict_proba(shuffled)))
        importance.append({
            "feature": feature, "importance_mean": np.mean(drops),
            "importance_std": np.std(drops, ddof=1), "held_out_roc_auc": baseline,
        })
    importance = pd.DataFrame(importance).sort_values("importance_mean", ascending=False)
    _save_table(importance, "xgboost_permutation_importance", out)
    top = importance.head(20).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#0072B2")
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_xlabel("Held-out-paper ROC-AUC decrease")
    fig.tight_layout()
    fig.savefig(out / "xgboost_permutation_importance.png", dpi=300)
    plt.close(fig)
    try:
        import shap
        transformed = np.asarray(model.preprocessor_.transform(held_out[model.features]), dtype=np.float32)
        names = model.get_feature_names_out()
        all_values = []
        for estimator in model.estimators_:
            values = shap.TreeExplainer(estimator).shap_values(transformed)
            if isinstance(values, list):
                values = values[-1]
            values = np.asarray(values)
            if values.ndim == 3:
                values = values[:, :, -1]
            all_values.append(values)
        values = np.mean(all_values, axis=0)
        ranking = pd.DataFrame({
            "feature": names, "mean_absolute_shap": np.abs(values).mean(axis=0),
        }).sort_values("mean_absolute_shap", ascending=False)
        _save_table(ranking, "xgboost_shap_ranking", out)
        shap.summary_plot(values, transformed, feature_names=names, show=False, max_display=20)
        plt.title(f"{xgb_id}: SHAP on held-out papers")
        plt.tight_layout()
        plt.savefig(out / "xgboost_shap_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close()
        shap.summary_plot(values, transformed, feature_names=names, plot_type="bar",
                          show=False, max_display=20)
        plt.title(f"{xgb_id}: mean absolute SHAP")
        plt.tight_layout()
        plt.savefig(out / "xgboost_shap_bar.png", dpi=300, bbox_inches="tight")
        plt.close()
        status = {"success": True, "model_id": xgb_id, "split_id": row["split_id"],
                  "held_out_rows": len(held_out),
                  "held_out_papers": held_out["paper_id"].nunique()}
        (out / "explainability_status.json").write_text(json.dumps(status, indent=2))
    except Exception as exc:
        status = {"success": False, "model_id": xgb_id, "error": str(exc),
                  "required_action": "Install/fix SHAP; report generation is non-successful."}
        (out / "explainability_status.json").write_text(json.dumps(status, indent=2))
        raise RuntimeError(f"Required SHAP explanation failed: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evaluation, artifacts, out = Path(args.evaluation), Path(args.artifacts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data, data_report = load_data(args.data, require_target=True)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    summary = pd.read_csv(evaluation / "summary_metrics.csv")
    predictions = pd.read_parquet(evaluation / "outer_fold_predictions.parquet")
    paired = pd.read_csv(evaluation / "paired_model_comparisons.csv")
    per_paper = pd.read_csv(evaluation / "per_paper_metrics.csv")
    integrity = json.loads((evaluation / "integrity_checks.json").read_text())
    cards = {
        name: json.loads((artifacts / f"{name}_champion" / "model_card.json").read_text())
        for name in ("interpolation", "extrapolation")
    }
    dataset_table = pd.DataFrame([{
        "rows": data_report["row_count"], "papers": data_report["paper_count"],
        "ignition_rows": data_report["label_counts"].get("1", 0),
        "no_ignition_rows": data_report["label_counts"].get("0", 0),
        "duplicates_removed": data_report["duplicate_count"],
        "dataset_sha256": data_report["dataset_sha256"],
        "evaluation_integrity_passed": integrity["passed"],
    }])
    _save_table(dataset_table, "dataset_integrity_summary", out)
    interpolation = _leaderboard(summary, "interpolation_stratified")
    interpolation_holdout = _leaderboard(summary, "interpolation_holdout")
    extrapolation = _leaderboard(summary, "extrapolation_grouped")
    _save_table(interpolation, "interpolation_leaderboard", out)
    _save_table(interpolation_holdout, "interpolation_holdout_leaderboard", out)
    _save_table(extrapolation, "extrapolation_leaderboard", out)
    both = summary[summary["protocol"].isin(COLORS)]
    _metric_plot(both, "roc_auc", out / "roc_auc_fold_uncertainty.png")
    _metric_plot(both, "pr_auc", out / "pr_auc_fold_uncertainty.png")
    _family_plot(summary, out / "five_model_family_comparison.png")
    drops = _drop_plot(summary, out / "interpolation_extrapolation_drop.png")
    _save_table(drops, "interpolation_extrapolation_drop", out)
    extrap_id = selection["extrapolation_champion"]["candidate_id"]
    _paired_plot(paired, extrap_id, out / "paired_model_deltas.png")
    best_family_ids = (extrapolation.sort_values("roc_auc_mean", ascending=False)
                       .groupby("model_family", as_index=False).first()["model_id"])
    paper_plot = per_paper[(per_paper["protocol"] == "lopo") &
                           (per_paper["model_id"].isin(best_family_ids))].dropna(subset=["roc_auc"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=paper_plot, x="roc_auc", hue="model_family", bins=20,
                 element="step", stat="count", common_norm=False, ax=ax)
    ax.axvline(.5, color="black", linestyle="--", linewidth=.8)
    ax.set_xlabel("Within-paper ROC-AUC under LOPO")
    fig.tight_layout()
    fig.savefig(out / "per_paper_extrapolation_histogram.png", dpi=300)
    plt.close(fig)
    _calibration(predictions, selection, out / "champion_calibration.png")
    thresholds = _threshold_table(predictions, selection)
    _save_table(thresholds, "champion_threshold_performance", out)
    _threshold_plot(thresholds, out / "champion_threshold_performance.png")
    eligible_extrapolation = set(extrapolation["model_id"])
    if selection["extrapolation_champion"]["model_family"] == "xgboost":
        xgb_id = extrap_id
    elif (selection["interpolation_champion"]["model_family"] == "xgboost" and
          selection["interpolation_champion"]["candidate_id"] in eligible_extrapolation):
        xgb_id = selection["interpolation_champion"]["candidate_id"]
    else:
        eligible_xgb = extrapolation[extrapolation["model_family"] == "xgboost"]
        if eligible_xgb.empty:
            status = {
                "success": False,
                "reason": "No eligible XGBoost candidate has held-out-paper evaluation evidence.",
            }
            (out / "explainability_status.json").write_text(json.dumps(status, indent=2))
            raise RuntimeError(status["reason"])
        xgb_id = eligible_xgb.iloc[0]["model_id"]
    _explainability(data, evaluation, xgb_id, out)
    commands = """cd "Ignition Classifiers"
python fable_splits.py --data ../Microgravity_Database_reduced.csv --out results/splits --n-seeds 3 --n-group-folds 5 --n-row-folds 5
python fable_evaluate.py --data ../Microgravity_Database_reduced.csv --splits results/splits --config configs/candidates.yaml --out results/evaluation --search-iterations 40 --inner-group-folds 3
python fable_select.py --evaluation results/evaluation --policy configs/selection_policy.yaml --out results/selection.json
python fable_refit.py --data ../Microgravity_Database_reduced.csv --selection results/selection.json --champion interpolation --out artifacts/interpolation_champion
python fable_refit.py --data ../Microgravity_Database_reduced.csv --selection results/selection.json --champion extrapolation --out artifacts/extrapolation_champion
python fable_report.py --data ../Microgravity_Database_reduced.csv --evaluation results/evaluation --selection results/selection.json --artifacts artifacts --out results/report"""
    readme = f"""# Ignition classification evaluation report

Interpolation evaluates similar row-level conditions; extrapolation holds out entire canonical
papers and is the relevant evidence for transfer to unseen campaigns. Grouped nested tuning,
frozen validation thresholds, persisted outer predictions, and paper-cluster bootstrap intervals
protect the distinction.

## Selected models

- Interpolation: `{selection['interpolation_champion']['candidate_id']}`
- Extrapolation: `{selection['extrapolation_champion']['candidate_id']}`

These champions answer different scientific questions. Fold uncertainty, paired deltas, per-paper
variation, and calibration figures must be considered with point estimates. LOPO is a robustness
analysis, not the sole selection basis. Database heterogeneity, sparse features, campaign effects,
and observational sampling limit causal or universal claims.

## Integrity

Evaluation integrity overall: `{integrity['passed']}`. Failed candidate/protocol combinations are
excluded and documented in `../evaluation/integrity_checks.json`.

## Exact commands

```bash
{commands}
```

Evaluation artifacts are unbiased comparison evidence, not deployable models. Refit artifacts are
deployable models, not unbiased evaluation evidence.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    (out / "report_manifest.json").write_text(json.dumps({
        "interpolation_model_card": cards["interpolation"],
        "extrapolation_model_card": cards["extrapolation"],
        "xgboost_explained": xgb_id,
        "figures": sorted(path.name for path in out.glob("*.png")),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out), "xgboost_explained": xgb_id}, indent=2))


if __name__ == "__main__":
    main()
