"""
fable_audit.py
==============
Forensic audit of the ignition/no-ignition pipeline. Produces the
quantitative evidence behind ``AUDIT_REPORT.md`` and ``LEAKAGE_ANALYSIS.md``.

What this script does
---------------------
1. Reproduces the baseline (xgb_ignition_model_2.py configuration) on the
   latest CSV: stratified hold-out, stratified 5-fold CV, GroupKFold CV.
2. Demonstrates the *grouping contamination* in the previous pipeline:
   grouped CV with the old ``DOI.fillna(Article)`` group key (which suffers
   from Excel-autofilled DOIs + citation-string aliases in the older file)
   versus the canonical physical-paper key.
3. Decomposes the stratified -> grouped AUC collapse:
   * pooled per-fold AUC,
   * Leave-One-Paper-Out (LOPO) per-paper discrimination,
   * within-paper vs between-paper AUC decomposition,
   * identifies which papers drive the collapse and which are OOD.
4. Leakage analysis:
   * paper-identifiability probe (predict paper_id from features),
   * mutual information of each feature with target AND with paper identity,
   * permutation importance under grouped CV,
   * SHAP values of the baseline model,
   * feature clustering (Spearman + hierarchical).

Outputs -> ``Fable/audit_outputs/``

Run::

    python Fable/fable_audit.py --data Microgravity_Database_Latest.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from fable_common import (
    ALL_CATEGORICAL,
    ALL_NUMERIC,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_clean,
)

warnings.filterwarnings("ignore", category=FutureWarning)

RNG = 42

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------------
# Model factory (baseline = v2 script configuration)
# ---------------------------------------------------------------------------

def make_pipeline(scale_pos_weight: float = 1.0) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", ALL_NUMERIC),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ALL_CATEGORICAL,
            ),
        ],
        remainder="drop",
    )
    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=RNG,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", pre), ("model", model)])


def feature_names_out(pipe: Pipeline) -> list[str]:
    names = list(ALL_NUMERIC)
    ohe = pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["oh"]
    for col, cats in zip(ALL_CATEGORICAL, ohe.categories_):
        names.extend(f"{col}={c}" for c in cats)
    return names


# ---------------------------------------------------------------------------
# 1. Baseline reproduction
# ---------------------------------------------------------------------------

def baseline_metrics(df: pd.DataFrame, out: dict) -> None:
    X = df[ALL_NUMERIC + ALL_CATEGORICAL]
    y = df["ignition_binary"].to_numpy()
    groups = df["paper_id"]
    spw = (y == 0).sum() / max((y == 1).sum(), 1)

    print("== Baseline reproduction (v2 config, latest CSV, canonical grouping) ==")
    pipe = make_pipeline(spw)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RNG
    )
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    out["holdout_roc_auc"] = float(roc_auc_score(y_te, p))
    print(f"  stratified 80/20 hold-out ROC-AUC: {out['holdout_roc_auc']:.4f}")

    skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
    s = cross_val_score(make_pipeline(spw), X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    out["stratified_cv_mean"], out["stratified_cv_std"] = float(s.mean()), float(s.std())
    print(f"  stratified 5-fold ROC-AUC: {s.mean():.4f} +/- {s.std():.4f}")

    gkf = GroupKFold(5)
    g = cross_val_score(
        make_pipeline(spw), X, y, cv=gkf, groups=groups, scoring="roc_auc", n_jobs=-1
    )
    out["group_cv_canonical_mean"], out["group_cv_canonical_std"] = float(g.mean()), float(g.std())
    out["group_cv_canonical_folds"] = [float(v) for v in g]
    print(f"  GroupKFold 5-fold (canonical paper) ROC-AUC: {g.mean():.4f} +/- {g.std():.4f}")
    print(f"    per fold: {np.round(g, 3)}")


def duplicate_inflation(data_path: str, out: dict) -> None:
    """Quantify how within-paper duplicate rows inflate random-split CV."""
    print("== Duplicate-row inflation check ==")
    df_dup = load_clean(data_path, dedupe=False)
    df_ded = load_clean(data_path, dedupe=True)
    n_dup = len(df_dup) - len(df_ded)
    out["n_rows_with_duplicates"] = int(len(df_dup))
    out["n_duplicate_rows_removed"] = int(n_dup)
    print(
        f"  labelled rows: {len(df_dup)}; exact (features+label+paper) duplicates: "
        f"{n_dup} ({100 * n_dup / len(df_dup):.1f}%)"
    )
    skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
    for name, frame in [("with duplicates", df_dup), ("deduplicated", df_ded)]:
        Xd = frame[ALL_NUMERIC + ALL_CATEGORICAL]
        yd = frame["ignition_binary"].to_numpy()
        spw = (yd == 0).sum() / max((yd == 1).sum(), 1)
        s = cross_val_score(make_pipeline(spw), Xd, yd, cv=skf, scoring="roc_auc", n_jobs=-1)
        out[f"stratified_cv_{name.replace(' ', '_')}"] = float(s.mean())
        print(f"  stratified 5-fold ROC-AUC ({name}): {s.mean():.4f} +/- {s.std():.4f}")


def grouping_contamination(out: dict) -> None:
    """Quantify how the old group key distorted grouped CV on database_xgb.csv.

    The old script's group key is ``raw DOI .fillna(raw article)``. In
    ``database_xgb.csv`` the DOI column of (at least) the 416-row Konno
    ISO-4589-4 paper was Excel-autofilled (``...103989, ...103990, ...``), so
    that paper alone contributes 243 fake single-row "groups" that GroupKFold
    scatters across all folds. Six further papers appear under two citation
    aliases. Here we re-run grouped CV on the *same rows / same model* with
    (a) the old key and (b) a canonical physical-paper key, where the
    canonical key is recovered by mapping every old key value back to the
    article string it co-occurs with in the raw file and collapsing aliases
    via shared canonical DOIs.
    """
    old_path = Path("database_xgb.csv")
    if not old_path.exists():
        print("  (database_xgb.csv not found; skipping old-file contamination check)")
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import importlib

    from fable_common import canonical_article, canonical_doi

    legacy = importlib.import_module("xgb_ignition_model_2")
    ldf = legacy.load_clean(old_path)
    X = ldf[legacy.NUMERIC_FEATURES + legacy.CATEGORICAL_FEATURES]
    y = ldf["ignition_binary"].to_numpy()
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    pipe = legacy.build_pipeline(scale_pos_weight=spw, random_state=RNG)

    g_old = ldf["source_group"]

    # Build mapping: old key value -> canonical paper.
    raw = pd.read_csv(old_path, skiprows=1)
    raw.columns = [c.strip() if isinstance(c, str) else c for c in raw.columns]
    raw_key = raw["DOI"].fillna(raw["Article"]).astype(str)
    raw_art = raw["Article"].map(canonical_article).fillna("none")
    key_to_article = (
        pd.DataFrame({"k": raw_key, "a": raw_art}).groupby("k")["a"].first()
    )
    # aliases: two article strings sharing one true DOI collapse to the same
    # canonical id via the canonical DOI of the key where available
    key_to_doi = raw_key.map(canonical_doi)
    doi_to_article = (
        pd.DataFrame({"d": key_to_doi, "a": raw_art})
        .dropna(subset=["d"])
        .groupby("d")["a"]
        .first()
    )

    def canon_group(k: str) -> str:
        d = canonical_doi(k)
        if pd.notna(d) and d in doi_to_article.index:
            return str(doi_to_article.loc[d])
        return str(key_to_article.get(k, k))

    g_canon = g_old.map(canon_group)

    print("== Grouping-contamination check on database_xgb.csv ==")
    print(
        f"  rows: {len(ldf)}, old key cardinality: {g_old.nunique()}, "
        f"canonical cardinality: {g_canon.nunique()}"
    )

    gkf = GroupKFold(5)
    s_old = cross_val_score(pipe, X, y, cv=gkf, groups=g_old, scoring="roc_auc", n_jobs=-1)
    out["old_file_group_cv_old_key_mean"] = float(s_old.mean())
    out["old_file_group_cv_old_key_std"] = float(s_old.std())
    print(
        f"  GroupKFold with OLD key ({g_old.nunique()} groups): "
        f"{s_old.mean():.4f} +/- {s_old.std():.4f}  folds={np.round(s_old, 3)}"
    )

    s_canon = cross_val_score(
        pipe, X, y, cv=gkf, groups=g_canon, scoring="roc_auc", n_jobs=-1
    )
    out["old_file_group_cv_canonical_mean"] = float(s_canon.mean())
    out["old_file_group_cv_canonical_std"] = float(s_canon.std())
    print(
        f"  GroupKFold with CANONICAL key ({g_canon.nunique()} groups): "
        f"{s_canon.mean():.4f} +/- {s_canon.std():.4f}  folds={np.round(s_canon, 3)}"
    )

    # how many rows sit in folds that also contain rows of the same physical
    # paper (i.e. directly contaminated test rows under the old key)?
    fold_of_row = np.empty(len(ldf), dtype=int)
    for f, (_, te) in enumerate(gkf.split(X, y, groups=g_old)):
        fold_of_row[te] = f
    contaminated = 0
    for paper, idx in pd.Series(range(len(ldf))).groupby(g_canon.to_numpy()):
        folds = np.unique(fold_of_row[idx.to_numpy()])
        if len(folds) > 1:
            contaminated += len(idx)
    out["old_key_rows_in_split_papers"] = int(contaminated)
    print(
        f"  rows belonging to physical papers that the OLD key split across "
        f"folds: {contaminated} / {len(ldf)} ({100 * contaminated / len(ldf):.1f}%)"
    )


# ---------------------------------------------------------------------------
# 2. Per-paper statistics + imbalance
# ---------------------------------------------------------------------------

def per_paper_stats(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    g = df.groupby("paper_id")
    stats = pd.DataFrame(
        {
            "paper_label": g["paper_label"].first(),
            "n_rows": g.size(),
            "n_ignition": g["ignition_binary"].sum(),
            "ignition_rate": g["ignition_binary"].mean(),
        }
    )
    stats["n_no_ignition"] = stats["n_rows"] - stats["n_ignition"]
    stats["single_class"] = stats["ignition_rate"].isin([0.0, 1.0])
    stats["share_of_dataset_pct"] = 100 * stats["n_rows"] / len(df)

    # feature coverage per paper (how many of the numeric features are non-NaN)
    cov = df.groupby("paper_id")[ALL_NUMERIC].apply(lambda s: s.notna().mean().mean())
    stats["numeric_coverage"] = cov
    stats = stats.sort_values("n_rows", ascending=False)
    stats.to_csv(out_dir / "per_paper_stats.csv")

    print("== Per-paper imbalance ==")
    print(f"  papers: {len(stats)}")
    print(
        f"  single-class papers: {int(stats['single_class'].sum())} "
        f"({int(stats.loc[stats['single_class'], 'n_rows'].sum())} rows, "
        f"all of them 100% ignition)"
    )
    print(
        f"  largest paper: {stats['n_rows'].max()} rows "
        f"({stats['share_of_dataset_pct'].max():.1f}% of data); "
        f"median paper: {stats['n_rows'].median():.0f} rows"
    )
    gini = _gini(stats["n_rows"].to_numpy(dtype=float))
    print(f"  Gini coefficient of paper sizes: {gini:.3f}")
    return stats


def _gini(x: np.ndarray) -> float:
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# ---------------------------------------------------------------------------
# 3. LOPO decomposition of the collapse
# ---------------------------------------------------------------------------

def lopo_decomposition(df: pd.DataFrame, out_dir: Path, out: dict) -> pd.DataFrame:
    """Leave-one-paper-out: fit on all other papers, score the held-out paper."""
    X = df[ALL_NUMERIC + ALL_CATEGORICAL]
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"].to_numpy()
    spw = (y == 0).sum() / max((y == 1).sum(), 1)

    print(f"== Leave-One-Paper-Out decomposition ({pd.unique(papers).size} fits) ==")
    rows = []
    oof = np.full(len(df), np.nan)
    for pid in pd.unique(papers):
        mask = papers == pid
        pipe = make_pipeline(spw)
        pipe.fit(X[~mask], y[~mask])
        p = pipe.predict_proba(X[mask])[:, 1]
        oof[mask] = p
        y_p = y[mask]
        auc = roc_auc_score(y_p, p) if len(np.unique(y_p)) == 2 else np.nan
        rows.append(
            {
                "paper_id": pid,
                "paper_label": df.loc[mask, "paper_label"].iloc[0],
                "n_rows": int(mask.sum()),
                "ignition_rate": float(y_p.mean()),
                "within_paper_auc": auc,
                "mean_pred_ign": float(p[y_p == 1].mean()) if (y_p == 1).any() else np.nan,
                "mean_pred_noign": float(p[y_p == 0].mean()) if (y_p == 0).any() else np.nan,
                "balanced_acc_05": balanced_accuracy_score(y_p, (p >= 0.5).astype(int))
                if len(np.unique(y_p)) == 2
                else np.nan,
            }
        )
    lopo = pd.DataFrame(rows).sort_values("within_paper_auc")
    lopo.to_csv(out_dir / "per_paper_lopo.csv", index=False)
    df_oof = df[["paper_id", "paper_label"]].copy()
    df_oof["y"] = y
    df_oof["oof_pred"] = oof
    df_oof.to_csv(out_dir / "lopo_oof_predictions.csv", index=False)

    # pooled AUC over all LOPO out-of-fold predictions
    pooled = roc_auc_score(y, oof)
    out["lopo_pooled_auc"] = float(pooled)

    two_class = lopo[lopo["within_paper_auc"].notna()]
    out["lopo_within_paper_auc_mean"] = float(two_class["within_paper_auc"].mean())
    out["lopo_within_paper_auc_median"] = float(two_class["within_paper_auc"].median())
    w = two_class["n_rows"].to_numpy(dtype=float)
    out["lopo_within_paper_auc_weighted"] = float(
        np.average(two_class["within_paper_auc"], weights=w)
    )
    out["lopo_papers_below_05"] = int((two_class["within_paper_auc"] < 0.5).sum())
    out["lopo_papers_below_06"] = int((two_class["within_paper_auc"] < 0.6).sum())
    out["lopo_n_two_class_papers"] = int(len(two_class))

    # decomposition: how much of pooled AUC comes from between-paper ranking?
    # Between-paper component: AUC computed on paper-mean predictions vs paper
    # ignition rates (a model that only predicts each paper's base rate).
    paper_mean_pred = df_oof.groupby("paper_id")["oof_pred"].transform("mean")
    out["lopo_between_paper_auc"] = float(roc_auc_score(y, paper_mean_pred))

    print(f"  pooled LOPO AUC (all out-of-paper preds):      {pooled:.4f}")
    print(f"  between-paper component (paper-mean pred):     {out['lopo_between_paper_auc']:.4f}")
    print(f"  mean within-paper AUC (53 two-class papers):   {out['lopo_within_paper_auc_mean']:.4f}")
    print(f"  median within-paper AUC:                       {out['lopo_within_paper_auc_median']:.4f}")
    print(f"  row-weighted within-paper AUC:                 {out['lopo_within_paper_auc_weighted']:.4f}")
    print(f"  papers with within-paper AUC < 0.5 (worse than random): {out['lopo_papers_below_05']}/{len(two_class)}")
    print("  worst 10 papers by within-paper AUC:")
    for _, r in lopo.head(10).iterrows():
        if pd.notna(r["within_paper_auc"]):
            print(
                f"    AUC={r['within_paper_auc']:.3f}  n={r['n_rows']:4d} "
                f"ign_rate={r['ignition_rate']:.2f}  {r['paper_label']}"
            )
    return lopo


# ---------------------------------------------------------------------------
# 4. OOD analysis: is the held-out paper inside the training distribution?
# ---------------------------------------------------------------------------

def ood_analysis(df: pd.DataFrame, lopo: pd.DataFrame, out_dir: Path) -> None:
    """Score each paper's feature-space novelty vs the rest of the corpus."""
    print("== Out-of-distribution analysis ==")
    feats = df[ALL_NUMERIC].copy()
    # robust z-score vs the rest (leave-paper-out median/IQR)
    novelty = {}
    for pid, sub in df.groupby("paper_id"):
        rest = feats.drop(index=sub.index)
        med = rest.median()
        iqr = (rest.quantile(0.75) - rest.quantile(0.25)).replace(0, np.nan)
        z = ((sub[ALL_NUMERIC] - med) / iqr).abs()
        novelty[pid] = float(z.median(axis=1).median())
    nov = pd.Series(novelty, name="median_robust_z")
    merged = lopo.set_index("paper_id").join(nov)
    merged.to_csv(out_dir / "per_paper_ood.csv")
    two = merged[merged["within_paper_auc"].notna()]
    if len(two) > 3:
        from scipy.stats import spearmanr

        rho, pval = spearmanr(two["median_robust_z"], two["within_paper_auc"])
        print(
            f"  Spearman(novelty, within-paper AUC) = {rho:.3f} (p={pval:.3f}) "
            f"over {len(two)} two-class papers"
        )
    print("  most novel papers (median robust z):")
    for pid, r in merged.sort_values("median_robust_z", ascending=False).head(5).iterrows():
        print(
            f"    z={r['median_robust_z']:.2f}  AUC={r['within_paper_auc'] if pd.notna(r['within_paper_auc']) else float('nan'):.3f}  "
            f"{r['paper_label']}"
        )


# ---------------------------------------------------------------------------
# 5. Leakage analysis
# ---------------------------------------------------------------------------

def paper_identifiability_probe(df: pd.DataFrame, out: dict) -> None:
    """Can a classifier identify the *paper* from the features alone?

    If yes, the feature vector is a fingerprint of the experimental campaign
    and a model can exploit it to memorise paper-level base rates.
    """
    print("== Paper-identifiability probe ==")
    X = df[ALL_NUMERIC + ALL_CATEGORICAL]
    papers = df["paper_id"].astype("category").cat.codes.to_numpy()

    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), ALL_NUMERIC),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ALL_CATEGORICAL,
            ),
        ]
    )
    from sklearn.ensemble import RandomForestClassifier

    probe = Pipeline(
        [
            ("pre", pre),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)),
        ]
    )
    skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
    acc = cross_val_score(probe, X, papers, cv=skf, scoring="accuracy", n_jobs=-1)
    out["paper_probe_accuracy_mean"] = float(acc.mean())
    out["paper_probe_chance"] = float(
        (df["paper_id"].value_counts(normalize=True) ** 2).sum()
    )
    print(
        f"  5-fold accuracy predicting paper_id from features: {acc.mean():.3f} "
        f"(chance level ~{out['paper_probe_chance']:.3f}, {df['paper_id'].nunique()} classes)"
    )


def mutual_information(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """MI(feature; ignition) and MI(feature; paper) for every feature."""
    print("== Mutual information ==")
    y = df["ignition_binary"].to_numpy()
    papers = df["paper_id"].astype("category").cat.codes.to_numpy()

    rows = []
    for col in ALL_NUMERIC:
        v = df[col].to_numpy(dtype=float)
        filled = np.where(np.isnan(v), np.nanmedian(v), v).reshape(-1, 1)
        mi_y = mutual_info_classif(filled, y, random_state=RNG, n_neighbors=5)[0]
        mi_p = mutual_info_classif(filled, papers, random_state=RNG, n_neighbors=5)[0]
        rows.append({"feature": col, "type": "numeric", "mi_target": mi_y, "mi_paper": mi_p})
    for col in ALL_CATEGORICAL:
        codes = df[col].astype("category").cat.codes.to_numpy().reshape(-1, 1)
        mi_y = mutual_info_classif(codes, y, discrete_features=True, random_state=RNG)[0]
        mi_p = mutual_info_classif(codes, papers, discrete_features=True, random_state=RNG)[0]
        rows.append({"feature": col, "type": "categorical", "mi_target": mi_y, "mi_paper": mi_p})

    mi = pd.DataFrame(rows)
    h_paper = -np.sum(
        df["paper_id"].value_counts(normalize=True) * np.log(df["paper_id"].value_counts(normalize=True))
    )
    mi["paper_identifiability_frac"] = mi["mi_paper"] / h_paper
    mi["ratio_paper_to_target"] = mi["mi_paper"] / mi["mi_target"].clip(lower=1e-6)
    mi = mi.sort_values("mi_paper", ascending=False)
    mi.to_csv(out_dir / "mutual_information.csv", index=False)
    print("  top paper-identifying features (MI with paper id, nats; H(paper)={:.2f}):".format(h_paper))
    for _, r in mi.head(8).iterrows():
        print(
            f"    {r['feature']:28s} MI(paper)={r['mi_paper']:.3f} "
            f"MI(target)={r['mi_target']:.3f} ratio={r['ratio_paper_to_target']:.1f}"
        )
    return mi


def grouped_permutation_importance(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Permutation importance measured on held-out *papers* (GroupKFold)."""
    print("== Permutation importance under grouped CV ==")
    from sklearn.inspection import permutation_importance

    X = df[ALL_NUMERIC + ALL_CATEGORICAL]
    y = df["ignition_binary"].to_numpy()
    groups = df["paper_id"]
    spw = (y == 0).sum() / max((y == 1).sum(), 1)

    gkf = GroupKFold(5)
    records = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        pipe = make_pipeline(spw)
        pipe.fit(X.iloc[tr], y[tr])
        r = permutation_importance(
            pipe, X.iloc[te], y[te], scoring="roc_auc", n_repeats=10, random_state=RNG, n_jobs=-1
        )
        for i, col in enumerate(X.columns):
            records.append({"fold": fold, "feature": col, "importance": r.importances_mean[i]})
    pi = (
        pd.DataFrame(records)
        .groupby("feature")["importance"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
    )
    pi.to_csv(out_dir / "permutation_importance_grouped.csv")
    print("  top 10 by grouped permutation importance (delta grouped ROC-AUC):")
    print(pi.head(10).round(4).to_string())
    return pi


def shap_analysis(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """SHAP values of the baseline model fitted on all data."""
    print("== SHAP analysis ==")
    import shap

    X = df[ALL_NUMERIC + ALL_CATEGORICAL]
    y = df["ignition_binary"].to_numpy()
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    pipe = make_pipeline(spw)
    pipe.fit(X, y)

    Xt = pipe.named_steps["preprocess"].transform(X).astype(float)
    names = feature_names_out(pipe)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    sv = explainer.shap_values(Xt)

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
    mean_abs.to_csv(out_dir / "shap_mean_abs.csv", header=["mean_abs_shap"])

    fig = plt.figure(figsize=(8, 7))
    shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=22)
    plt.title("SHAP summary - baseline XGBoost (fit on all rows)")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_shap_summary.png", bbox_inches="tight")
    plt.close(fig)

    # aggregate one-hot groups back to their parent feature
    agg = {}
    for n, v in mean_abs.items():
        parent = n.split("=")[0]
        agg[parent] = agg.get(parent, 0.0) + v
    agg = pd.Series(agg).sort_values(ascending=False)
    agg.to_csv(out_dir / "shap_mean_abs_by_parent.csv", header=["mean_abs_shap"])
    print("  top 10 parent features by mean |SHAP|:")
    print(agg.head(10).round(4).to_string())
    return agg


def feature_clustering(df: pd.DataFrame, out_dir: Path) -> None:
    """Spearman-correlation hierarchical clustering of all encoded features."""
    print("== Feature clustering ==")
    from scipy.stats import spearmanr

    X = df[ALL_NUMERIC].copy()
    for c in ALL_CATEGORICAL:
        X[c] = df[c].astype("category").cat.codes
    Xf = X.fillna(X.median(numeric_only=True))
    corr, _ = spearmanr(Xf)
    corr = np.nan_to_num(corr)
    np.fill_diagonal(corr, 1.0)
    corr = (corr + corr.T) / 2
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    link = hierarchy.linkage(squareform(dist, checks=False), method="average")

    fig, ax = plt.subplots(figsize=(9, 7))
    hierarchy.dendrogram(
        link, labels=list(Xf.columns), orientation="right", ax=ax, color_threshold=0.3
    )
    ax.set_xlabel("1 - |Spearman rho|")
    ax.set_title("Feature clustering (clusters below 0.3 are near-redundant)")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_feature_clustering.png", bbox_inches="tight")
    plt.close(fig)

    clusters = hierarchy.fcluster(link, t=0.3, criterion="distance")
    cl = pd.DataFrame({"feature": Xf.columns, "cluster": clusters}).sort_values("cluster")
    cl.to_csv(out_dir / "feature_clusters.csv", index=False)
    sizes = cl["cluster"].value_counts()
    print(f"  {sizes.gt(1).sum()} clusters with >1 feature (redundancy groups)")


# ---------------------------------------------------------------------------
# 6. Figures for the imbalance / collapse story
# ---------------------------------------------------------------------------

def audit_figures(df: pd.DataFrame, stats: pd.DataFrame, lopo: pd.DataFrame, out_dir: Path) -> None:
    # paper size vs ignition rate
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    sorted_sizes = stats["n_rows"].sort_values(ascending=False).to_numpy()
    ax.bar(range(len(sorted_sizes)), sorted_sizes, color="#27598e")
    ax.set_xlabel("paper rank")
    ax.set_ylabel("rows contributed")
    ax.set_title(f"Paper size distribution (Gini = {_gini(sorted_sizes.astype(float)):.2f})")

    ax = axes[1]
    colors = np.where(stats["single_class"], "#c23b22", "#27598e")
    ax.scatter(stats["n_rows"], stats["ignition_rate"], c=colors, s=22, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("paper size (rows, log)")
    ax.set_ylabel("paper ignition rate")
    ax.set_title("Class composition per paper (red = single-class paper)")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_paper_imbalance.png", bbox_inches="tight")
    plt.close(fig)

    # LOPO within-paper AUC distribution
    two = lopo[lopo["within_paper_auc"].notna()].sort_values("within_paper_auc")
    fig, ax = plt.subplots(figsize=(7.5, 8))
    colors = ["#c23b22" if a < 0.5 else "#27598e" for a in two["within_paper_auc"]]
    ax.barh(range(len(two)), two["within_paper_auc"], color=colors)
    ax.set_yticks(range(len(two)))
    ax.set_yticklabels(two["paper_label"], fontsize=6)
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("within-paper ROC-AUC (leave-one-paper-out)")
    ax.set_title("Unseen-paper discrimination, paper by paper\n(red = worse than random)")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_lopo_within_paper_auc.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="Microgravity_Database_Latest.csv")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "audit_outputs"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    df = load_clean(args.data)
    out["n_rows"] = int(len(df))
    out["n_papers"] = int(df["paper_id"].nunique())
    out["class_balance"] = {
        "ignition": int(df["ignition_binary"].sum()),
        "no_ignition": int((1 - df["ignition_binary"]).sum()),
    }
    print(
        f"Loaded {len(df)} labelled rows, {out['n_papers']} canonical papers, "
        f"{out['class_balance']['ignition']} ignition / {out['class_balance']['no_ignition']} no-ignition\n"
    )

    baseline_metrics(df, out)
    print()
    duplicate_inflation(args.data, out)
    print()
    grouping_contamination(out)
    print()
    stats = per_paper_stats(df, out_dir)
    print()
    lopo = lopo_decomposition(df, out_dir, out)
    print()
    ood_analysis(df, lopo, out_dir)
    print()
    paper_identifiability_probe(df, out)
    print()
    mutual_information(df, out_dir)
    print()
    grouped_permutation_importance(df, out_dir)
    print()
    shap_analysis(df, out_dir)
    print()
    feature_clustering(df, out_dir)
    audit_figures(df, stats, lopo, out_dir)

    with open(out_dir / "audit_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nAudit artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
