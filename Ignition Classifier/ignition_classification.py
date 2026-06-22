#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ignition_classification.py
==========================

Paper-aware (extrapolation-first) machine-learning study of **Ignition (Yes/No)**
for the microgravity-combustion literature database (``Microgravity_Database.xlsm``).

A note on "regression" vs "classification"
------------------------------------------
The companion study predicts Flame Spread Rate, a continuous quantity, and is a
*regression* problem. This script is requested to live in an ``Ignition Regression``
folder and to mirror that study with the same three model families. However, the
ignition target is **binary** ("Yes"/"No"), so the scientifically correct task is
**classification**, not regression. We therefore keep the requested folder name but
use the *classifier* variants of the three estimators:

  1. DecisionTreeClassifier
  2. GradientBoostingClassifier
  3. KNeighborsClassifier

and report classification metrics (ROC-AUC, PR-AUC, F1, balanced accuracy, MCC, ...)
instead of regression metrics. Everything else -- the extrapolation-first,
paper-aware methodology -- is identical to the FSR study.

Scientific motivation (identical to the FSR study)
--------------------------------------------------
The database is a *literature aggregation*: one paper / campaign / rig contributes
many correlated rows. A plain random split leaks paper-specific information into the
test set and inflates the metrics. The scientifically meaningful question is:

    "For a brand-new paper / campaign / rig the model has never seen, can we predict
     whether a sample will ignite?"

That is an **extrapolation** problem, so we keep *all rows of a paper together*
(entirely train OR entirely test) using ``GroupShuffleSplit`` (hold-out) and
``GroupKFold`` (CV + hyper-parameter tuning), keyed on the paper identifier. All
tuning is driven by GroupKFold so the selected models generalise to *unseen papers*.

Features available for ignition prediction
------------------------------------------
The ignition *stimulus* is known before we observe the outcome, so the ignition
method / power / time are valid predictors. By contrast, FSR, flame length, heat
release rate and smoke are *post-ignition outcomes* and are removed as leakage.

Outputs (written to ``results/`` next to this script)
-----------------------------------------------------
  * confusion-matrix, ROC-curve and precision-recall-curve plots per model
  * per-paper performance distributions (accuracy / F1 / ROC-AUC + boxplot)
  * feature-importance plots (Decision Tree & Gradient Boosting)
  * permutation-importance bar chart + ranked table (best group-aware model)
  * SHAP summary + bar plots (best group-aware model)
  * ``model_comparison.csv`` and ``generalization_gap.csv``
  * ``metrics.json`` and per-paper CSVs
  * ``best_decision_tree.joblib`` / ``best_gradient_boosting.joblib`` / ``best_knn.joblib``

Run
---
    python "Ignition Regression/ignition_classification.py"
    python "Ignition Regression/ignition_classification.py" --no-shap --n-iter 60

Reproducibility: random_state = 42 wherever a seed is accepted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Headless backend must be set before importing pyplot (servers / CI / no display).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import joblib

# A single global seed reused everywhere for reproducibility.
RANDOM_STATE = 42

# Directory that contains this script -- used so default input/output paths do not
# depend on the current working directory (runs the same from root or this folder).
SCRIPT_DIR = Path(__file__).resolve().parent

# Class label convention: positive class = ignition occurred ("Yes" -> 1).
POS_LABEL = 1
CLASS_NAMES = ["No ignition", "Ignition"]

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")


# =============================================================================
# 0. Small, dependency-free helpers
# =============================================================================

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _clean_text(x):
    """Normalise a raw cell to a tidy string, or NaN for placeholder/empties."""
    if pd.isna(x):
        return np.nan
    if not isinstance(x, str):
        return x
    s = x.replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s if s.lower() not in {"", "-", "--", "n/a", "na", "nan", "none"} else np.nan


def _first_number(x):
    """Extract the first numeric value from a possibly unit-laden cell ("94 W")."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).replace(",", ".").replace("\u2212", "-")
    m = _NUM_RE.search(s)
    return float(m.group(0)) if m else np.nan


def _normalize_yes_no(x):
    """Map assorted yes/no spellings to 1/0; everything else -> NaN."""
    s = _clean_text(x)
    if pd.isna(s):
        return np.nan
    l = str(s).lower().strip()
    if l in {"yes", "y", "1", "true", "ignition", "ignited"}:
        return 1
    if l in {"no", "n", "0", "false", "no ignition", "not ignited"}:
        return 0
    return np.nan


def _slug(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()


def _print_header(title: str) -> None:
    line = "=" * 78
    print("\n" + line)
    print(title)
    print(line)


def locate_data_file(user_path: str) -> Path:
    """Resolve the database path robustly, independent of the cwd."""
    given = Path(user_path)
    if given.exists():
        return given
    name = given.name
    candidates = [SCRIPT_DIR / name] + [p / name for p in SCRIPT_DIR.parents]
    for cand in candidates:
        if cand.exists():
            return cand
    return given


# =============================================================================
# 1. Data loading (robust to the two-row "section / field" Excel header)
# =============================================================================

def load_database(path: Path) -> pd.DataFrame:
    """Load the database into a flat, single-header DataFrame (see FSR script)."""
    read_kwargs = {}
    if path.suffix.lower() in {".xlsm", ".xlsx"}:
        read_kwargs["engine"] = "openpyxl"

    def _flatten_two_level(df: pd.DataFrame) -> pd.DataFrame:
        flat = []
        for top, sub in df.columns:
            sub_s, top_s = str(sub), str(top)
            name = top_s if (pd.isna(sub) or sub_s.startswith("Unnamed")) else sub_s
            flat.append(name.strip())
        df = df.copy()
        df.columns = flat
        return df

    try:
        xls = pd.ExcelFile(path, **read_kwargs)
        candidate_sheets = list(xls.sheet_names)
    except Exception:
        candidate_sheets = [0]

    best_df, best_score = None, -1
    for sheet in candidate_sheets:
        for header in ([0, 1], 0):
            try:
                df = pd.read_excel(path, sheet_name=sheet, header=header, **read_kwargs)
            except Exception:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df = _flatten_two_level(df)
            df.columns = [str(c).strip() for c in df.columns]
            has_target = any(_looks_like_target(c) for c in df.columns)
            score = len(df) * df.shape[1] + (10_000_000 if has_target else 0)
            if score > best_score:
                best_score, best_df = score, df

    if best_df is None:
        raise RuntimeError(f"Could not read any usable sheet from {path}")

    best_df = best_df.loc[:, ~best_df.columns.duplicated()]
    best_df = best_df.dropna(axis=1, how="all")
    for c in best_df.columns:
        if best_df[c].dtype == object:
            best_df[c] = best_df[c].map(_clean_text)
    return best_df.reset_index(drop=True)


# =============================================================================
# 2. Automatic column-role detection
# =============================================================================

# The ignition *outcome* column is named like "Ignition (Yes/No)". Note that the
# database also has "Ignition method/power/time", which are *inputs*, so target
# detection must additionally require the column to be (nearly) binary yes/no.
def _looks_like_target(col: str) -> bool:
    c = str(col).lower()
    if "ignition" not in c:
        return False
    return ("yes" in c or "no" in c or "y/n" in c or "binary" in c or "(y" in c)


def detect_target_column(df: pd.DataFrame) -> str:
    """Find the binary Ignition (Yes/No) outcome column automatically.

    Among columns whose name suggests an ignition yes/no flag, choose the one
    whose values most cleanly map to a binary {0,1} encoding. If the name-based
    search fails, fall back to *any* column that is overwhelmingly yes/no valued.
    """
    named = [c for c in df.columns if _looks_like_target(c)]
    candidates = named if named else list(df.columns)
    best, best_frac = None, -1.0
    for c in candidates:
        non_null = df[c].notna().sum()
        if non_null == 0:
            continue
        frac = df[c].map(_normalize_yes_no).notna().sum() / non_null
        # Require a genuinely binary column.
        if frac >= 0.8 and frac > best_frac:
            best, best_frac = c, frac
    if best is None:
        raise RuntimeError(
            "No binary Ignition (Yes/No) column found. Available columns: "
            f"{list(df.columns)}"
        )
    return best


_GROUP_KEYWORDS = [
    "article", "paper", "mla", "citation", "doi", "publication", "reference", "author",
]


def detect_group_column(df: pd.DataFrame) -> str:
    """Find the per-paper grouping column (see FSR study for the rationale)."""
    n = len(df)
    scored = []
    for c in df.columns:
        cl = str(c).lower()
        for rank, kw in enumerate(_GROUP_KEYWORDS):
            if kw in cl:
                non_null = df[c].notna().sum()
                nunique = df[c].nunique(dropna=True)
                if nunique >= max(2, 0.9 * n):
                    continue
                completeness = non_null / max(n, 1)
                scored.append((rank, -completeness, c))
                break
    if not scored:
        raise RuntimeError(
            f"No paper/grouping column found. Keywords: {_GROUP_KEYWORDS}. "
            f"Columns: {list(df.columns)}"
        )
    scored.sort()
    return scored[0][2]


# Post-ignition outcomes -> leakage. CRITICALLY, "ignition" is NOT a leakage
# keyword here (it is the target / a valid stimulus input); only the downstream
# observations recorded *after* ignition are leakage.
_LEAKAGE_KEYWORDS = [
    "fsr", "flame spread", "spread rate",
    "flame length",
    "hrr", "heat release",
    "smoke", "aerosol", "areosol",
    "burn", "extinction", "extinguish",
]
_NOTE_KEYWORDS = ["info", "note", "comment", "remark"]


def detect_leakage_columns(df: pd.DataFrame, target: str, group_col: str) -> list[str]:
    """Columns to drop before modelling (besides target & group).

    Removes post-ignition outcome columns (FSR / flame length / HRR / smoke),
    free-text notes, and paper-identity fingerprints (authors / DOI / article).
    The ignition stimulus columns (method / power / time) are deliberately kept
    as predictors.
    """
    leak = []
    for c in df.columns:
        if c in (target, group_col):
            continue
        cl = str(c).lower()
        if any(k in cl for k in _LEAKAGE_KEYWORDS):
            leak.append(c)
        elif any(k in cl for k in _NOTE_KEYWORDS):
            leak.append(c)
        elif any(k in cl for k in _GROUP_KEYWORDS):
            leak.append(c)
    seen = set()
    return [c for c in df.columns if c in leak and not (c in seen or seen.add(c))]


def detect_feature_types(
    df: pd.DataFrame, feature_cols: list[str], numeric_frac_threshold: float = 0.60
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Split features into numeric vs categorical automatically (unit-string aware)."""
    out = df.copy()
    numeric_cols, categorical_cols = [], []
    for c in feature_cols:
        series = out[c]
        non_null = series.notna().sum()
        if non_null == 0:
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(c)
            continue
        parsed = series.map(_first_number)
        if parsed.notna().sum() / non_null >= numeric_frac_threshold:
            out[c] = parsed
            numeric_cols.append(c)
        else:
            out[c] = series.map(_clean_text)
            categorical_cols.append(c)
    return numeric_cols, categorical_cols, out


# =============================================================================
# 3. Metrics
# =============================================================================

def classification_metrics(y_true, y_pred, y_proba) -> dict:
    """Full battery of binary-classification metrics.

    * Accuracy           -- overall fraction correct (can mislead under imbalance).
    * Balanced Accuracy  -- mean of per-class recall; robust to the 76/24 imbalance.
    * Precision / Recall -- positive-class (ignition) quality and coverage.
    * F1                 -- harmonic mean of precision and recall.
    * ROC-AUC            -- threshold-independent ranking quality.
    * PR-AUC             -- average precision; the right summary under imbalance.
    * MCC                -- Matthews correlation; balanced single-number score.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = len(y_true)
    if n == 0:
        keys = ["Accuracy", "BalancedAcc", "Precision", "Recall", "F1",
                "ROC_AUC", "PR_AUC", "MCC", "n", "n_pos"]
        return {k: np.nan for k in keys}

    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "BalancedAcc": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "n": int(n),
        "n_pos": int(y_true.sum()),
    }
    # ROC-AUC / PR-AUC need both classes present and a probability score.
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["ROC_AUC"] = float(roc_auc_score(y_true, y_proba))
        out["PR_AUC"] = float(average_precision_score(y_true, y_proba))
    else:
        out["ROC_AUC"] = np.nan
        out["PR_AUC"] = np.nan
    return out


def _proba_positive(estimator, X) -> np.ndarray:
    """Return P(ignition) using predict_proba, robust to class ordering."""
    proba = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    pos_idx = classes.index(POS_LABEL) if POS_LABEL in classes else 1
    return proba[:, pos_idx]


# =============================================================================
# 4. Preprocessing + models
# =============================================================================

def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """Shared preprocessing for every model (identical philosophy to FSR study)."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def get_models_and_spaces() -> dict:
    """Three classifiers + RandomizedSearchCV search spaces.

    Ranges favour regularisation (the main risk for extrapolation). The Decision
    Tree also tunes ``class_weight`` to cope with the ignition class imbalance;
    Gradient Boosting and KNN have no native class-weight knob, so we rely on
    ROC-AUC tuning and report balanced metrics for a fair read under imbalance.
    """
    return {
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "param_dist": {
                "model__max_depth": [3, 4, 5, 6, 8, 10, 12, None],
                "model__min_samples_split": [2, 5, 10, 20, 40],
                "model__min_samples_leaf": [1, 2, 4, 8, 16, 32],
                "model__max_features": ["sqrt", "log2", 0.5, 0.8, 1.0, None],
                "model__class_weight": [None, "balanced"],
            },
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_dist": {
                "model__n_estimators": [100, 200, 300, 500, 800],
                "model__learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 4, 5],
                "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "model__min_samples_leaf": [1, 2, 4, 8, 16, 32],
            },
        },
        "KNN": {
            "estimator": KNeighborsClassifier(),
            "param_dist": {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        },
    }


def get_output_feature_names(preprocessor, numeric_cols, categorical_cols) -> list[str]:
    """Recover human-readable feature names after preprocessing."""
    try:
        names = list(preprocessor.get_feature_names_out())
        return [re.sub(r"^(num|cat)__", "", n) for n in names]
    except Exception:
        names = list(numeric_cols)
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        for col, cats in zip(categorical_cols, ohe.categories_):
            names.extend(f"{col}={c}" for c in cats)
        return names


# =============================================================================
# 5. Plotting helpers
# =============================================================================

def plot_confusion(y_true, y_pred, title, path):
    """Confusion matrix heatmap with counts."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, y_proba, title, path):
    """ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title); ax.legend(loc="lower right"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_pr(y_true, y_proba, title, path):
    """Precision-recall curve with average-precision annotation."""
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    base = float(np.mean(y_true))  # positive prevalence = no-skill baseline
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
    ax.axhline(base, color="k", ls="--", alpha=0.4, label=f"baseline = {base:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend(loc="lower left"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_importance_bar(names, values, title, path, top_n=20):
    """Horizontal bar chart of the top-N most important features."""
    order = np.argsort(values)[::-1][:top_n]
    sel_names = [names[i] for i in order]
    sel_vals = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(sel_names))))
    ax.barh(range(len(sel_names)), sel_vals, color="seagreen", edgecolor="k")
    ax.set_yticks(range(len(sel_names))); ax.set_yticklabels(sel_names, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("Importance"); ax.set_title(title)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# =============================================================================
# 6. Per-paper generalisation analysis
# =============================================================================

def per_paper_analysis(y_true, y_pred, y_proba, groups, out_dir: Path, model_name: str):
    """Score each held-out paper separately; average metrics hide per-paper failures."""
    df = pd.DataFrame({
        "y_true": np.asarray(y_true).astype(int),
        "y_pred": np.asarray(y_pred).astype(int),
        "y_proba": np.asarray(y_proba),
        "paper": np.asarray(groups),
    })
    rows = []
    for paper, g in df.groupby("paper"):
        m = classification_metrics(g["y_true"].values, g["y_pred"].values, g["y_proba"].values)
        rows.append({"paper": paper, **m})
    per_paper = pd.DataFrame(rows).sort_values("Accuracy").reset_index(drop=True)

    slug = _slug(model_name)
    per_paper.to_csv(out_dir / f"per_paper_metrics_{slug}.csv", index=False)

    # 1+2) Accuracy & F1 distributions across papers.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(per_paper["Accuracy"].dropna(), bins=20, color="indianred",
                 edgecolor="k", alpha=0.8)
    axes[0].set_title("Per-paper accuracy distribution")
    axes[0].set_xlabel("Accuracy"); axes[0].set_ylabel("Number of papers")
    axes[1].hist(per_paper["F1"].dropna(), bins=20, color="slateblue",
                 edgecolor="k", alpha=0.8)
    axes[1].set_title("Per-paper F1 distribution")
    axes[1].set_xlabel("F1"); axes[1].set_ylabel("Number of papers")
    fig.suptitle(f"{model_name} -- per-paper performance (unseen papers)")
    fig.tight_layout()
    fig.savefig(out_dir / f"per_paper_acc_f1_hist_{slug}.png", dpi=150)
    plt.close(fig)

    # 3) Per-paper ROC-AUC histogram (only papers with both classes present).
    auc_vals = per_paper["ROC_AUC"].dropna()
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    if len(auc_vals):
        ax.hist(auc_vals, bins=20, color="darkgreen", edgecolor="k", alpha=0.8)
    ax.set_title(f"{model_name} -- per-paper ROC-AUC distribution")
    ax.set_xlabel("ROC-AUC on a single held-out paper")
    ax.set_ylabel("Number of papers (both classes present)")
    fig.tight_layout()
    fig.savefig(out_dir / f"per_paper_auc_hist_{slug}.png", dpi=150)
    plt.close(fig)

    # 4) Boxplot of per-paper accuracy.
    fig, ax = plt.subplots(figsize=(4.5, 5))
    ax.boxplot(per_paper["Accuracy"].dropna(), patch_artist=True,
               boxprops=dict(facecolor="lightcoral"))
    ax.set_ylabel("Per-paper accuracy")
    ax.set_title(f"{model_name}\nper-paper accuracy boxplot")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / f"per_paper_accuracy_boxplot_{slug}.png", dpi=150)
    plt.close(fig)

    return per_paper


# =============================================================================
# 7. SHAP analysis
# =============================================================================

def run_shap(best_pipe, X_train, X_test, numeric_cols, categorical_cols,
             out_dir: Path, model_name: str):
    """SHAP for the best group-aware classifier (explains P(ignition))."""
    try:
        import shap
    except Exception as exc:
        print(f"  [SHAP] shap not available ({exc}); skipping SHAP analysis.")
        return

    pre = best_pipe.named_steps["preprocess"]
    model = best_pipe.named_steps["model"]
    feat_names = get_output_feature_names(pre, numeric_cols, categorical_cols)

    Xt_train = pre.transform(X_train)
    Xt_test = pre.transform(X_test)

    rng = np.random.RandomState(RANDOM_STATE)
    n_explain = min(300, Xt_test.shape[0])
    idx = rng.choice(Xt_test.shape[0], size=n_explain, replace=False)
    Xt_sample = Xt_test[idx]

    try:
        if isinstance(model, (DecisionTreeClassifier, GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(Xt_sample)
        else:
            n_bg = min(100, Xt_train.shape[0])
            bg = shap.sample(Xt_train, n_bg, random_state=RANDOM_STATE)
            explainer = shap.KernelExplainer(lambda d: model.predict_proba(d)[:, 1], bg)
            shap_values = explainer.shap_values(Xt_sample, nsamples=100)
    except Exception as exc:
        print(f"  [SHAP] explainer failed ({exc}); skipping SHAP analysis.")
        return

    # For some explainers / sklearn versions the positive class comes back as a
    # list [class0, class1] or a 3-D array; reduce to the positive-class matrix.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    slug = _slug(model_name)

    plt.figure()
    shap.summary_plot(shap_values, Xt_sample, feature_names=feat_names,
                      show=False, max_display=20)
    plt.title(f"SHAP summary -- {model_name} (P(ignition))")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_summary_{slug}.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, Xt_sample, feature_names=feat_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP mean(|value|) -- {model_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_bar_{slug}.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_rank = (
        pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_rank.to_csv(out_dir / f"shap_ranking_{slug}.csv", index=False)

    print(f"\n  SHAP top drivers of Ignition ({model_name}):")
    for i, row in shap_rank.head(10).iterrows():
        print(f"    {i + 1:2d}. {row['feature']:<45s} mean|SHAP| = {row['mean_abs_shap']:.4f}")
    print(
        "  Interpretation: features at the top move the model's ignition\n"
        "  probability the most. Physically, ignition in microgravity is governed\n"
        "  primarily by oxidiser availability (oxygen concentration, pressure),\n"
        "  the ignition stimulus (power / time / method), flow velocity, and the\n"
        "  fuel material / geometry (thermal inertia)."
    )


# =============================================================================
# 8. Main driver
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="Microgravity_Database.xlsm",
                        help="Path to the Excel database (auto-located if missing).")
    parser.add_argument("--out", default=str(SCRIPT_DIR / "results"),
                        help="Output directory (defaults to 'results' next to this script).")
    parser.add_argument("--n-iter", type=int, default=40,
                        help="RandomizedSearchCV iterations per model.")
    parser.add_argument("--cv-splits", type=int, default=5,
                        help="Number of GroupKFold splits for tuning.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test fraction for both split strategies.")
    parser.add_argument("--no-shap", action="store_true", help="Disable SHAP analysis.")
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    _print_header("STEP 1  |  LOAD DATABASE")
    data_path = locate_data_file(args.data)
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found: {data_path.resolve()}")
    df_raw = load_database(data_path)
    n_rows_initial = len(df_raw)
    print(f"Loaded '{data_path}' -> {n_rows_initial} rows x {df_raw.shape[1]} columns.")

    # ---------------------------------------------------------------- target
    _print_header("STEP 2  |  DETECT TARGET (Ignition Yes/No)")
    target_col = detect_target_column(df_raw)
    print(f"Detected target column: '{target_col}'")

    df = df_raw.copy()
    df[target_col] = df[target_col].map(_normalize_yes_no)
    n_before = len(df)
    df = df[df[target_col].notna()].copy().reset_index(drop=True)
    df[target_col] = df[target_col].astype(int)
    n_after = len(df)
    pos = int(df[target_col].sum())
    neg = n_after - pos
    print(f"Initial dataset size (all rows)            : {n_before}")
    print(f"Final dataset size (rows with valid label) : {n_after}")
    print(f"Rows removed due to missing/invalid target : {n_before - n_after}")
    print(f"Class balance -> Ignition(1)={pos} ({pos / n_after:.1%})  "
          f"No-ignition(0)={neg} ({neg / n_after:.1%})")

    # ------------------------------------------------------------- paper id
    _print_header("STEP 3  |  DETECT PAPER / GROUPING IDENTIFIER")
    group_col = detect_group_column(df)
    df[group_col] = df[group_col].fillna("UNKNOWN_PAPER").astype(str)
    print(f"Detected paper/grouping column: '{group_col}'")
    groups_all = df[group_col]
    spp = groups_all.value_counts()
    print(f"Number of unique papers          : {groups_all.nunique()}")
    print(f"Mean samples per paper           : {spp.mean():.2f}")
    print(f"Median samples per paper         : {spp.median():.1f}")
    print(f"Maximum samples per paper        : {spp.max()}")
    print(f"Minimum samples per paper        : {spp.min()}")
    print("Samples-per-paper describe():")
    print(spp.describe().to_string())

    # ----------------------------------------------------------- leakage drop
    _print_header("STEP 4  |  LEAKAGE PREVENTION (remove unsafe columns)")
    leak_cols = detect_leakage_columns(df, target_col, group_col)
    print("Removed before modelling (post-ignition outcomes, notes, paper")
    print("fingerprints). NOTE: ignition method/power/time are KEPT as valid")
    print("pre-outcome stimulus predictors.")
    for c in leak_cols:
        print(f"   - {c}")
    print(f"   - {target_col}   (target, held out separately)")
    print(f"   - {group_col}   (grouping key, held out separately)")
    feature_cols = [c for c in df.columns if c not in set(leak_cols) | {target_col, group_col}]

    # ------------------------------------------------- numeric/categorical
    _print_header("STEP 5  |  AUTOMATIC FEATURE-TYPE DETECTION")
    numeric_cols, categorical_cols, df_feat = detect_feature_types(df, feature_cols)
    df_feat[target_col] = df[target_col].values
    df_feat[group_col] = df[group_col].values
    print(f"Total rows                       : {len(df_feat)}")
    print(f"Total feature columns            : {len(numeric_cols) + len(categorical_cols)}")
    print(f"Numerical features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    X = df_feat[numeric_cols + categorical_cols].copy()
    y = df_feat[target_col].astype(int).values
    groups = df_feat[group_col].values

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # ---------------------------------------------------------- the splits
    _print_header("STEP 6  |  BUILD THE TWO EVALUATION SPLITS")
    # Strategy A: random split (stratified to preserve class balance).
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[A] Random split        : train={len(Xr_tr)}  test={len(Xr_te)}")

    # Strategy B: group split (extrapolation / PRIMARY).
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=RANDOM_STATE)
    g_tr_idx, g_te_idx = next(gss.split(X, y, groups=groups))
    Xg_tr, Xg_te = X.iloc[g_tr_idx], X.iloc[g_te_idx]
    yg_tr, yg_te = y[g_tr_idx], y[g_te_idx]
    groups_tr, groups_te = groups[g_tr_idx], groups[g_te_idx]
    n_train_papers = pd.Series(groups_tr).nunique()
    n_test_papers = pd.Series(groups_te).nunique()
    assert set(groups_tr).isdisjoint(set(groups_te)), "Group leakage detected!"
    print(f"[B] Group/paper split   : train={len(Xg_tr)} ({n_train_papers} papers)  "
          f"test={len(Xg_te)} ({n_test_papers} papers)  -- no shared papers.")
    print(f"    Test-set class balance: Ignition={int(yg_te.sum())}/{len(yg_te)} "
          f"({yg_te.mean():.1%})")

    n_cv = max(2, min(args.cv_splits, n_train_papers))
    if n_cv < args.cv_splits:
        print(f"    (reduced GroupKFold splits to {n_cv} to match #train papers)")

    # ------------------------------------------------- tune + evaluate models
    models = get_models_and_spaces()
    group_cv = GroupKFold(n_splits=n_cv)

    comparison_rows, gap_rows = [], []
    fitted_group_models, best_params_store, cv_auc_store = {}, {}, {}
    METRIC_KEYS = ["Accuracy", "BalancedAcc", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "MCC"]

    for name, spec in models.items():
        _print_header(f"STEP 7  |  TUNE + EVALUATE: {name}")
        pipe = Pipeline([("preprocess", preprocessor), ("model", spec["estimator"])])

        # Tuning ALWAYS via GroupKFold on the group-train set; scoring = ROC-AUC
        # (threshold-independent and robust to the ignition class imbalance).
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=spec["param_dist"],
            n_iter=args.n_iter,
            scoring="roc_auc",
            cv=group_cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            error_score=np.nan,
        )
        search.fit(Xg_tr, yg_tr, groups=groups_tr)
        best_cv_auc = float(search.best_score_)
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        best_params_store[name] = best_params
        cv_auc_store[name] = best_cv_auc
        print(f"Best GroupKFold CV ROC-AUC (extrapolation): {best_cv_auc:.4f}")
        print(f"Best hyper-parameters                     : {best_params}")

        # Group-aware evaluation (PRIMARY) on unseen papers.
        group_pipe = search.best_estimator_
        yg_proba = _proba_positive(group_pipe, Xg_te)
        yg_pred = group_pipe.predict(Xg_te)
        m_group = classification_metrics(yg_te, yg_pred, yg_proba)
        fitted_group_models[name] = group_pipe
        print(f"[GROUP/UNSEEN-PAPERS] Acc={m_group['Accuracy']:.3f}  "
              f"BalAcc={m_group['BalancedAcc']:.3f}  F1={m_group['F1']:.3f}  "
              f"ROC-AUC={m_group['ROC_AUC']:.3f}  PR-AUC={m_group['PR_AUC']:.3f}  "
              f"MCC={m_group['MCC']:.3f}")

        # Random-split evaluation (BASELINE) with the same chosen hyper-parameters.
        random_pipe = Pipeline([
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", spec["estimator"].__class__(
                **{**spec["estimator"].get_params(), **best_params})),
        ])
        random_pipe.fit(Xr_tr, yr_tr)
        yr_proba = _proba_positive(random_pipe, Xr_te)
        yr_pred = random_pipe.predict(Xr_te)
        m_random = classification_metrics(yr_te, yr_pred, yr_proba)
        print(f"[RANDOM/BASELINE]     Acc={m_random['Accuracy']:.3f}  "
              f"BalAcc={m_random['BalancedAcc']:.3f}  F1={m_random['F1']:.3f}  "
              f"ROC-AUC={m_random['ROC_AUC']:.3f}  PR-AUC={m_random['PR_AUC']:.3f}  "
              f"MCC={m_random['MCC']:.3f}")

        comparison_rows.append({"Model": name, "Validation Strategy": "Random Split",
                                **{k: m_random[k] for k in METRIC_KEYS}})
        comparison_rows.append({"Model": name, "Validation Strategy": "Group-Aware",
                                **{k: m_group[k] for k in METRIC_KEYS}})
        # Gap on ROC-AUC: higher AUC is better, so gap = random - group; a large
        # positive gap means the model overfits to paper-specific signal.
        gap_rows.append({
            "Model": name,
            "Random ROC-AUC": m_random["ROC_AUC"],
            "Group ROC-AUC": m_group["ROC_AUC"],
            "Generalization Gap": m_random["ROC_AUC"] - m_group["ROC_AUC"],
        })

        # Per-model diagnostic plots on the PRIMARY group split.
        slug = _slug(name)
        plot_confusion(yg_te, yg_pred, f"{name} -- Confusion matrix (unseen papers)",
                       out_dir / f"confusion_matrix_{slug}.png")
        plot_roc(yg_te, yg_proba, f"{name} -- ROC curve (unseen papers)",
                 out_dir / f"roc_curve_{slug}.png")
        plot_pr(yg_te, yg_proba, f"{name} -- Precision-Recall (unseen papers)",
                out_dir / f"pr_curve_{slug}.png")

        per_paper_analysis(yg_te, yg_pred, yg_proba, groups_te, out_dir, name)

        if name in ("Decision Tree", "Gradient Boosting"):
            pre = group_pipe.named_steps["preprocess"]
            mdl = group_pipe.named_steps["model"]
            feat_names = get_output_feature_names(pre, numeric_cols, categorical_cols)
            importances = mdl.feature_importances_
            imp_df = (pd.DataFrame({"feature": feat_names, "importance": importances})
                      .sort_values("importance", ascending=False).reset_index(drop=True))
            imp_df.to_csv(out_dir / f"feature_importance_{slug}.csv", index=False)
            plot_importance_bar(feat_names, importances,
                                f"{name} -- Top 20 feature importances",
                                out_dir / f"feature_importance_{slug}.png", top_n=20)
            print(f"Top 10 features ({name}):")
            print(imp_df.head(10).to_string(index=False))

        model_file = {
            "Decision Tree": "best_decision_tree.joblib",
            "Gradient Boosting": "best_gradient_boosting.joblib",
            "KNN": "best_knn.joblib",
        }[name]
        joblib.dump(group_pipe, out_dir / model_file)
        print(f"Saved best {name} model -> {out_dir / model_file}")

    # --------------------------------------------------- model selection
    _print_header("STEP 8  |  MODEL SELECTION (by GroupKFold CV ROC-AUC)")
    best_model_name = max(cv_auc_store, key=cv_auc_store.get)
    print("GroupKFold CV ROC-AUC per model (higher = better extrapolation):")
    for nm, v in sorted(cv_auc_store.items(), key=lambda kv: kv[1], reverse=True):
        flag = "  <-- BEST" if nm == best_model_name else ""
        print(f"   {nm:<20s} CV ROC-AUC = {v:.4f}{flag}")
    print(f"\nSelected best model for unseen-paper ignition prediction: {best_model_name}")
    best_pipe = fitted_group_models[best_model_name]

    # ----------------------------------------- permutation importance (best)
    _print_header("STEP 9  |  PERMUTATION IMPORTANCE (best group-aware model)")
    perm = permutation_importance(
        best_pipe, Xg_te, yg_te, n_repeats=10,
        random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
    )
    perm_df = (pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True))
    perm_df.to_csv(out_dir / "permutation_importance.csv", index=False)
    plot_importance_bar(list(perm_df["feature"]), list(perm_df["importance_mean"]),
                        f"Permutation importance -- {best_model_name} (unseen papers)",
                        out_dir / "permutation_importance.png", top_n=min(20, len(perm_df)))
    print(f"Permutation importance ranking ({best_model_name}):")
    print(perm_df.head(15).to_string(index=False))

    # ------------------------------------------------------- SHAP (best)
    if not args.no_shap:
        _print_header("STEP 10 |  SHAP ANALYSIS (best group-aware model)")
        run_shap(best_pipe, Xg_tr, Xg_te, numeric_cols, categorical_cols,
                 out_dir, best_model_name)
    else:
        print("\n[SHAP] skipped (--no-shap).")

    # ------------------------------------------------- comparison tables
    _print_header("STEP 11 |  MODEL COMPARISON TABLES")
    comp_df = pd.DataFrame(comparison_rows)
    # Sort by Group ROC-AUC (descending -> best first).
    group_auc_order = (
        comp_df[comp_df["Validation Strategy"] == "Group-Aware"]
        .set_index("Model")["ROC_AUC"].sort_values(ascending=False)
    )
    comp_df["__order"] = comp_df["Model"].map({m: i for i, m in enumerate(group_auc_order.index)})
    comp_df = (comp_df.sort_values(["__order", "Validation Strategy"])
               .drop(columns="__order").reset_index(drop=True))
    comp_round = comp_df.copy()
    for col in METRIC_KEYS:
        comp_round[col] = comp_round[col].astype(float).round(4)
    comp_round.to_csv(out_dir / "model_comparison.csv", index=False)
    print("Full comparison (sorted by Group ROC-AUC):")
    print(comp_round.to_string(index=False))

    gap_df = pd.DataFrame(gap_rows).sort_values("Group ROC-AUC", ascending=False).reset_index(drop=True)
    for col in ["Random ROC-AUC", "Group ROC-AUC", "Generalization Gap"]:
        gap_df[col] = gap_df[col].astype(float).round(4)
    gap_df.to_csv(out_dir / "generalization_gap.csv", index=False)
    print("\nGeneralization gap (Random ROC-AUC - Group ROC-AUC; larger = more")
    print("paper-specific overfitting):")
    print(gap_df.to_string(index=False))

    # ----------------------------------------------------------- metrics dump
    metrics_blob = {
        "task": "binary classification (Ignition Yes/No)",
        "random_state": RANDOM_STATE,
        "n_rows_initial": int(n_rows_initial),
        "n_rows_final": int(n_after),
        "class_balance": {"ignition": pos, "no_ignition": neg},
        "target_column": target_col,
        "group_column": group_col,
        "n_unique_papers": int(groups_all.nunique()),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "best_model": best_model_name,
        "cv_group_roc_auc": {k: float(v) for k, v in cv_auc_store.items()},
        "best_params": best_params_store,
        "comparison": comp_round.to_dict(orient="records"),
        "generalization_gap": gap_df.to_dict(orient="records"),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_blob, f, indent=2, default=str)

    _print_header("DONE")
    print(f"All artefacts written to: {out_dir.resolve()}")
    print(f"PRIMARY scientific result = Group-Aware rows. Best model: {best_model_name}.")


if __name__ == "__main__":
    main()
