#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fsr_extrapolation_regression.py
===============================

Paper-aware (extrapolation-first) machine-learning study of **Flame Spread Rate
(FSR)** for the microgravity-combustion literature database
(``Microgravity_Database.xlsm``).

Scientific motivation
---------------------
This database is a *literature aggregation*: a single research paper (one
experimental campaign, one rig, one team) typically contributes *many* rows.
Those rows are highly correlated -- same facility, same diagnostic technique,
same calibration, often the same material at a handful of conditions.

If we evaluate a model with a plain random train/test split, measurements from
the *same paper* land in both the training and the test set. The model then only
has to *interpolate between rows of a paper it has already partly seen*. That
flatters the metrics but tells us almost nothing about the question a combustion
scientist actually cares about:

    "Given a brand-new paper / campaign / rig that the model has never seen,
     how well can we predict its flame spread rate?"

That is an **extrapolation** problem, and the only honest way to estimate it is
to make sure that *all rows of any given paper are kept together* -- entirely in
train OR entirely in test, never split across the two. This is "group-aware" or
"leave-papers-out" evaluation, implemented here with ``GroupShuffleSplit`` (for
the hold-out) and ``GroupKFold`` (for cross-validation and hyper-parameter
tuning), using the paper identifier as the grouping key.

The script therefore reports TWO evaluation strategies for every model:

  * Strategy A -- Random split            (baseline, optimistic, "interpolation")
  * Strategy B -- Group/paper split       (PRIMARY scientific result, "extrapolation")

and crucially **all hyper-parameter tuning is driven by GroupKFold**, so the
selected models are the ones that generalise best to *unseen papers*, not the
ones that merely interpolate within papers.

Models compared
---------------
  1. DecisionTreeRegressor
  2. XGBRegressor               (XGBoost -- replaces the earlier Gradient Boosting)
  3. KNeighborsRegressor
  4. MLPRegressor               (Multi-Layer Perceptron neural network)

Everything (feature detection, target detection, paper-id detection, leakage
removal) is done **automatically** with robust heuristics rather than hard-coded
column names, so the script tolerates renamed / reordered columns.

Data augmentation: Bootstrap resampling (BT)
--------------------------------------------
Following the founding paper of this work --

    Jose Rivera, Daniel San Martin, Carlos Fernandez-Pello, Michael J. Gollner,
    Augustin Guibaud, Sandra Olson, Dennis Stocker,
    "Using Data Categorization and Augmentation Strategies to improve Machine
     Learning Frameworks for Flame Spread over Electrical Wires."

-- we add **bootstrap resampling** as a data-augmentation strategy. As described
in that paper (Sec. 3.3): bootstrap resampling is implemented by sampling rows of
the (training) dataset *with replacement* (random seed 42), generating an
additional ~1000 synthetic training points. This "increases the density of
empirically observed joint feature combinations without altering physical
relationships" and "naturally preserves the physical admissibility of the
original observations". The paper reports that BT is the most consistent
augmentation strategy for tree- and neighbourhood-based models, pushing
R2 > 0.9 under random cross-validation.

CRITICAL adaptation for extrapolation: bootstrap rows are drawn ONLY from the
*training* split (i.e. only from training papers). The test set is never
augmented and never resampled, so the augmentation cannot leak unseen-paper
information. Each model is therefore evaluated under four conditions:
Random/RD, Random/RD+BT, Group/RD and Group/RD+BT, so the augmentation effect can
be read directly.

Outputs
-------
Everything is written to ``results/``:
  * predicted-vs-experimental, residual and error-histogram plots per model
  * per-paper performance distributions (RMSE / MAE / R2 histograms + boxplot)
  * feature-importance plots (Decision Tree & XGBoost)
  * permutation-importance bar chart + ranked table (best group-aware model)
  * SHAP summary + bar plots (best group-aware model)
  * bootstrap FSR-distribution figure + augmentation-effect bar chart
  * ``model_comparison.csv``, ``generalization_gap.csv``, ``augmentation_comparison.csv``
  * ``metrics.json`` and per-paper CSVs
  * ``best_decision_tree.joblib`` / ``best_xgboost.joblib`` / ``best_knn.joblib`` /
    ``best_mlp.joblib``

Run
---
    python fsr_extrapolation_regression.py
    # optional flags:
    python fsr_extrapolation_regression.py --data Microgravity_Database.xlsm \
        --out results --n-iter 40 --no-shap --bootstrap-n 1000

Author: ML / combustion research assistant
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

# Matplotlib must be configured for a headless ("Agg") backend *before* pyplot
# is imported, so the script runs on servers / CI with no display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# XGBoost replaces the previous Gradient Boosting estimator. It is a regularised,
# histogram-based gradient-boosted-tree implementation that typically matches or
# beats scikit-learn's GradientBoostingRegressor while training much faster.
from xgboost import XGBRegressor

import joblib

# A single global seed reused everywhere for reproducibility.
RANDOM_STATE = 42

# Directory that contains this script. Used to make the default input/output
# paths independent of the current working directory, so the script runs the
# same way whether launched from the repository root or from its own folder.
SCRIPT_DIR = Path(__file__).resolve().parent

# Silence a few noisy-but-harmless warnings (e.g. undefined R2 on tiny papers,
# MAPE on near-zero targets). We handle those cases explicitly ourselves.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*R\\^2 score is not well-defined.*")


# =============================================================================
# 0. Small, dependency-free helpers
# =============================================================================

# Matches the first numeric token in a string, including scientific notation
# and a leading sign, e.g. "101.3 kPa" -> "101.3", "-60 mm/s" -> "-60".
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _clean_text(x):
    """Normalise a raw cell to a tidy string, or NaN for placeholder/empties.

    The Excel sheet is hand-curated and contains non-breaking spaces, stray
    whitespace and a zoo of "missing" tokens ("-", "n/a", "none"...). Treating
    them all as NaN keeps the categorical encoders clean.
    """
    if pd.isna(x):
        return np.nan
    if not isinstance(x, str):
        return x
    s = x.replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s if s.lower() not in {"", "-", "--", "n/a", "na", "nan", "none"} else np.nan


def _first_number(x):
    """Extract the first numeric value from a possibly unit-laden cell.

    Many physical columns are stored as strings *with units* ("94 W", "8 s",
    "101.3 kPa"). For a tree / distance based regressor we only need a clean,
    *consistent* numeric value, so we pull out the leading number. This is the
    core trick that lets automatic numeric detection work on this messy sheet
    without hard-coding per-column unit parsers.
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).replace(",", ".").replace("\u2212", "-")  # unicode minus -> ascii
    m = _NUM_RE.search(s)
    return float(m.group(0)) if m else np.nan


def _slug(text: str) -> str:
    """Filesystem-safe slug used when naming per-model output files."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()


def _print_header(title: str) -> None:
    """Pretty section banner for publication-quality console logs."""
    line = "=" * 78
    print("\n" + line)
    print(title)
    print(line)


def locate_data_file(user_path: str) -> Path:
    """Resolve the database path robustly, independent of the cwd.

    Search order: (1) the path exactly as given, (2) the same filename next to
    this script, (3) the same filename in the script's parent directories (the
    repository root). This lets the script live inside the ``FSR Regression``
    folder while the workbook sits at the project root.
    """
    given = Path(user_path)
    if given.exists():
        return given
    name = given.name
    candidates = [SCRIPT_DIR / name]
    candidates += [parent / name for parent in SCRIPT_DIR.parents]
    for cand in candidates:
        if cand.exists():
            return cand
    return given  # return the original so the caller can emit a clear error


# =============================================================================
# 1. Data loading (robust to the two-row "section / field" Excel header)
# =============================================================================

def load_database(path: Path) -> pd.DataFrame:
    """Load the microgravity database into a flat, single-header DataFrame.

    The primary data sheet ("Sheet2") uses a *two-level* header: a top "section"
    banner row (Citation / Sample / Flow / Outputs / Info) above the real field
    names. We try that layout first and flatten it (keeping the descriptive
    field name). If anything about the layout differs we gracefully fall back to
    a single-header read and finally to whatever sheet/parse succeeds, so the
    script does not hard-fail on a re-saved workbook.
    """
    read_kwargs = {}
    # .xlsm needs openpyxl; pandas picks it automatically, but be explicit.
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

    # Identify the most plausible data sheet automatically: prefer the sheet
    # that actually contains a flame-spread-rate column and the most rows.
    candidate_sheets = []
    try:
        xls = pd.ExcelFile(path, **read_kwargs)
        candidate_sheets = list(xls.sheet_names)
    except Exception:  # pragma: no cover - extremely defensive
        candidate_sheets = [0]

    best_df = None
    best_score = -1
    for sheet in candidate_sheets:
        for header in ([0, 1], 0):  # two-row header first, then single row
            try:
                df = pd.read_excel(path, sheet_name=sheet, header=header, **read_kwargs)
            except Exception:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df = _flatten_two_level(df)
            df.columns = [str(c).strip() for c in df.columns]
            # Score this parse: many rows + presence of an FSR-like column wins.
            has_fsr = any(_looks_like_target(c) for c in df.columns)
            score = len(df) * df.shape[1] + (10_000_000 if has_fsr else 0)
            if score > best_score:
                best_score, best_df = score, df

    if best_df is None:
        raise RuntimeError(f"Could not read any usable sheet from {path}")

    # Drop fully-empty and duplicated columns that pandas sometimes invents.
    best_df = best_df.loc[:, ~best_df.columns.duplicated()]
    best_df = best_df.dropna(axis=1, how="all")

    # Tidy every text cell once, up front.
    for c in best_df.columns:
        if best_df[c].dtype == object:
            best_df[c] = best_df[c].map(_clean_text)

    return best_df.reset_index(drop=True)


# =============================================================================
# 2. Automatic column-role detection (target, paper id, leakage, num/cat)
# =============================================================================

# --- Target ----------------------------------------------------------------
# Keywords that identify the Flame Spread Rate column under various spellings.
_TARGET_KEYWORDS = ["fsr", "flame spread", "spread rate", "spreadrate"]


def _looks_like_target(col: str) -> bool:
    c = str(col).lower()
    return any(k in c for k in _TARGET_KEYWORDS)


def detect_target_column(df: pd.DataFrame) -> str:
    """Find the Flame Spread Rate column automatically.

    Strategy: prefer columns whose *name* clearly mentions flame spread / FSR.
    Among those, pick the one with the most *numeric-coercible* values so we get
    the real measurement column rather than, say, a notes column that merely
    mentions "flame spread".
    """
    candidates = [c for c in df.columns if _looks_like_target(c)]
    if not candidates:
        raise RuntimeError(
            "No Flame Spread Rate column found. Looked for keywords: "
            f"{_TARGET_KEYWORDS}. Available columns: {list(df.columns)}"
        )
    # Rank by how many values parse to a number.
    best = max(candidates, key=lambda c: df[c].map(_first_number).notna().sum())
    return best


# --- Paper / grouping identifier ------------------------------------------
# Keywords (ordered by preference) that identify a per-paper grouping column.
_GROUP_KEYWORDS = [
    "article",
    "paper",
    "mla",
    "citation",
    "doi",
    "publication",
    "reference",
    "author",
]


def detect_group_column(df: pd.DataFrame) -> str:
    """Find the column that identifies the source paper / campaign.

    A good grouping column is (a) named like a citation/article/DOI and
    (b) reasonably *complete* (few missing values) so that every row can be
    assigned to a paper. We score candidates by keyword priority and by
    completeness, and break ties toward fewer missing values. A column that is
    unique-per-row (a pure row index) is rejected -- it cannot group anything.
    """
    n = len(df)
    scored = []
    for c in df.columns:
        cl = str(c).lower()
        for rank, kw in enumerate(_GROUP_KEYWORDS):
            if kw in cl:
                non_null = df[c].notna().sum()
                nunique = df[c].nunique(dropna=True)
                # Reject would-be identifiers that are essentially one-per-row.
                if nunique >= max(2, 0.9 * n):
                    continue
                completeness = non_null / max(n, 1)
                # Lower keyword rank == higher preference; then more complete.
                scored.append((rank, -completeness, c, nunique, completeness))
                break
    if not scored:
        raise RuntimeError(
            "No paper/grouping column found. Looked for keywords: "
            f"{_GROUP_KEYWORDS}. Available columns: {list(df.columns)}"
        )
    scored.sort()
    chosen = scored[0][2]
    return chosen


# --- Leakage columns -------------------------------------------------------
# Other "Outputs" of an experiment are recorded *simultaneously* with FSR
# (post-ignition observations). Using them as predictors would leak information
# that is not available when forecasting FSR for an unseen experiment, so we
# remove any column whose name matches these outcome keywords.
_LEAKAGE_KEYWORDS = [
    "flame length",
    "hrr",
    "heat release",
    "smoke",
    "aerosol",
    "areosol",  # the sheet has this typo
    "ignition (yes",  # ignition outcome flag (all FSR rows ignited -> constant)
    "burn",
    "extinction",
    "extinguish",
]

# Free-text / note columns that carry no generalisable signal and could even
# encode the answer in prose -> dropped from the feature matrix.
_NOTE_KEYWORDS = ["info", "note", "comment", "remark"]


def detect_leakage_columns(df: pd.DataFrame, target: str, group_col: str) -> list[str]:
    """Return columns to drop before modelling (besides target & group).

    We remove:
      * any *other* column that also looks like a flame-spread target
        (duplicate target representations),
      * post-experiment outcome columns (leakage),
      * free-text note columns,
      * identifier columns used purely to cite the paper (authors / DOI /
        article): these are paper fingerprints. Keeping them would let the
        model memorise paper identity (a severe leak under random splitting and
        useless under group splitting where test papers are unseen).
    """
    leak = []
    for c in df.columns:
        if c in (target, group_col):
            continue
        cl = str(c).lower()
        if _looks_like_target(c):
            leak.append(c)  # duplicate target representation
        elif any(k in cl for k in _LEAKAGE_KEYWORDS):
            leak.append(c)  # post-experiment outcome
        elif any(k in cl for k in _NOTE_KEYWORDS):
            leak.append(c)  # free-text notes
        elif any(k in cl for k in _GROUP_KEYWORDS):
            leak.append(c)  # paper-identity fingerprint (author/doi/article...)
    # Preserve original order, de-duplicate.
    seen = set()
    ordered = [c for c in df.columns if c in leak and not (c in seen or seen.add(c))]
    return ordered


# --- Numeric vs categorical -----------------------------------------------

def detect_feature_types(
    df: pd.DataFrame, feature_cols: list[str], numeric_frac_threshold: float = 0.60
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Split features into numeric vs categorical *automatically*.

    A column is treated as numeric if at least ``numeric_frac_threshold`` of its
    non-missing values can be coerced to a number (after stripping units). This
    correctly classifies unit-laden string columns such as "94 W" or "101.3 kPa"
    as numeric, while leaving genuine categoricals (Material, Geometry, ...) as
    categorical. The returned DataFrame has the numeric columns replaced by their
    parsed float values.

    Returns ``(numeric_cols, categorical_cols, transformed_df)``.
    """
    out = df.copy()
    numeric_cols, categorical_cols = [], []
    for c in feature_cols:
        series = out[c]
        non_null = series.notna().sum()
        if non_null == 0:
            # An all-missing column carries no signal; skip it entirely.
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(c)
            continue
        parsed = series.map(_first_number)
        frac = parsed.notna().sum() / non_null
        if frac >= numeric_frac_threshold:
            out[c] = parsed  # commit the numeric parse
            numeric_cols.append(c)
        else:
            out[c] = series.map(_clean_text)
            categorical_cols.append(c)
    return numeric_cols, categorical_cols, out


# =============================================================================
# 3. Metrics
# =============================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the full battery of regression metrics requested for the study.

    * R2     -- coefficient of determination (fraction of variance explained).
    * RMSE   -- root mean squared error (penalises large errors, target units).
    * MAE    -- mean absolute error (robust, target units).
    * MAPE   -- mean absolute percentage error, computed on a mask that excludes
                near-zero targets (FSR can be ~0, which would blow MAPE up).
    * MBE    -- mean bias error = mean(pred - true); >0 over-prediction.
    * NRMSE  -- RMSE normalised by mean(true); a dimensionless error scale.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)
    if n == 0:
        return {k: np.nan for k in ["R2", "RMSE", "MAE", "MAPE", "MBE", "NRMSE", "n"]}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    # R2 is undefined when the test target has zero variance (e.g. a one-row
    # paper); report NaN rather than a misleading value.
    r2 = float(r2_score(y_true, y_pred)) if n >= 2 and np.var(y_true) > 0 else np.nan

    mean_true = float(np.mean(y_true))
    mbe = float(np.mean(y_pred - y_true))
    nrmse = float(rmse / mean_true) if abs(mean_true) > 1e-12 else np.nan

    # Safe MAPE: only over samples whose true magnitude is meaningfully non-zero.
    eps = 1e-6
    mask = np.abs(y_true) > eps
    mape = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
        if mask.any()
        else np.nan
    )

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "MBE": mbe,
        "NRMSE": nrmse,
        "n": int(n),
    }


# =============================================================================
# 4. Preprocessing pipeline + model/search-space definitions
# =============================================================================

def build_preprocessor(
    numeric_cols: list[str], categorical_cols: list[str]
) -> ColumnTransformer:
    """One shared preprocessing pipeline used by *every* model.

    * Numeric branch:  median imputation (robust to outliers/skew) + StandardScaler.
      Scaling is essential for KNN (distance based) and harmless for the trees.
    * Categorical branch: most-frequent imputation + OneHotEncoder with
      ``handle_unknown='ignore'`` so categories that appear only in *unseen test
      papers* do not crash inference (they simply encode as all-zeros).
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def get_models_and_spaces() -> dict:
    """Return the four estimators and their RandomizedSearchCV search spaces.

    The hyper-parameter ranges are deliberately chosen to fight over-fitting --
    the central risk for extrapolation. Shallower trees, larger leaf sizes and
    feature/row subsampling all push the models toward smoother, more
    transferable functions. The ``model__`` prefix targets the estimator step
    inside the shared Pipeline.

    Model line-up (per the user's request): the previous Gradient Boosting model
    is replaced by **XGBoost**, and a **Multi-Layer Perceptron** is added as a
    fourth, neural-network model (the founding paper also uses DT, GB/XGB-style
    boosting, KNN and MLP).
    """
    models = {
        # --- Decision Tree -------------------------------------------------
        "Decision Tree": {
            "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "param_dist": {
                "model__max_depth": [3, 4, 5, 6, 8, 10, 12, None],
                "model__min_samples_split": [2, 5, 10, 20, 40],
                "model__min_samples_leaf": [1, 2, 4, 8, 16, 32],
                "model__max_features": ["sqrt", "log2", 0.5, 0.8, 1.0, None],
            },
        },
        # --- XGBoost (replaces Gradient Boosting) --------------------------
        # Regularised, histogram-based gradient-boosted trees. We expose the
        # tree-depth, shrinkage, sub-/column-sampling and L1/L2/gamma penalties
        # that most directly control over-fitting on unseen papers.
        "XGBoost": {
            "estimator": XGBRegressor(
                random_state=RANDOM_STATE,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=1,  # parallelism is handled by the outer RandomizedSearchCV
                verbosity=0,
            ),
            "param_dist": {
                "model__n_estimators": [100, 200, 300, 500, 800],
                "model__learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 4, 5, 6],
                "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.5, 0.7, 0.8, 1.0],
                "model__min_child_weight": [1, 2, 4, 6, 10],
                "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
                "model__reg_alpha": [0.0, 0.1, 0.5, 1.0],
                "model__gamma": [0.0, 0.1, 0.3, 1.0],
            },
        },
        # --- K-Nearest Neighbors ------------------------------------------
        "KNN": {
            "estimator": KNeighborsRegressor(),
            "param_dist": {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],  # 1 = Manhattan, 2 = Euclidean
            },
        },
        # --- Multi-Layer Perceptron (neural network) ----------------------
        # Operates on the StandardScaler-ed features. early_stopping + an L2
        # penalty (alpha) guard against over-fitting; a high max_iter ensures
        # convergence within each CV fit.
        "MLP": {
            "estimator": MLPRegressor(
                random_state=RANDOM_STATE,
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=20,
            ),
            "param_dist": {
                "model__hidden_layer_sizes": [(100,), (128, 64), (100, 50), (64, 64, 32)],
                "model__activation": ["relu", "tanh"],
                "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
                "model__learning_rate_init": [1e-3, 5e-3, 1e-2],
            },
        },
    }
    return models


# =============================================================================
# 4b. Bootstrap data augmentation (founding paper, Rivera et al., Sec. 3.3)
# =============================================================================

def bootstrap_augment(X: pd.DataFrame, y: np.ndarray, n_extra: int,
                      random_state: int = RANDOM_STATE):
    """Bootstrap resampling augmentation, exactly as in the founding paper.

    Sample ``n_extra`` rows *with replacement* from (X, y) and append them to the
    original data, producing an augmented training set of size ``len(X)+n_extra``.

    Why this is sound for extrapolation: bootstrap rows are exact copies of
    existing observations, so they "preserve the physical admissibility of the
    original observations" (Rivera et al.). We only ever call this on the
    *training* split, so no unseen-paper (test) information is introduced. The
    net effect is to re-weight the empirical distribution toward its high-density
    (low-FSR) regions, which the paper found especially helps tree- and
    neighbourhood-based models.
    """
    if n_extra <= 0:
        return X.copy(), np.asarray(y).copy()
    rng = np.random.RandomState(random_state)
    idx = rng.randint(0, len(X), size=n_extra)
    X_aug = pd.concat([X, X.iloc[idx]], axis=0, ignore_index=True)
    y_aug = np.concatenate([np.asarray(y), np.asarray(y)[idx]])
    return X_aug, y_aug


def fit_pipeline_with_params(estimator, best_params, numeric_cols, categorical_cols,
                             X_tr, y_tr):
    """Build a fresh pipeline (own preprocessor) with the tuned params and fit it.

    A new preprocessor + estimator are instantiated each time so that the four
    evaluation conditions (Random/RD, Random/RD+BT, Group/RD, Group/RD+BT) never
    share fitted state. Hyper-parameters are always the ones chosen by the
    GroupKFold search, so only the *training data* differs across conditions.
    """
    pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", estimator.__class__(**{**estimator.get_params(), **best_params})),
        ]
    )
    pipe.fit(X_tr, y_tr)
    return pipe


# =============================================================================
# 5. Feature-name recovery after the ColumnTransformer / OneHotEncoder
# =============================================================================

def get_output_feature_names(
    preprocessor: ColumnTransformer,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[str]:
    """Recover human-readable feature names after preprocessing.

    Prefers scikit-learn's ``get_feature_names_out`` and falls back to manual
    reconstruction so that feature-importance / SHAP plots show meaningful
    labels like ``cat__Material of sample=PMMA`` instead of ``x37``.
    """
    try:
        names = list(preprocessor.get_feature_names_out())
        # Tidy the ``num__`` / ``cat__`` transformer prefixes for readability.
        names = [re.sub(r"^(num|cat)__", "", n) for n in names]
        return names
    except Exception:
        names = list(numeric_cols)
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        for col, cats in zip(categorical_cols, ohe.categories_):
            names.extend(f"{col}={c}" for c in cats)
        return names


# =============================================================================
# 6. Plotting helpers
# =============================================================================

def plot_pred_vs_true(y_true, y_pred, title, metrics, path):
    """Predicted-vs-experimental scatter with a 1:1 reference line."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.45, edgecolor="k", linewidth=0.3, s=28)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.5, label="1:1 line")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Experimental FSR")
    ax.set_ylabel("Predicted FSR")
    r2 = metrics.get("R2", np.nan)
    rmse = metrics.get("RMSE", np.nan)
    ax.set_title(f"{title}\n$R^2$={r2:.3f}   RMSE={rmse:.3f}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, title, path):
    """Residual (prediction - truth) vs predicted value."""
    resid = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(y_pred, resid, alpha=0.45, edgecolor="k", linewidth=0.3, s=28)
    ax.axhline(0.0, color="r", ls="--", lw=1.5)
    ax.set_xlabel("Predicted FSR")
    ax.set_ylabel("Residual (Predicted - Experimental)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_hist(y_true, y_pred, title, path):
    """Histogram of prediction errors."""
    err = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.hist(err, bins=30, color="steelblue", edgecolor="k", alpha=0.8)
    ax.axvline(0.0, color="r", ls="--", lw=1.5)
    ax.axvline(float(np.mean(err)), color="orange", ls="-", lw=1.5,
               label=f"mean bias = {np.mean(err):.3f}")
    ax.set_xlabel("Prediction error (Predicted - Experimental)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance_bar(names, values, title, path, top_n=20):
    """Horizontal bar chart of the top-N most important features."""
    order = np.argsort(values)[::-1][:top_n]
    sel_names = [names[i] for i in order]
    sel_vals = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(sel_names))))
    ax.barh(range(len(sel_names)), sel_vals, color="seagreen", edgecolor="k")
    ax.set_yticks(range(len(sel_names)))
    ax.set_yticklabels(sel_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# 7. Per-paper generalisation analysis
# =============================================================================

def per_paper_analysis(y_true, y_pred, groups, out_dir: Path, model_name: str):
    """Compute and visualise model performance *separately for every paper*.

    Average metrics can hide catastrophic failure on individual papers. By
    scoring each held-out paper on its own we expose that heterogeneity, which is
    exactly the information a reviewer needs to trust an extrapolation claim.
    """
    df = pd.DataFrame(
        {"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred), "paper": np.asarray(groups)}
    )
    rows = []
    for paper, g in df.groupby("paper"):
        m = regression_metrics(g["y_true"].values, g["y_pred"].values)
        rows.append({"paper": paper, **m})
    per_paper = pd.DataFrame(rows).sort_values("RMSE", ascending=False).reset_index(drop=True)

    slug = _slug(model_name)
    per_paper.to_csv(out_dir / f"per_paper_metrics_{slug}.csv", index=False)

    # 1+2) RMSE & MAE distributions across papers (histograms).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(per_paper["RMSE"].dropna(), bins=20, color="indianred",
                 edgecolor="k", alpha=0.8)
    axes[0].set_title("Per-paper RMSE distribution")
    axes[0].set_xlabel("RMSE")
    axes[0].set_ylabel("Number of papers")
    axes[1].hist(per_paper["MAE"].dropna(), bins=20, color="slateblue",
                 edgecolor="k", alpha=0.8)
    axes[1].set_title("Per-paper MAE distribution")
    axes[1].set_xlabel("MAE")
    axes[1].set_ylabel("Number of papers")
    fig.suptitle(f"{model_name} -- per-paper error distributions (unseen papers)")
    fig.tight_layout()
    fig.savefig(out_dir / f"per_paper_rmse_mae_hist_{slug}.png", dpi=150)
    plt.close(fig)

    # 3) Histogram of per-paper R2 (where defined).
    r2_vals = per_paper["R2"].dropna()
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    if len(r2_vals):
        ax.hist(r2_vals, bins=20, color="darkgreen", edgecolor="k", alpha=0.8)
    ax.set_title(f"{model_name} -- per-paper $R^2$ distribution")
    ax.set_xlabel("$R^2$ on a single held-out paper")
    ax.set_ylabel("Number of papers")
    fig.tight_layout()
    fig.savefig(out_dir / f"per_paper_r2_hist_{slug}.png", dpi=150)
    plt.close(fig)

    # 4) Boxplot of per-paper RMSE.
    fig, ax = plt.subplots(figsize=(4.5, 5))
    # `vert=`/`orientation=` keyword changed across matplotlib versions; call the
    # simplest signature that works everywhere (default is a vertical boxplot).
    ax.boxplot(per_paper["RMSE"].dropna(), patch_artist=True,
               boxprops=dict(facecolor="lightcoral"))
    ax.set_ylabel("Per-paper RMSE")
    ax.set_title(f"{model_name}\nper-paper RMSE boxplot")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / f"per_paper_rmse_boxplot_{slug}.png", dpi=150)
    plt.close(fig)

    return per_paper


# =============================================================================
# 8. SHAP analysis
# =============================================================================

def run_shap(best_pipe, X_train, X_test, numeric_cols, categorical_cols,
             out_dir: Path, model_name: str):
    """Compute SHAP values for the best group-aware model and save plots.

    SHAP runs on the *transformed* feature space (post one-hot), so we pull the
    fitted preprocessor out of the pipeline, transform the data, and explain the
    bare regressor. Tree models use the fast exact ``TreeExplainer``; other
    models (e.g. KNN) fall back to the model-agnostic ``Explainer`` on a small
    background sample to keep runtime reasonable.
    """
    try:
        import shap
    except Exception as exc:  # pragma: no cover
        print(f"  [SHAP] shap not available ({exc}); skipping SHAP analysis.")
        return

    pre = best_pipe.named_steps["preprocess"]
    model = best_pipe.named_steps["model"]
    feat_names = get_output_feature_names(pre, numeric_cols, categorical_cols)

    Xt_train = pre.transform(X_train)
    Xt_test = pre.transform(X_test)

    # Cap the number of explained rows; SHAP is the slow part of the script.
    rng = np.random.RandomState(RANDOM_STATE)
    n_explain = min(300, Xt_test.shape[0])
    idx = rng.choice(Xt_test.shape[0], size=n_explain, replace=False)
    Xt_sample = Xt_test[idx]

    try:
        if isinstance(model, (DecisionTreeRegressor, XGBRegressor)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(Xt_sample)
        else:
            n_bg = min(100, Xt_train.shape[0])
            bg = shap.sample(Xt_train, n_bg, random_state=RANDOM_STATE)
            explainer = shap.KernelExplainer(model.predict, bg)
            shap_values = explainer.shap_values(Xt_sample, nsamples=100)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  [SHAP] explainer failed ({exc}); skipping SHAP analysis.")
        return

    slug = _slug(model_name)

    # SHAP summary (beeswarm) plot.
    plt.figure()
    shap.summary_plot(shap_values, Xt_sample, feature_names=feat_names,
                      show=False, max_display=20)
    plt.title(f"SHAP summary -- {model_name} (best group-aware model)")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_summary_{slug}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # SHAP bar plot (mean |SHAP|).
    plt.figure()
    shap.summary_plot(shap_values, Xt_sample, feature_names=feat_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP mean(|value|) -- {model_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_bar_{slug}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Ranked table + plain-language interpretation of the top drivers.
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_rank = (
        pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_rank.to_csv(out_dir / f"shap_ranking_{slug}.csv", index=False)

    print(f"\n  SHAP top drivers of Flame Spread Rate ({model_name}):")
    for i, row in shap_rank.head(10).iterrows():
        print(f"    {i + 1:2d}. {row['feature']:<45s} mean|SHAP| = {row['mean_abs_shap']:.4f}")
    print(
        "  Interpretation: features at the top of this ranking move the model's\n"
        "  FSR prediction the most. In microgravity flame-spread physics the\n"
        "  dominant drivers are typically oxygen concentration, flow velocity and\n"
        "  pressure (oxidiser transport to the flame), followed by sample material\n"
        "  / geometry (fuel chemistry and thermal thickness) and gravity level."
    )


# =============================================================================
# 9. Main driver
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="Microgravity_Database.xlsm",
                        help="Path to the Excel database (auto-located if not "
                             "found in the current directory).")
    parser.add_argument("--out", default=str(SCRIPT_DIR / "results"),
                        help="Output directory for all artefacts "
                             "(defaults to a 'results' folder next to this script).")
    parser.add_argument("--n-iter", type=int, default=40,
                        help="RandomizedSearchCV iterations per model.")
    parser.add_argument("--cv-splits", type=int, default=5,
                        help="Number of GroupKFold splits for tuning.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test fraction for both split strategies.")
    parser.add_argument("--bootstrap-n", type=int, default=1000,
                        help="Number of bootstrap rows to ADD to the training set "
                             "(founding paper uses 1000). Set 0 to disable BT.")
    parser.add_argument("--no-shap", action="store_true",
                        help="Disable the (slower) SHAP analysis.")
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
    _print_header("STEP 2  |  DETECT TARGET (Flame Spread Rate)")
    target_col = detect_target_column(df_raw)
    print(f"Detected target column: '{target_col}'")

    # Coerce the target to numeric (some cells are ratio strings like
    # "Vf0G/Vf1G = 0.92") and drop rows where FSR is missing/non-numeric.
    df = df_raw.copy()
    df[target_col] = df[target_col].map(_first_number)
    n_before = len(df)
    df = df[df[target_col].notna()].copy().reset_index(drop=True)
    n_after = len(df)
    print(f"Initial dataset size (all rows)            : {n_before}")
    print(f"Final dataset size (rows with valid FSR)   : {n_after}")
    print(f"Rows removed due to missing/invalid target : {n_before - n_after}")

    # ------------------------------------------------------------- paper id
    _print_header("STEP 3  |  DETECT PAPER / GROUPING IDENTIFIER")
    group_col = detect_group_column(df)
    # Fill any residual missing group labels so every row belongs to a group.
    df[group_col] = df[group_col].fillna("UNKNOWN_PAPER").astype(str)
    print(f"Detected paper/grouping column: '{group_col}'")

    groups_all = df[group_col]
    samples_per_paper = groups_all.value_counts()
    print(f"Number of unique papers          : {groups_all.nunique()}")
    print(f"Mean samples per paper           : {samples_per_paper.mean():.2f}")
    print(f"Median samples per paper         : {samples_per_paper.median():.1f}")
    print(f"Maximum samples per paper        : {samples_per_paper.max()}")
    print(f"Minimum samples per paper        : {samples_per_paper.min()}")
    print("Samples-per-paper describe():")
    print(samples_per_paper.describe().to_string())

    # ----------------------------------------------------------- leakage drop
    _print_header("STEP 4  |  LEAKAGE PREVENTION (remove unsafe columns)")
    leak_cols = detect_leakage_columns(df, target_col, group_col)
    print("The following columns are removed before modelling because they are")
    print("the target, duplicate target representations, post-experiment")
    print("outcomes, free-text notes, or paper-identity fingerprints:")
    for c in leak_cols:
        print(f"   - {c}")
    print(f"   - {target_col}   (target, held out separately)")
    print(f"   - {group_col}   (grouping key, held out separately)")

    feature_cols = [
        c for c in df.columns if c not in set(leak_cols) | {target_col, group_col}
    ]

    # ------------------------------------------------- numeric/categorical
    _print_header("STEP 5  |  AUTOMATIC FEATURE-TYPE DETECTION")
    numeric_cols, categorical_cols, df_feat = detect_feature_types(df, feature_cols)
    # df_feat has the parsed numeric columns; carry over target & group.
    df_feat[target_col] = df[target_col].values
    df_feat[group_col] = df[group_col].values

    print(f"Total rows                       : {len(df_feat)}")
    print(f"Total feature columns            : {len(numeric_cols) + len(categorical_cols)}")
    print(f"Numerical features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # Assemble the modelling matrices.
    X = df_feat[numeric_cols + categorical_cols].copy()
    y = df_feat[target_col].astype(float).values
    groups = df_feat[group_col].values

    # ----------------------------------------------------------- preprocess
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # ---------------------------------------------------------- the splits
    _print_header("STEP 6  |  BUILD THE TWO EVALUATION SPLITS")

    # Strategy A: random split (baseline / interpolation reference).
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE
    )
    print(f"[A] Random split        : train={len(Xr_tr)}  test={len(Xr_te)}")

    # Strategy B: group split (extrapolation / PRIMARY scientific evaluation).
    gss = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=RANDOM_STATE
    )
    g_tr_idx, g_te_idx = next(gss.split(X, y, groups=groups))
    Xg_tr, Xg_te = X.iloc[g_tr_idx], X.iloc[g_te_idx]
    yg_tr, yg_te = y[g_tr_idx], y[g_te_idx]
    groups_tr, groups_te = groups[g_tr_idx], groups[g_te_idx]
    n_train_papers = pd.Series(groups_tr).nunique()
    n_test_papers = pd.Series(groups_te).nunique()
    # Guarantee the cardinal rule: no paper appears in both train and test.
    assert set(groups_tr).isdisjoint(set(groups_te)), "Group leakage detected!"
    print(f"[B] Group/paper split   : train={len(Xg_tr)} ({n_train_papers} papers)  "
          f"test={len(Xg_te)} ({n_test_papers} papers)  -- no shared papers.")

    # GroupKFold for tuning needs n_splits <= number of training papers.
    n_cv = max(2, min(args.cv_splits, n_train_papers))
    if n_cv < args.cv_splits:
        print(f"    (reduced GroupKFold splits to {n_cv} to match #train papers)")

    # --------------------------------------------- bootstrap augmentation setup
    _print_header("STEP 6b |  BOOTSTRAP DATA AUGMENTATION (Rivera et al.)")
    n_boot = max(0, int(args.bootstrap_n))
    if n_boot > 0:
        print(f"Bootstrap resampling enabled: +{n_boot} rows sampled WITH "
              f"replacement from the TRAINING split only (seed {RANDOM_STATE}).")
        print("Per the founding paper, BT preserves physical admissibility (rows")
        print("are exact copies of real observations) and densifies the empirical")
        print("distribution. The test set is never augmented (no leakage).")
        # Replicate the paper's Figure 4: real vs bootstrap-augmented FSR histogram
        # (illustrated on the group-train split, the data the models actually see).
        _, yg_tr_bt = bootstrap_augment(Xg_tr, yg_tr, n_boot)
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        bins = np.linspace(float(np.min(y)), float(np.max(y)), 40)
        ax.hist(yg_tr, bins=bins, alpha=0.6, label=f"Real train (n={len(yg_tr)})",
                color="steelblue", edgecolor="k")
        ax.hist(yg_tr_bt, bins=bins, alpha=0.5,
                label=f"Real + Bootstrap (n={len(yg_tr_bt)})",
                color="darkorange", edgecolor="k")
        ax.set_xlabel("Flame Spread Rate"); ax.set_ylabel("Frequency")
        ax.set_title("Bootstrap augmentation: FSR distribution (training split)")
        ax.legend(); ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "bootstrap_fsr_distribution.png", dpi=150)
        plt.close(fig)
        print("Saved bootstrap FSR-distribution figure -> bootstrap_fsr_distribution.png")
    else:
        print("Bootstrap augmentation disabled (--bootstrap-n 0).")

    # ------------------------------------------------- tune + evaluate models
    models = get_models_and_spaces()
    group_cv = GroupKFold(n_splits=n_cv)

    comparison_rows = []   # long-format: one row per (model, strategy x augmentation)
    gap_rows = []          # one row per model: random vs group RMSE + gap (RD only)
    aug_rows = []          # one row per model: RD vs RD+BT effect (both splits)
    fitted_group_models = {}  # name -> pipeline refit on group-train (RD)
    best_params_store = {}
    cv_rmse_store = {}
    METRIC_KEYS = ["R2", "RMSE", "MAE", "MAPE", "NRMSE", "MBE"]

    for name, spec in models.items():
        _print_header(f"STEP 7  |  TUNE + EVALUATE: {name}")

        pipe = Pipeline(
            steps=[("preprocess", preprocessor), ("model", spec["estimator"])]
        )

        # --- Hyper-parameter tuning -- ALWAYS GroupKFold on the group-train set.
        # This is the heart of the extrapolation-first design: hyper-parameters
        # are chosen to minimise RMSE on *held-out papers*, never on randomly
        # mixed rows. The same paper can never be in both a CV-train and a
        # CV-validation fold because GroupKFold respects `groups`.
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=spec["param_dist"],
            n_iter=args.n_iter,
            scoring="neg_root_mean_squared_error",
            cv=group_cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(Xg_tr, yg_tr, groups=groups_tr)
        best_cv_rmse = -search.best_score_
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        best_params_store[name] = best_params
        cv_rmse_store[name] = best_cv_rmse
        print(f"Best GroupKFold CV RMSE (extrapolation) : {best_cv_rmse:.4f}")
        print(f"Best hyper-parameters                   : {best_params}")

        # --- Group-aware evaluation (PRIMARY): refit best estimator already done
        # by RandomizedSearchCV on the whole group-train set; score unseen papers.
        group_pipe = search.best_estimator_
        yg_pred = group_pipe.predict(Xg_te)
        m_group = regression_metrics(yg_te, yg_pred)
        fitted_group_models[name] = group_pipe
        print(f"[GROUP/UNSEEN-PAPERS] R2={m_group['R2']:.3f}  RMSE={m_group['RMSE']:.3f}  "
              f"MAE={m_group['MAE']:.3f}  MAPE={m_group['MAPE']:.1f}%  "
              f"NRMSE={m_group['NRMSE']:.3f}  MBE={m_group['MBE']:.3f}")

        # --- Random-split evaluation (BASELINE): same chosen hyper-parameters,
        # fitted on a random-train split, scored on a random-test split. The
        # gap between this and the group result quantifies paper-overfitting.
        random_pipe = fit_pipeline_with_params(
            spec["estimator"], best_params, numeric_cols, categorical_cols, Xr_tr, yr_tr)
        m_random = regression_metrics(yr_te, random_pipe.predict(Xr_te))
        print(f"[RANDOM / RD]         R2={m_random['R2']:.3f}  RMSE={m_random['RMSE']:.3f}  "
              f"MAE={m_random['MAE']:.3f}  NRMSE={m_random['NRMSE']:.3f}")

        # --- Bootstrap-augmented (RD+BT) variants. Augment ONLY the training
        # data (group-train and random-train respectively); test sets untouched.
        if n_boot > 0:
            Xg_tr_bt, yg_tr_bt = bootstrap_augment(Xg_tr, yg_tr, n_boot)
            Xr_tr_bt, yr_tr_bt = bootstrap_augment(Xr_tr, yr_tr, n_boot)
            group_bt_pipe = fit_pipeline_with_params(
                spec["estimator"], best_params, numeric_cols, categorical_cols,
                Xg_tr_bt, yg_tr_bt)
            random_bt_pipe = fit_pipeline_with_params(
                spec["estimator"], best_params, numeric_cols, categorical_cols,
                Xr_tr_bt, yr_tr_bt)
            m_group_bt = regression_metrics(yg_te, group_bt_pipe.predict(Xg_te))
            m_random_bt = regression_metrics(yr_te, random_bt_pipe.predict(Xr_te))
            print(f"[GROUP / RD+BT]       R2={m_group_bt['R2']:.3f}  "
                  f"RMSE={m_group_bt['RMSE']:.3f}  (Delta RMSE vs RD = "
                  f"{m_group_bt['RMSE'] - m_group['RMSE']:+.3f})")
            print(f"[RANDOM / RD+BT]      R2={m_random_bt['R2']:.3f}  "
                  f"RMSE={m_random_bt['RMSE']:.3f}  (Delta RMSE vs RD = "
                  f"{m_random_bt['RMSE'] - m_random['RMSE']:+.3f})")
        else:
            m_group_bt = m_random_bt = None

        # --- Record for the comparison tables (one row per condition).
        comparison_rows.append({"Model": name, "Validation Strategy": "Group-Aware (RD)",
                                **{k: m_group[k] for k in METRIC_KEYS}})
        comparison_rows.append({"Model": name, "Validation Strategy": "Random Split (RD)",
                                **{k: m_random[k] for k in METRIC_KEYS}})
        if m_group_bt is not None:
            comparison_rows.append({"Model": name, "Validation Strategy": "Group-Aware (RD+BT)",
                                    **{k: m_group_bt[k] for k in METRIC_KEYS}})
            comparison_rows.append({"Model": name, "Validation Strategy": "Random Split (RD+BT)",
                                    **{k: m_random_bt[k] for k in METRIC_KEYS}})

        # Generalization gap is defined on the un-augmented (RD) results.
        gap_rows.append({
            "Model": name,
            "Random RMSE": m_random["RMSE"],
            "Group RMSE": m_group["RMSE"],
            "Generalization Gap": m_group["RMSE"] - m_random["RMSE"],
        })
        # Augmentation effect table (negative Delta = BT improved RMSE).
        aug_rows.append({
            "Model": name,
            "Group RMSE (RD)": m_group["RMSE"],
            "Group RMSE (RD+BT)": m_group_bt["RMSE"] if m_group_bt else np.nan,
            "Group dRMSE (BT-RD)": (m_group_bt["RMSE"] - m_group["RMSE"]) if m_group_bt else np.nan,
            "Random RMSE (RD)": m_random["RMSE"],
            "Random RMSE (RD+BT)": m_random_bt["RMSE"] if m_random_bt else np.nan,
            "Random dRMSE (BT-RD)": (m_random_bt["RMSE"] - m_random["RMSE"]) if m_random_bt else np.nan,
        })

        # --- Per-model diagnostic plots (computed on the PRIMARY group split).
        slug = _slug(name)
        plot_pred_vs_true(yg_te, yg_pred,
                          f"{name} -- Predicted vs Experimental (unseen papers)",
                          m_group, out_dir / f"pred_vs_true_{slug}.png")
        plot_residuals(yg_te, yg_pred, f"{name} -- Residuals (unseen papers)",
                       out_dir / f"residuals_{slug}.png")
        plot_error_hist(yg_te, yg_pred, f"{name} -- Error histogram (unseen papers)",
                        out_dir / f"error_hist_{slug}.png")

        # --- Per-paper generalisation analysis (group split only).
        per_paper_analysis(yg_te, yg_pred, groups_te, out_dir, name)

        # --- Feature importance for the tree-based models.
        if name in ("Decision Tree", "XGBoost"):
            pre = group_pipe.named_steps["preprocess"]
            mdl = group_pipe.named_steps["model"]
            feat_names = get_output_feature_names(pre, numeric_cols, categorical_cols)
            importances = mdl.feature_importances_
            imp_df = (
                pd.DataFrame({"feature": feat_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
            imp_df.to_csv(out_dir / f"feature_importance_{slug}.csv", index=False)
            plot_importance_bar(feat_names, importances,
                                f"{name} -- Top 20 feature importances",
                                out_dir / f"feature_importance_{slug}.png", top_n=20)
            print(f"Top 10 features ({name}):")
            print(imp_df.head(10).to_string(index=False))

        # --- Persist the best (group-aware, RD) model for this estimator family.
        model_file = {
            "Decision Tree": "best_decision_tree.joblib",
            "XGBoost": "best_xgboost.joblib",
            "KNN": "best_knn.joblib",
            "MLP": "best_mlp.joblib",
        }[name]
        joblib.dump(group_pipe, out_dir / model_file)
        print(f"Saved best {name} model -> {out_dir / model_file}")

    # --------------------------------------------------- model selection
    _print_header("STEP 8  |  MODEL SELECTION (by GroupKFold CV RMSE)")
    best_model_name = min(cv_rmse_store, key=cv_rmse_store.get)
    print("GroupKFold CV RMSE per model (lower = better extrapolation):")
    for nm, v in sorted(cv_rmse_store.items(), key=lambda kv: kv[1]):
        flag = "  <-- BEST" if nm == best_model_name else ""
        print(f"   {nm:<20s} CV RMSE = {v:.4f}{flag}")
    print(f"\nSelected best model for unseen-paper prediction: {best_model_name}")
    best_pipe = fitted_group_models[best_model_name]

    # ----------------------------------------- permutation importance (best)
    _print_header("STEP 9  |  PERMUTATION IMPORTANCE (best group-aware model)")
    # Permutation importance is model-agnostic and is computed on the held-out
    # *unseen papers* so it reflects what actually drives extrapolation accuracy.
    perm = permutation_importance(
        best_pipe, Xg_te, yg_te, n_repeats=10,
        random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    # Importances are over the *input* columns (before one-hot), which is the
    # most interpretable level for permutation importance.
    perm_df = (
        pd.DataFrame({
            "feature": list(X.columns),
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    perm_df.to_csv(out_dir / "permutation_importance.csv", index=False)
    plot_importance_bar(
        list(perm_df["feature"]), list(perm_df["importance_mean"]),
        f"Permutation importance -- {best_model_name} (unseen papers)",
        out_dir / "permutation_importance.png", top_n=min(20, len(perm_df)),
    )
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
    # Sort by the primary Group-Aware (RD) RMSE per model, then by strategy.
    group_rmse_order = (
        comp_df[comp_df["Validation Strategy"] == "Group-Aware (RD)"]
        .set_index("Model")["RMSE"].sort_values()
    )
    comp_df["__order"] = comp_df["Model"].map(
        {m: i for i, m in enumerate(group_rmse_order.index)}
    )
    comp_df = (
        comp_df.sort_values(["__order", "Validation Strategy"])
        .drop(columns="__order")
        .reset_index(drop=True)
    )
    comp_df_round = comp_df.copy()
    for col in METRIC_KEYS:
        comp_df_round[col] = comp_df_round[col].astype(float).round(4)
    comp_df_round.to_csv(out_dir / "model_comparison.csv", index=False)
    print("Full comparison (sorted by primary Group-Aware (RD) RMSE):")
    print(comp_df_round.to_string(index=False))

    gap_df = (
        pd.DataFrame(gap_rows)
        .sort_values("Group RMSE")
        .reset_index(drop=True)
    )
    for col in ["Random RMSE", "Group RMSE", "Generalization Gap"]:
        gap_df[col] = gap_df[col].astype(float).round(4)
    gap_df.to_csv(out_dir / "generalization_gap.csv", index=False)
    print("\nGeneralization gap (Group RMSE - Random RMSE; larger = more")
    print("paper-specific overfitting):")
    print(gap_df.to_string(index=False))

    # --- Bootstrap augmentation effect table + bar chart (RD vs RD+BT).
    if n_boot > 0:
        aug_df = pd.DataFrame(aug_rows).sort_values("Group RMSE (RD)").reset_index(drop=True)
        for col in aug_df.columns:
            if col != "Model":
                aug_df[col] = aug_df[col].astype(float).round(4)
        aug_df.to_csv(out_dir / "augmentation_comparison.csv", index=False)
        print(f"\nBootstrap augmentation effect (+{n_boot} rows; negative dRMSE = BT helped):")
        print(aug_df.to_string(index=False))

        # Grouped bar chart of RMSE across the four conditions per model.
        order = list(aug_df["Model"])
        labels = ["Group RD", "Group RD+BT", "Random RD", "Random RD+BT"]
        cols = ["Group RMSE (RD)", "Group RMSE (RD+BT)", "Random RMSE (RD)", "Random RMSE (RD+BT)"]
        x = np.arange(len(order))
        w = 0.2
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["indianred", "darkorange", "steelblue", "skyblue"]
        for i, (lab, col, c) in enumerate(zip(labels, cols, colors)):
            vals = [float(aug_df.loc[aug_df["Model"] == m, col].values[0]) for m in order]
            ax.bar(x + (i - 1.5) * w, vals, width=w, label=lab, color=c, edgecolor="k")
        ax.set_xticks(x); ax.set_xticklabels(order)
        ax.set_ylabel("RMSE (lower is better)")
        ax.set_title(f"FSR RMSE by model and data condition (RD vs RD+BT, +{n_boot} rows)")
        ax.legend(); ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "augmentation_comparison.png", dpi=150)
        plt.close(fig)
    else:
        aug_df = pd.DataFrame(aug_rows)

    # ----------------------------------------------------------- metrics dump
    metrics_blob = {
        "random_state": RANDOM_STATE,
        "n_rows_initial": int(n_rows_initial),
        "n_rows_final": int(n_after),
        "target_column": target_col,
        "group_column": group_col,
        "n_unique_papers": int(groups_all.nunique()),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "best_model": best_model_name,
        "bootstrap_n": n_boot,
        "models": list(models.keys()),
        "cv_group_rmse": {k: float(v) for k, v in cv_rmse_store.items()},
        "best_params": best_params_store,
        "comparison": comp_df_round.to_dict(orient="records"),
        "generalization_gap": gap_df.to_dict(orient="records"),
        "augmentation_comparison": aug_df.to_dict(orient="records"),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_blob, f, indent=2, default=str)

    _print_header("DONE")
    print(f"All artefacts written to: {out_dir.resolve()}")
    print("Key files:")
    print("  - model_comparison.csv         (R2/RMSE/MAE/MAPE/NRMSE/MBE; 4 conditions)")
    print("  - generalization_gap.csv       (random vs group RMSE + gap)")
    print("  - augmentation_comparison.csv  (RD vs RD+BT bootstrap effect)")
    print("  - metrics.json                 (machine-readable summary)")
    print("  - pred_vs_true_*.png / residuals_*.png / error_hist_*.png")
    print("  - per_paper_*                  (per-paper extrapolation analysis)")
    print("  - feature_importance_*         (Decision Tree & XGBoost)")
    print("  - permutation_importance.*     (best model)")
    print("  - shap_summary_* / shap_bar_*  (best model)")
    print("  - bootstrap_fsr_distribution.png / augmentation_comparison.png")
    print("  - best_decision_tree.joblib / best_xgboost.joblib / best_knn.joblib / best_mlp.joblib")
    print(f"\nPRIMARY scientific result = the Group-Aware (RD) rows. Best model: {best_model_name}.")


if __name__ == "__main__":
    main()
