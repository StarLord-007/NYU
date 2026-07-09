#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fsr_extrapolation_regression.py
===============================

Paper-aware (extrapolation-first) machine-learning study of **Flame Spread Rate
(FSR)** for the microgravity-combustion literature database
(``Microgravity_Database_reduced.csv``, or the original
``Microgravity_Database.xlsm`` workbook).

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

That is a **extrapolation** problem, and the only honest way to estimate it is
to make sure that *all rows of any given paper are kept together* -- entirely in
train OR entirely in test, never split across the two. This is "group-aware" or
"leave-papers-out" evaluation, implemented here with repeated ``GroupShuffleSplit``
hold-outs (for the benchmark) and ``GroupKFold`` (for cross-validation and
hyper-parameter tuning), using the paper identifier as the grouping key.

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

Repeated train/test averaging (10 repeats by default)
-----------------------------------------------------
To get stable, defensible metric estimates the whole tune+evaluate cycle is
repeated ``--n-repeats`` times (DEFAULT 10) with distinct random seeds. For each
repeat we build a fresh random split *and* a fresh paper-grouped split, re-run
GroupKFold hyper-parameter tuning, fit the four conditions (Random/RD,
Random/RD+BT, Group/RD, Group/RD+BT) and store every metric. The headline
tables (``model_comparison.csv``, ``generalization_gap.csv``,
``augmentation_comparison.csv``) report mean +/- std across the 10 repeats; the
raw per-repeat rows are also saved (``*_repeats.csv``) and a comparison bar
chart with error bars is produced (``model_comparison_bars.png``) so the
run-to-run variability is visible. Per-repeat predictions are dumped to
``predictions/`` and per-repeat best hyper-parameters to
``best_params_repeats.csv``.

Run
---
    python fsr_extrapolation_regression3.py
    # optional flags:
    python fsr_extrapolation_regression3.py --data Microgravity_Database_reduced.csv \
        --out results --n-iter 40 --no-shap --bootstrap-n 1000 --n-repeats 10

    # The loader also still accepts the original Excel workbook:
    python fsr_extrapolation_regression3.py --data Microgravity_Database.xlsm

Data input
----------
The script accepts either the CSV export (``Microgravity_Database_reduced.csv``,
the new default) or the original Excel workbook (``Microgravity_Database.xlsm``).
Both share the same two-row "section banner / field name" header, which the
loader flattens automatically. All column-role detection (target, paper id,
leakage, numeric/categorical) is done by heuristics, so renamed / reordered
columns are tolerated.

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

def _load_csv_database(path: Path) -> pd.DataFrame:
    """Load the microgravity database from a CSV export.

    The CSV export (``Microgravity_Database_reduced.csv``) mirrors the Excel
    sheet: a *section banner* first row (Citation / Sample / Flow / Extra /
    Outputs / Info) sits above the real field-name row. We therefore skip the
    first row and treat the second as the header. Windows exports use ``\\r\\n``
    line endings and cp1252 bytes, so we try several encodings. A single-header
    read is attempted as a fallback if the banner-skip parse does not surface an
    FSR-like column.
    """
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

    def _read(skiprows: int) -> pd.DataFrame | None:
        for enc in encodings:
            try:
                df = pd.read_csv(path, skiprows=skiprows, encoding=enc)
            except UnicodeDecodeError:
                continue
            except Exception:
                return None
            df.columns = [str(c).strip() for c in df.columns]
            return df
        return None

    best_df = None
    best_score = -1
    for skip in (1, 0):  # banner-skip first, then plain single header
        df = _read(skip)
        if df is None:
            continue
        has_fsr = any(_looks_like_target(c) for c in df.columns)
        score = len(df) * df.shape[1] + (10_000_000 if has_fsr else 0)
        if score > best_score:
            best_score, best_df = score, df

    if best_df is None:
        raise RuntimeError(f"Could not read CSV database {path} under any encoding.")

    # Drop unlabeled 'Unnamed: N' columns. In this CSV export they are the
    # trailing free-text "extra info" cells (e.g. "Orientation=horizontal",
    # "T_init=20C") with no field name of their own. They carry no
    # generalisable numeric signal and, worse, _first_number() would pick a
    # spurious number out of them ("T=20C" -> 20) and mis-file the column as a
    # numeric feature. Removing them keeps the feature matrix honest.
    unnamed = [c for c in best_df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        best_df = best_df.drop(columns=unnamed)
    # Drop duplicated + fully-empty columns, then tidy text cells once.
    best_df = best_df.loc[:, ~best_df.columns.duplicated()]
    best_df = best_df.dropna(axis=1, how="all")
    for c in best_df.columns:
        if best_df[c].dtype == object:
            best_df[c] = best_df[c].map(_clean_text)
    return best_df.reset_index(drop=True)


def load_database(path: Path) -> pd.DataFrame:
    """Load the microgravity database into a flat, single-header DataFrame.

    Supports both the Excel workbook (``.xlsm`` / ``.xlsx``) and the CSV export
    (``.csv``). For CSV the loading is delegated to :func:`_load_csv_database`.

    The primary Excel data sheet ("Sheet2") uses a *two-level* header: a top
    "section" banner row (Citation / Sample / Flow / Outputs / Info) above the
    real field names. We try that layout first and flatten it (keeping the
    descriptive field name). If anything about the layout differs we gracefully
    fall back to a single-header read and finally to whatever sheet/parse
    succeeds, so the script does not hard-fail on a re-saved workbook.
    """
    # --- CSV export -------------------------------------------------------
    if path.suffix.lower() == ".csv":
        return _load_csv_database(path)

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

    One important guard: chemical-formula / label columns such as the diluent
    ("N2", "CO2", "Ar", "He") would otherwise be *mis-classified* as numeric,
    because ``_first_number`` extracts the trailing subscript ("N2" -> 2,
    "CO2" -> 2). We detect this case -- the values are dominated by short tokens
    whose *first character is a letter* -- and force such columns to stay
    categorical, which is what they physically are.

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
        if frac >= numeric_frac_threshold and not _looks_like_label_column(series):
            out[c] = parsed  # commit the numeric parse
            numeric_cols.append(c)
        else:
            out[c] = series.map(_clean_text)
            categorical_cols.append(c)
    return numeric_cols, categorical_cols, out


# Matches a token that is really a category label rather than a measurement:
# it begins with a letter and is short, e.g. a chemical formula ("N2", "CO2",
# "Ar", "He") or a code. Genuine numeric cells ("94 W", "-60 mm/s", "101.3")
# begin with a digit or sign, so they never match.
_LABEL_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\-/]{0,9}$")


def _looks_like_label_column(series: pd.Series, label_frac_threshold: float = 0.60) -> bool:
    """True when a string column is dominated by short letter-led label tokens.

    Used to stop ``_first_number`` from turning chemical-formula categoricals
    (diluent = N2 / CO2 / Ar / He) into bogus numeric features via their
    subscripts. We look at the *cleaned string* values (not the parsed number)
    and ask what fraction start with a letter and are short/formula-like.
    """
    vals = series.dropna().map(_clean_text).dropna().astype(str)
    if len(vals) == 0:
        return False
    label_like = vals.map(lambda s: bool(_LABEL_TOKEN_RE.match(s.strip())))
    return float(label_like.mean()) >= label_frac_threshold


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


def _safe_quantile_bins(values: np.ndarray, max_bins: int = 6) -> np.ndarray | None:
    """Return quantile bins with duplicate edges removed, or None if impossible."""
    series = pd.Series(np.asarray(values, dtype=float))
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 8:
        return None
    n_unique = int(series.nunique())
    n_bins = int(min(max_bins, max(2, n_unique)))
    if n_bins < 2:
        return None
    try:
        bins = pd.qcut(series, q=n_bins, labels=False, duplicates='drop')
    except Exception:
        return None
    bins = pd.Series(bins.astype(float), index=series.index)
    full = pd.Series(np.nan, index=pd.RangeIndex(len(values)), dtype=float)
    full.loc[bins.index] = bins.values
    return full.to_numpy()


def _distribution_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1 distance between two discrete distributions."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.sum() <= 0 and b.sum() <= 0:
        return 0.0
    a = a / max(a.sum(), 1e-12)
    b = b / max(b.sum(), 1e-12)
    return float(0.5 * np.abs(a - b).sum())


def _score_split_balance(
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_fraction: float,
    full_std: float,
    train_weights: np.ndarray | None = None,
    test_weights: np.ndarray | None = None,
) -> float:
    """Lower is better: prefer similar target moments and close test fraction."""
    y_train = np.asarray(y_train, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    full_std = float(full_std) if np.isfinite(full_std) and full_std > 1e-12 else 1.0

    score = abs(np.nanmean(y_train) - np.nanmean(y_test)) / full_std
    score += 0.5 * abs(np.nanmedian(y_train) - np.nanmedian(y_test)) / full_std
    score += 0.25 * abs(np.nanstd(y_train) - np.nanstd(y_test)) / full_std
    score += 2.0 * abs(len(y_test) / max(len(y_train) + len(y_test), 1) - test_fraction)

    if train_weights is not None and test_weights is not None:
        score += 1.5 * _distribution_distance(train_weights, test_weights)

    return float(score)


def _stratified_random_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float,
    seed: int,
    quantile_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Random hold-out, optionally stratified by target quantile bins."""
    y = np.asarray(y, dtype=float)
    bins = _safe_quantile_bins(y, max_bins=quantile_bins)
    stratify = None
    if bins is not None:
        bins_series = pd.Series(bins)
        counts = bins_series.dropna().astype(int).value_counts()
        if len(counts) >= 2 and counts.min() >= 2:
            stratify = bins_series.astype(int).to_numpy()

    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    meta = {'stratified_random': stratify is not None}
    return Xr_tr, Xr_te, yr_tr, yr_te, meta


def _candidate_group_split(
    groups: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
    n_candidates: int,
    quantile_bins: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Choose the most balanced GroupShuffleSplit candidate from many seeds."""
    groups = np.asarray(groups)
    y = np.asarray(y, dtype=float)
    full_std = float(np.nanstd(y))

    group_df = pd.DataFrame({'group': groups, 'y': y}).groupby('group', as_index=False).agg(
        y_mean=('y', 'mean'),
        y_median=('y', 'median'),
        y_count=('y', 'size'),
    )

    bin_values = _safe_quantile_bins(group_df['y_mean'].to_numpy(), max_bins=quantile_bins)
    if bin_values is not None and np.isfinite(bin_values).any():
        group_df['bin'] = pd.Series(bin_values).round().astype('Int64')
        max_bin = int(group_df['bin'].dropna().max())
    else:
        group_df['bin'] = pd.Series([pd.NA] * len(group_df), dtype='Int64')
        max_bin = -1

    best = None
    best_score = np.inf
    n_candidates = max(1, int(n_candidates))

    for k in range(n_candidates):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + 1000 * k)
        tr_idx, te_idx = next(splitter.split(np.zeros(len(y)), y, groups=groups))
        y_tr = y[tr_idx]
        y_te = y[te_idx]

        tr_groups = np.unique(groups[tr_idx])
        te_groups = np.unique(groups[te_idx])

        if max_bin >= 0:
            tr_bins = group_df.loc[group_df['group'].isin(tr_groups), 'bin'].dropna().astype(int)
            te_bins = group_df.loc[group_df['group'].isin(te_groups), 'bin'].dropna().astype(int)
            tr_hist = np.bincount(tr_bins.to_numpy(), minlength=max_bin + 1) if len(tr_bins) else np.zeros(max_bin + 1)
            te_hist = np.bincount(te_bins.to_numpy(), minlength=max_bin + 1) if len(te_bins) else np.zeros(max_bin + 1)
            bin_distance = _distribution_distance(tr_hist, te_hist)
        else:
            tr_hist = te_hist = None
            bin_distance = 0.0

        score = _score_split_balance(
            y_tr,
            y_te,
            test_size,
            full_std,
            train_weights=tr_hist,
            test_weights=te_hist,
        )
        score += bin_distance

        if score < best_score:
            best_score = score
            best = (
                tr_idx,
                te_idx,
                {
                    'balance_score': float(score),
                    'target_bin_distance': float(bin_distance),
                },
            )

    assert best is not None
    return best


def build_holdout_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
    repeat_idx: int,
    n_split_candidates: int = 50,
    target_bins: int = 6,
) -> dict:
    """Build one random split and one paper-grouped split for a repeat.

    The random split is the interpolation baseline. The grouped split uses a
    balance-aware GroupShuffleSplit search so every paper stays entirely in
    train or entirely in test, while also avoiding pathological target skews
    that can make R² swing wildly between repeats.
    """
    seed = random_state + repeat_idx

    Xr_tr, Xr_te, yr_tr, yr_te, random_meta = _stratified_random_split(
        X, y, test_size=test_size, seed=seed, quantile_bins=target_bins
    )

    g_tr_idx, g_te_idx, group_meta = _candidate_group_split(
        groups=groups,
        y=y,
        test_size=test_size,
        seed=seed,
        n_candidates=n_split_candidates,
        quantile_bins=target_bins,
    )
    Xg_tr, Xg_te = X.iloc[g_tr_idx], X.iloc[g_te_idx]
    yg_tr, yg_te = y[g_tr_idx], y[g_te_idx]
    groups_tr, groups_te = groups[g_tr_idx], groups[g_te_idx]

    return {
        'repeat': repeat_idx + 1,
        'seed': seed,
        'random_stratified': bool(random_meta.get('stratified_random', False)),
        'group_balance_score': float(group_meta.get('balance_score', np.nan)),
        'group_target_bin_distance': float(group_meta.get('target_bin_distance', np.nan)),
        'Xr_tr': Xr_tr,
        'Xr_te': Xr_te,
        'yr_tr': yr_tr,
        'yr_te': yr_te,
        'Xg_tr': Xg_tr,
        'Xg_te': Xg_te,
        'yg_tr': yg_tr,
        'yg_te': yg_te,
        'groups_tr': groups_tr,
        'groups_te': groups_te,
        'n_train_papers': int(pd.Series(groups_tr).nunique()),
        'n_test_papers': int(pd.Series(groups_te).nunique()),
        'y_train_mean': float(np.mean(yg_tr)),
        'y_test_mean': float(np.mean(yg_te)),
        'y_train_std': float(np.std(yg_tr)),
        'y_test_std': float(np.std(yg_te)),
    }


def summarize_metric_rows(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    """Aggregate repeated benchmark rows into mean/std summary tables."""
    summary = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(['mean', 'std'])
        .reset_index()
    )
    summary.columns = [
        '_'.join([part for part in col if part]).rstrip('_') if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def select_representative_repeat(
    df: pd.DataFrame,
    model_name: str,
    strategy_name: str = "Group-Aware (RD)",
    metric_name: str = "RMSE",
) -> int:
    """Choose the repeat whose metric is closest to the model's mean metric."""
    subset = df[(df["Model"] == model_name) & (df["Validation Strategy"] == strategy_name)]
    if subset.empty:
        raise RuntimeError(f"No rows found for model={model_name!r}, strategy={strategy_name!r}")
    target = float(subset[f"{metric_name}_mean"].mean()) if f"{metric_name}_mean" in subset.columns else float(subset[metric_name].mean())
    if f"{metric_name}_mean" in subset.columns:
        distance = (subset[f"{metric_name}_mean"] - target).abs()
    else:
        distance = (subset[metric_name] - target).abs()
    return int(subset.iloc[int(distance.idxmin())]["Repeat"])


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
    parser.add_argument("--data", default="Microgravity_Database_reduced.csv",
                        help="Path to the database (CSV export or Excel workbook; "
                             "auto-located if not found in the current directory). "
                             "Defaults to the reduced CSV export.")
    parser.add_argument("--out", default=str(SCRIPT_DIR / "results"),
                        help="Output directory for all artefacts "
                             "(defaults to a 'results' folder next to this script).")
    parser.add_argument("--n-iter", type=int, default=40,
                        help="RandomizedSearchCV iterations per model.")
    parser.add_argument("--cv-splits", type=int, default=5,
                        help="Number of GroupKFold splits for tuning.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test fraction for both split strategies.")
    parser.add_argument("--n-repeats", type=int, default=10,
                        help="Number of repeated train/test benchmarks to run and average. "
                             "Each repeat re-tunes hyper-parameters with a new seed and "
                             "evaluates all four conditions; final tables report mean +/- std.")
    parser.add_argument("--save-repeat-models", action="store_true",
                        help="Also persist every per-repeat best pipeline under "
                             "results/models_per_repeat/ (may consume substantial disk).")
    parser.add_argument("--bootstrap-n", type=int, default=1000,
                        help="Number of bootstrap rows to ADD to the training set "
                             "(founding paper uses 1000). Set 0 to disable BT.")
    parser.add_argument("--split-candidates", type=int, default=50,
                        help="Number of candidate GroupShuffleSplit seeds evaluated per repeat; "
                             "the most balanced one is selected.")
    parser.add_argument("--target-bins", type=int, default=6,
                        help="Quantile bins used to stabilize the random split and the group split "
                             "selection by target distribution.")
    parser.add_argument("--no-shap", action="store_true",
                        help="Disable the (slower) SHAP analysis.")
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-repeat artefact subdirectories.
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "save_repeat_models", False):
        repeat_models_dir = out_dir / "models_per_repeat"
        repeat_models_dir.mkdir(parents=True, exist_ok=True)
    else:
        repeat_models_dir = None

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

    # ------------------------------------------------ repeated benchmark setup
    _print_header("STEP 6  |  REPEATED TRAIN/TEST BENCHMARK")
    n_repeats = max(1, int(args.n_repeats))
    if n_repeats != args.n_repeats:
        print(f"Requested {args.n_repeats} repeats, using {n_repeats}.")

    n_groups = pd.Series(groups).nunique()
    if n_groups < 2:
        sys.exit("ERROR: need at least 2 papers for group validation.")

    models = get_models_and_spaces()
    METRIC_KEYS = ["R2", "RMSE", "MAE", "MAPE", "NRMSE", "MBE"]
    comparison_rows = []
    gap_rows = []
    aug_rows = []
    tune_rows = []
    split_meta_by_repeat = {}
    best_params_by_model_repeat = {}
    n_boot = max(0, int(args.bootstrap_n))

    if n_boot > 0:
        _print_header("STEP 6b |  BOOTSTRAP DATA AUGMENTATION (Rivera et al.)")
        print(f"Bootstrap resampling enabled: +{n_boot} rows sampled WITH replacement from the TRAINING split only.")
        print("The test set is never augmented (no leakage).")
    else:
        print("Bootstrap augmentation disabled (--bootstrap-n 0).")

    for repeat_idx in range(n_repeats):
        split = build_holdout_splits(
            X, y, groups, args.test_size, RANDOM_STATE, repeat_idx,
            n_split_candidates=args.split_candidates,
            target_bins=args.target_bins,
        )
        split_meta_by_repeat[split["repeat"]] = split

        print(
            f"\nRepeat {split['repeat']:02d}/{n_repeats:02d} | "
            f"Random train={len(split['Xr_tr'])} test={len(split['Xr_te'])} "
            f"({'stratified' if split['random_stratified'] else 'unstratified'}) | "
            f"Group train={len(split['Xg_tr'])} ({split['n_train_papers']} papers) "
            f"test={len(split['Xg_te'])} ({split['n_test_papers']} papers) | "
            f"balance={split['group_balance_score']:.3f}"
        )

        n_cv = max(2, min(args.cv_splits, split["n_train_papers"]))
        if n_cv < args.cv_splits:
            print(f"  (reduced GroupKFold splits to {n_cv} to match #train papers)")
        group_cv = GroupKFold(n_splits=n_cv)

        if n_boot > 0 and repeat_idx == 0:
            _, yg_tr_bt = bootstrap_augment(split["Xg_tr"], split["yg_tr"], n_boot)
            fig, ax = plt.subplots(figsize=(6.5, 4.4))
            bins = np.linspace(float(np.min(y)), float(np.max(y)), 40)
            ax.hist(split["yg_tr"], bins=bins, alpha=0.6,
                    label=f"Real train (n={len(split['yg_tr'])})", color="steelblue", edgecolor="k")
            ax.hist(yg_tr_bt, bins=bins, alpha=0.5,
                    label=f"Real + Bootstrap (n={len(yg_tr_bt)})", color="darkorange", edgecolor="k")
            ax.set_xlabel("Flame Spread Rate")
            ax.set_ylabel("Frequency")
            ax.set_title("Bootstrap augmentation: FSR distribution (training split)")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / "bootstrap_fsr_distribution.png", dpi=150)
            plt.close(fig)
            print("  Saved bootstrap FSR-distribution figure -> bootstrap_fsr_distribution.png")

        for name, spec in models.items():
            _print_header(f"STEP 7  |  TUNE + EVALUATE (repeat {split['repeat']:02d}) : {name}")

            pipe = Pipeline(
                steps=[("preprocess", preprocessor), ("model", spec["estimator"])]
            )

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
            search.fit(split["Xg_tr"], split["yg_tr"], groups=split["groups_tr"])
            best_cv_rmse = -search.best_score_
            best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
            best_params_by_model_repeat.setdefault(name, {})[split["repeat"]] = best_params
            tune_rows.append({
                "Repeat": split["repeat"],
                "Seed": split["seed"],
                "Model": name,
                "Group CV RMSE": best_cv_rmse,
                "n_train_papers": split["n_train_papers"],
                "n_test_papers": split["n_test_papers"],
                "group_balance_score": split["group_balance_score"],
                "group_target_bin_distance": split["group_target_bin_distance"],
                "random_stratified": split["random_stratified"],
            })

            group_pipe = search.best_estimator_
            yg_pred = group_pipe.predict(split["Xg_te"])
            m_group = regression_metrics(split["yg_te"], yg_pred)
            print(f"  [GROUP/UNSEEN-PAPERS] R2={m_group['R2']:.3f}  RMSE={m_group['RMSE']:.3f}  "
                  f"MAE={m_group['MAE']:.3f}  MAPE={m_group['MAPE']:.1f}%  "
                  f"NRMSE={m_group['NRMSE']:.3f}  MBE={m_group['MBE']:.3f}")

            random_pipe = fit_pipeline_with_params(
                spec["estimator"], best_params, numeric_cols, categorical_cols,
                split["Xr_tr"], split["yr_tr"]
            )
            yr_pred = random_pipe.predict(split["Xr_te"])
            m_random = regression_metrics(split["yr_te"], yr_pred)
            print(f"  [RANDOM / RD]         R2={m_random['R2']:.3f}  RMSE={m_random['RMSE']:.3f}  "
                  f"MAE={m_random['MAE']:.3f}  NRMSE={m_random['NRMSE']:.3f}")

            yg_pred_bt = yr_pred_bt = None
            if n_boot > 0:
                Xg_tr_bt, yg_tr_bt = bootstrap_augment(split["Xg_tr"], split["yg_tr"], n_boot)
                Xr_tr_bt, yr_tr_bt = bootstrap_augment(split["Xr_tr"], split["yr_tr"], n_boot)
                group_bt_pipe = fit_pipeline_with_params(
                    spec["estimator"], best_params, numeric_cols, categorical_cols,
                    Xg_tr_bt, yg_tr_bt
                )
                random_bt_pipe = fit_pipeline_with_params(
                    spec["estimator"], best_params, numeric_cols, categorical_cols,
                    Xr_tr_bt, yr_tr_bt
                )
                yg_pred_bt = group_bt_pipe.predict(split["Xg_te"])
                yr_pred_bt = random_bt_pipe.predict(split["Xr_te"])
                m_group_bt = regression_metrics(split["yg_te"], yg_pred_bt)
                m_random_bt = regression_metrics(split["yr_te"], yr_pred_bt)
                print(f"  [GROUP / RD+BT]       R2={m_group_bt['R2']:.3f}  RMSE={m_group_bt['RMSE']:.3f}  "
                      f"(Delta RMSE vs RD = {m_group_bt['RMSE'] - m_group['RMSE']:+.3f})")
                print(f"  [RANDOM / RD+BT]      R2={m_random_bt['R2']:.3f}  RMSE={m_random_bt['RMSE']:.3f}  "
                      f"(Delta RMSE vs RD = {m_random_bt['RMSE'] - m_random['RMSE']:+.3f})")
            else:
                m_group_bt = m_random_bt = None

            # ------------------------------------------------------------------
            # Persist per-repeat predictions so reviewers can recompute any metric
            # downstream and audit the run-to-run variability.
            # ------------------------------------------------------------------
            slug_model = _slug(name)
            pred_group_df = pd.DataFrame({
                "paper": split["groups_te"],
                "y_true": split["yg_te"],
                "y_pred_RD": yg_pred,
            })
            if yg_pred_bt is not None:
                pred_group_df["y_pred_RD_BT"] = yg_pred_bt
            pred_group_df.to_csv(
                pred_dir / f"predictions_group_repeat{split['repeat']:02d}_{slug_model}.csv",
                index=False,
            )

            pred_random_df = pd.DataFrame({
                "y_true": split["yr_te"],
                "y_pred_RD": yr_pred,
            })
            if yr_pred_bt is not None:
                pred_random_df["y_pred_RD_BT"] = yr_pred_bt
            pred_random_df.to_csv(
                pred_dir / f"predictions_random_repeat{split['repeat']:02d}_{slug_model}.csv",
                index=False,
            )

            if repeat_models_dir is not None:
                joblib.dump(
                    group_pipe,
                    repeat_models_dir / f"repeat{split['repeat']:02d}_{slug_model}_group.joblib",
                )

            comparison_rows.append({
                "Repeat": split["repeat"],
                "Seed": split["seed"],
                "Model": name,
                "Validation Strategy": "Group-Aware (RD)",
                **{k: m_group[k] for k in METRIC_KEYS},
            })
            comparison_rows.append({
                "Repeat": split["repeat"],
                "Seed": split["seed"],
                "Model": name,
                "Validation Strategy": "Random Split (RD)",
                **{k: m_random[k] for k in METRIC_KEYS},
            })
            if m_group_bt is not None:
                comparison_rows.append({
                    "Repeat": split["repeat"],
                    "Seed": split["seed"],
                    "Model": name,
                    "Validation Strategy": "Group-Aware (RD+BT)",
                    **{k: m_group_bt[k] for k in METRIC_KEYS},
                })
                comparison_rows.append({
                    "Repeat": split["repeat"],
                    "Seed": split["seed"],
                    "Model": name,
                    "Validation Strategy": "Random Split (RD+BT)",
                    **{k: m_random_bt[k] for k in METRIC_KEYS},
                })

            gap_rows.append({
                "Repeat": split["repeat"],
                "Seed": split["seed"],
                "Model": name,
                "Random RMSE": m_random["RMSE"],
                "Group RMSE": m_group["RMSE"],
                "Generalization Gap": m_group["RMSE"] - m_random["RMSE"],
            })
            aug_rows.append({
                "Repeat": split["repeat"],
                "Seed": split["seed"],
                "Model": name,
                "Group RMSE (RD)": m_group["RMSE"],
                "Group RMSE (RD+BT)": m_group_bt["RMSE"] if m_group_bt else np.nan,
                "Group dRMSE (BT-RD)": (m_group_bt["RMSE"] - m_group["RMSE"]) if m_group_bt else np.nan,
                "Random RMSE (RD)": m_random["RMSE"],
                "Random RMSE (RD+BT)": m_random_bt["RMSE"] if m_random_bt else np.nan,
                "Random dRMSE (BT-RD)": (m_random_bt["RMSE"] - m_random["RMSE"]) if m_random_bt else np.nan,
            })

    # ------------------------------------------------- summary tables
    _print_header("STEP 8  |  MODEL SELECTION (mean GroupKFold CV RMSE)")
    tune_raw = pd.DataFrame(tune_rows)
    tune_summary = summarize_metric_rows(tune_raw, ["Model"], ["Group CV RMSE"])
    tune_summary = tune_summary.sort_values("Group CV RMSE_mean").reset_index(drop=True)
    print("Repeated GroupKFold CV RMSE per model (mean +/- std; lower is better):")
    print(tune_summary.to_string(index=False))

    comp_raw = pd.DataFrame(comparison_rows)
    comp_summary = summarize_metric_rows(comp_raw, ["Model", "Validation Strategy"], METRIC_KEYS)
    group_rmse_order = (
        comp_summary[comp_summary["Validation Strategy"] == "Group-Aware (RD)"]
        .set_index("Model")["RMSE_mean"]
        .sort_values()
    )
    comp_summary["__order"] = comp_summary["Model"].map({m: i for i, m in enumerate(group_rmse_order.index)})
    comp_summary = (
        comp_summary.sort_values(["__order", "Validation Strategy"])
        .drop(columns="__order")
        .reset_index(drop=True)
    )

    gap_raw = pd.DataFrame(gap_rows)
    gap_summary = summarize_metric_rows(gap_raw, ["Model"], ["Random RMSE", "Group RMSE", "Generalization Gap"])
    gap_summary = gap_summary.sort_values("Group RMSE_mean").reset_index(drop=True)

    if n_boot > 0:
        aug_raw = pd.DataFrame(aug_rows)
        aug_summary = summarize_metric_rows(
            aug_raw,
            ["Model"],
            ["Group RMSE (RD)", "Group RMSE (RD+BT)", "Group dRMSE (BT-RD)",
             "Random RMSE (RD)", "Random RMSE (RD+BT)", "Random dRMSE (BT-RD)"]
        )
        aug_summary = aug_summary.sort_values("Group RMSE (RD)_mean").reset_index(drop=True)
    else:
        aug_raw = pd.DataFrame(aug_rows)
        aug_summary = aug_raw.copy()

    best_model_name = tune_summary.iloc[0]["Model"]
    print(f"\nSelected best model by mean repeated GroupKFold CV RMSE: {best_model_name}")

    representative_repeat_by_model = {}
    for name in models:
        subset = comp_raw[(comp_raw["Model"] == name) & (comp_raw["Validation Strategy"] == "Group-Aware (RD)")]
        target = float(subset["RMSE"].mean())
        rep_idx = (subset["RMSE"] - target).abs().idxmin()
        representative_repeat_by_model[name] = int(subset.loc[rep_idx, "Repeat"])

    best_pipe = None
    best_repeat_split = None

    _print_header("STEP 9  |  RE-TRAIN REPRESENTATIVE MODELS FOR OUTPUT FILES")
    fitted_group_models = {}
    best_params_store = {}
    cv_rmse_store = {row["Model"]: float(row["Group CV RMSE_mean"]) for _, row in tune_summary.iterrows()}

    for name, spec in models.items():
        rep = representative_repeat_by_model[name]
        split = split_meta_by_repeat[rep]
        best_params = best_params_by_model_repeat[name][rep]
        best_params_store[name] = best_params

        group_pipe = fit_pipeline_with_params(
            spec["estimator"], best_params, numeric_cols, categorical_cols,
            split["Xg_tr"], split["yg_tr"]
        )
        fitted_group_models[name] = group_pipe
        yg_pred = group_pipe.predict(split["Xg_te"])
        m_group = regression_metrics(split["yg_te"], yg_pred)
        print(f"{name:<15s} representative repeat {rep:02d}: group RMSE={m_group['RMSE']:.4f}")

        slug = _slug(name)
        plot_pred_vs_true(
            split["yg_te"], yg_pred,
            f"{name} -- Predicted vs Experimental (representative unseen papers)",
            m_group, out_dir / f"pred_vs_true_{slug}.png"
        )
        plot_residuals(split["yg_te"], yg_pred, f"{name} -- Residuals (representative unseen papers)",
                       out_dir / f"residuals_{slug}.png")
        plot_error_hist(split["yg_te"], yg_pred, f"{name} -- Error histogram (representative unseen papers)",
                        out_dir / f"error_hist_{slug}.png")
        per_paper_analysis(split["yg_te"], yg_pred, split["groups_te"], out_dir, name)

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

        model_file = {
            "Decision Tree": "best_decision_tree.joblib",
            "XGBoost": "best_xgboost.joblib",
            "KNN": "best_knn.joblib",
            "MLP": "best_mlp.joblib",
        }[name]
        joblib.dump(group_pipe, out_dir / model_file)
        print(f"Saved best {name} model -> {out_dir / model_file}")

        if name == best_model_name:
            best_pipe = group_pipe
            best_repeat_split = split

    if best_pipe is None or best_repeat_split is None:
        sys.exit("ERROR: failed to prepare the representative best model.")

    _print_header("STEP 10 |  PERMUTATION IMPORTANCE (best repeated benchmark model)")
    perm = permutation_importance(
        best_pipe, best_repeat_split["Xg_te"], best_repeat_split["yg_te"], n_repeats=10,
        random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
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
        f"Permutation importance -- {best_model_name} (representative unseen papers)",
        out_dir / "permutation_importance.png", top_n=min(20, len(perm_df)),
    )
    print(f"Permutation importance ranking ({best_model_name}):")
    print(perm_df.head(15).to_string(index=False))

    if not args.no_shap:
        _print_header("STEP 11 |  SHAP ANALYSIS (best repeated benchmark model)")
        run_shap(best_pipe, best_repeat_split["Xg_tr"], best_repeat_split["Xg_te"], numeric_cols, categorical_cols,
                 out_dir, best_model_name)
    else:
        print("\n[SHAP] skipped (--no-shap).")

    _print_header("STEP 12 |  MODEL COMPARISON TABLES")
    comp_raw_round = comp_raw.copy()
    for col in METRIC_KEYS:
        comp_raw_round[col] = comp_raw_round[col].astype(float).round(4)
    comp_raw_round.to_csv(out_dir / "model_comparison_repeats.csv", index=False)

    comp_summary_round = comp_summary.copy()
    for col in [c for c in comp_summary_round.columns if c not in {"Model", "Validation Strategy"}]:
        comp_summary_round[col] = comp_summary_round[col].astype(float).round(4)
    comp_summary_round.to_csv(out_dir / "model_comparison.csv", index=False)
    print("Repeated comparison summary (mean +/- std by validation strategy):")
    print(comp_summary_round.to_string(index=False))

    gap_raw_round = gap_raw.copy()
    for col in ["Random RMSE", "Group RMSE", "Generalization Gap"]:
        gap_raw_round[col] = gap_raw_round[col].astype(float).round(4)
    gap_raw_round.to_csv(out_dir / "generalization_gap_repeats.csv", index=False)

    gap_summary_round = gap_summary.copy()
    for col in [c for c in gap_summary_round.columns if c != "Model"]:
        gap_summary_round[col] = gap_summary_round[col].astype(float).round(4)
    gap_summary_round.to_csv(out_dir / "generalization_gap.csv", index=False)
    print("\nGeneralization gap summary (mean +/- std; larger = more paper-specific overfitting):")
    print(gap_summary_round.to_string(index=False))

    r2_stability = (
        comp_raw.groupby(["Model", "Validation Strategy"], dropna=False)["R2"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
        .sort_values(["Validation Strategy", "mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    r2_stability_round = r2_stability.copy()
    for col in [c for c in r2_stability_round.columns if c not in {"Model", "Validation Strategy"}]:
        r2_stability_round[col] = r2_stability_round[col].astype(float).round(4)
    r2_stability_round.to_csv(out_dir / "r2_stability.csv", index=False)
    print("\nR² stability across repeats (diagnostic):")
    print(r2_stability_round.to_string(index=False))

    if n_boot > 0:
        aug_raw_round = aug_raw.copy()
        for col in [c for c in aug_raw_round.columns if c not in {"Model"}]:
            aug_raw_round[col] = aug_raw_round[col].astype(float).round(4)
        aug_raw_round.to_csv(out_dir / "augmentation_comparison_repeats.csv", index=False)

        aug_summary_round = aug_summary.copy()
        for col in [c for c in aug_summary_round.columns if c != "Model"]:
            aug_summary_round[col] = aug_summary_round[col].astype(float).round(4)
        aug_summary_round.to_csv(out_dir / "augmentation_comparison.csv", index=False)
        print(f"\nBootstrap augmentation summary (+{n_boot} rows; negative dRMSE = BT helped):")
        print(aug_summary_round.to_string(index=False))

        order = list(aug_summary_round["Model"])
        labels = ["Group RD", "Group RD+BT", "Random RD", "Random RD+BT"]
        cols = ["Group RMSE (RD)_mean", "Group RMSE (RD+BT)_mean", "Random RMSE (RD)_mean", "Random RMSE (RD+BT)_mean"]
        x = np.arange(len(order))
        w = 0.2
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["indianred", "darkorange", "steelblue", "skyblue"]
        for i, (lab, col, c) in enumerate(zip(labels, cols, colors)):
            vals = [float(aug_summary_round.loc[aug_summary_round["Model"] == m, col].values[0]) for m in order]
            ax.bar(x + (i - 1.5) * w, vals, width=w, label=lab, color=c, edgecolor="k")
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_ylabel("RMSE (lower is better)")
        ax.set_title(f"FSR RMSE by model and data condition (mean over {n_repeats} repeats)")
        ax.legend()
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "augmentation_comparison.png", dpi=150)
        plt.close(fig)
    else:
        aug_raw_round = aug_raw.copy()
        aug_summary_round = aug_summary.copy()

    # ----------------------------------------------------------- per-repeat best-params dump
    bp_rows = []
    for mname, by_rep in best_params_by_model_repeat.items():
        for rep_i, params in by_rep.items():
            row = {"Model": mname, "Repeat": rep_i}
            row.update({f"param__{k}": v for k, v in params.items()})
            bp_rows.append(row)
    if bp_rows:
        pd.DataFrame(bp_rows).sort_values(["Model", "Repeat"]).to_csv(
            out_dir / "best_params_repeats.csv", index=False
        )

    split_diag_rows = [
        {
            "Repeat": rep,
            "Seed": meta["seed"],
            "Random stratified": meta["random_stratified"],
            "Group balance score": meta["group_balance_score"],
            "Group target-bin distance": meta["group_target_bin_distance"],
            "Train papers": meta["n_train_papers"],
            "Test papers": meta["n_test_papers"],
            "Group train y mean": meta["y_train_mean"],
            "Group test y mean": meta["y_test_mean"],
            "Group train y std": meta["y_train_std"],
            "Group test y std": meta["y_test_std"],
        }
        for rep, meta in sorted(split_meta_by_repeat.items())
    ]
    pd.DataFrame(split_diag_rows).to_csv(out_dir / "split_diagnostics.csv", index=False)

    # ----------------------------------------------------------- headline comparison bar chart
    # Mean +/- std RMSE per model under each validation strategy across the n_repeats runs.
    _print_header("STEP 12b |  HEADLINE COMPARISON FIGURE (mean +/- std across repeats)")
    strategy_order = [s for s in [
        "Random Split (RD)",
        "Random Split (RD+BT)",
        "Group-Aware (RD)",
        "Group-Aware (RD+BT)",
    ] if s in comp_summary["Validation Strategy"].unique()]
    model_order = list(group_rmse_order.index)
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(model_order) * len(strategy_order) / 2), 5.2))
    width = 0.8 / max(1, len(strategy_order))
    palette = ["#4C9AFF", "#80C2FF", "#E07A5F", "#F4A261"]
    x = np.arange(len(model_order))
    for i, strat in enumerate(strategy_order):
        means, stds = [], []
        for m in model_order:
            row = comp_summary[(comp_summary["Model"] == m) & (comp_summary["Validation Strategy"] == strat)]
            means.append(float(row["RMSE_mean"].values[0]) if len(row) else np.nan)
            stds.append(float(row["RMSE_std"].values[0]) if len(row) else 0.0)
        offset = (i - (len(strategy_order) - 1) / 2.0) * width
        ax.bar(x + offset, means, width=width, yerr=stds, capsize=4,
               label=strat, color=palette[i % len(palette)], edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(model_order)
    ax.set_ylabel("RMSE (lower = better)")
    ax.set_title(f"Model comparison across {n_repeats} repeats (mean +/- std)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison_bars.png", dpi=150)
    plt.close(fig)
    print("Saved headline comparison bar chart -> model_comparison_bars.png")

    # Also a per-repeat strip plot of Group-Aware (RD) RMSE so reviewers see variability.
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(model_order)), 4.5))
    for i, m in enumerate(model_order):
        sub = comp_raw[(comp_raw["Model"] == m) & (comp_raw["Validation Strategy"] == "Group-Aware (RD)")]
        rmses = sub["RMSE"].astype(float).values
        jitter = (np.random.RandomState(RANDOM_STATE + i).rand(len(rmses)) - 0.5) * 0.18
        ax.scatter(np.full_like(rmses, i, dtype=float) + jitter, rmses,
                   color="#E07A5F", edgecolor="k", s=42, zorder=3, label="per repeat" if i == 0 else None)
        if len(rmses):
            ax.hlines(rmses.mean(), i - 0.25, i + 0.25, colors="black", linewidth=2,
                      label="mean" if i == 0 else None)
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order)
    ax.set_ylabel("Group-Aware (RD) RMSE")
    ax.set_title(f"Per-repeat Group-Aware RMSE ({n_repeats} repeats)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "per_repeat_group_rmse_strip.png", dpi=150)
    plt.close(fig)
    print("Saved per-repeat strip plot      -> per_repeat_group_rmse_strip.png")

    # ----------------------------------------------------------- markdown headline report
    summary_lines = [
        "# FSR Extrapolation Regression -- Repeated-Benchmark Summary",
        "",
        f"- Random state: **{RANDOM_STATE}**",
        f"- Number of train/test repeats: **{n_repeats}**",
        f"- Rows (initial / after target cleaning): **{int(n_rows_initial)} / {int(n_after)}**",
        f"- Unique papers / groups: **{int(groups_all.nunique())}**",
        f"- Target column: `{target_col}`",
        f"- Bootstrap rows added per train split: **{n_boot}**",
        f"- Best model (lowest mean GroupKFold CV RMSE): **{best_model_name}**",
        "",
        "## Mean GroupKFold CV RMSE (lower is better)",
        "",
        "```",
        tune_summary.round(4).to_string(index=False),
        "```",
        "",
        "## Model comparison -- mean +/- std across repeats (4 conditions)",
        "",
        "```",
        comp_summary_round.to_string(index=False),
        "```",
        "",
        "## Generalization gap (Group RMSE - Random RMSE)",
        "",
        "```",
        gap_summary_round.to_string(index=False),
        "```",
        "",
    ]
    if n_boot > 0:
        summary_lines += [
            "## Bootstrap augmentation effect (negative dRMSE = BT helps)",
            "",
            "```",
            aug_summary_round.to_string(index=False),
            "```",
            "",
        ]
    summary_lines += [
        "## Output files (all under `results/`)",
        "",
        "- `model_comparison.csv` -- headline mean +/- std table (PRIMARY).",
        "- `model_comparison_repeats.csv` -- one row per (repeat, model, strategy).",
        "- `generalization_gap.csv` / `generalization_gap_repeats.csv`",
        "- `augmentation_comparison.csv` / `augmentation_comparison_repeats.csv`",
        "- `best_params_repeats.csv` -- tuned hyper-parameters per repeat.",
        "- `predictions/` -- per-repeat y_true / y_pred CSVs for every model x strategy.",
        "- `model_comparison_bars.png` -- mean +/- std RMSE per model (4 strategies).",
        "- `per_repeat_group_rmse_strip.png` -- run-to-run variability for Group-Aware RD.",
        "- `pred_vs_true_*.png`, `residuals_*.png`, `error_hist_*.png` (representative repeat).",
        "- `per_paper_*` -- per-paper extrapolation analysis (representative repeat).",
        "- `feature_importance_*` -- Decision Tree & XGBoost.",
        "- `permutation_importance.*` -- best repeated-benchmark model.",
        "- `shap_summary_*` / `shap_bar_*` -- best model (unless --no-shap).",
        "- `metrics.json` -- machine-readable bundle.",
        "- `best_*.joblib` -- best representative pipelines per model.",
    ]
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(summary_lines))
    print("Saved SUMMARY.md                 -> SUMMARY.md")

    # ----------------------------------------------------------- metrics dump (single, repeat-aware)
    metrics_blob = {
        "random_state": RANDOM_STATE,
        "n_repeats": n_repeats,
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
        "best_params_representative": best_params_store,
        "best_params_per_repeat": {
            m: {str(r): p for r, p in by_rep.items()}
            for m, by_rep in best_params_by_model_repeat.items()
        },
        "representative_repeat_by_model": representative_repeat_by_model,
        "repeat_summary": {
            "model_comparison": comp_summary_round.to_dict(orient="records"),
            "generalization_gap": gap_summary_round.to_dict(orient="records"),
            "augmentation_comparison": aug_summary_round.to_dict(orient="records"),
        },
        "comparison_repeats": comp_raw_round.to_dict(orient="records"),
        "generalization_gap_repeats": gap_raw_round.to_dict(orient="records"),
        "augmentation_comparison_repeats": aug_raw_round.to_dict(orient="records"),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_blob, f, indent=2, default=str)

    _print_header("DONE")
    print(f"All artefacts written to: {out_dir.resolve()}")
    print(f"Repeats averaged: {n_repeats}")
    print("Key files:")
    print("  - SUMMARY.md                     (human-readable headline report)")
    print("  - model_comparison.csv           (PRIMARY: mean +/- std across repeats)")
    print("  - model_comparison_repeats.csv   (raw rows across all repeats)")
    print("  - model_comparison_bars.png      (bar chart with error bars)")
    print("  - per_repeat_group_rmse_strip.png(per-repeat variability)")
    print("  - generalization_gap[_repeats].csv")
    print("  - r2_stability.csv               (diagnostic repeat-to-repeat variability)")
    print("  - augmentation_comparison[_repeats].csv")
    print("  - best_params_repeats.csv        (tuned params per repeat)")
    print("  - split_diagnostics.csv          (repeat split balance diagnostics)")
    print("  - predictions/                   (per-repeat y_true / y_pred CSVs)")
    print("  - metrics.json                   (machine-readable bundle, repeat-aware)")
    print("  - pred_vs_true_*.png / residuals_*.png / error_hist_*.png")
    print("  - per_paper_*                    (representative repeat)")
    print("  - feature_importance_*           (Decision Tree & XGBoost)")
    print("  - permutation_importance.*       (best repeated benchmark model)")
    print("  - shap_summary_* / shap_bar_*    (best repeated benchmark model)")
    print("  - bootstrap_fsr_distribution.png / augmentation_comparison.png")
    print("  - best_decision_tree.joblib / best_xgboost.joblib / best_knn.joblib / best_mlp.joblib")
    print(f"\nPRIMARY scientific result = the Group-Aware (RD) rows averaged over {n_repeats} repeats.")
    print(f"Best model: {best_model_name}.")


if __name__ == "__main__":
    main()
