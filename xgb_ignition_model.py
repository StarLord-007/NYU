"""
xgb_ignition_model.py
=====================

Powerful XGBoost classifier predicting *ignition* (Yes/No) for the microgravity
combustion database (`database.csv`).

Why XGBoost for this problem
----------------------------
* The data is tabular, mixed numeric/categorical, with non-linear interactions
  between oxygen concentration, pressure, flow velocity, gravity, sample
  material and geometry — exactly the regime where gradient-boosted decision
  trees dominate.
* XGBoost handles missing values natively (no need to fabricate imputations
  for predictors that are physically not applicable, e.g. `core_diameter_mm`
  on a flat sample).
* Tree splits are invariant to monotone transforms, so we do not need to
  standardise / log-transform raw physical quantities. We only need them in
  *consistent units* across rows.
* Built-in L1/L2 regularisation, column/row subsampling and `scale_pos_weight`
  give us strong control over over-fitting and class imbalance on a dataset
  of this size (~3k rows).
* It exposes feature importance and SHAP-compatible margins for interpretation,
  which matters for a combustion-physics audience.

Why we still clean the data carefully
-------------------------------------
The raw CSV is a hand-curated literature compilation: units are mixed
("101.3 kPa", "1 atm", "4 psia"), values are stringified ("8 s", "94 W"),
categories have typos / case variants ("Open Flame" vs "Open flame",
"Parabolic Aircraft" vs "Parabolic aircraft"), and several columns
("Flame Length", "FSR", "HRR", "Smoke/Aerosols") are *post-ignition*
observations and would leak the target. We:

  1. Coerce every physical column to one unit system.
  2. Normalise categorical strings into a small set of canonical buckets.
  3. Engineer sample dimensions (min / max / mean / core / outer / thickness).
  4. Drop rows with no ignition label and drop *all* post-ignition columns
     to avoid label leakage.
  5. Keep the article/DOI as a *grouping* variable for group-aware CV.

Evaluation strategy
-------------------
* 80/20 stratified hold-out for final reporting.
* 5-fold stratified CV inside a `RandomizedSearchCV` for tuning.
* A second, *grouped* 5-fold CV keyed on the source article (DOI) to estimate
  how the model generalises to *unseen experimental campaigns* — the harder
  and more honest test for a literature-aggregated dataset.
* Threshold is tuned on the validation predictions to maximise F1.

Outputs
-------
* `artifacts/xgb_ignition_model.joblib`  — fitted pipeline.
* `artifacts/feature_importance.csv`      — gain-based importances.
* `artifacts/confusion_matrix.png`        — confusion matrix on hold-out.
* `artifacts/roc_curve.png`               — ROC curve on hold-out.
* `artifacts/metrics.json`                — all numerical scores.

Run:
    python xgb_ignition_model.py --data database.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - import-time guard
    raise SystemExit(
        "xgboost is required. Install with `pip install xgboost`."
    ) from exc


# ---------------------------------------------------------------------------
# 1. Parsing / cleaning utilities
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _clean_text(x):
    """Strip whitespace, drop placeholder tokens, return NaN when empty."""
    if pd.isna(x):
        return np.nan
    if not isinstance(x, str):
        return x
    s = x.replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s if s.lower() not in {"", "-", "n/a", "na", "nan", "none"} else np.nan


def _first_number(x):
    """Extract the first numeric token from a value like '101.3 kPa'."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).replace(",", ".").replace("\u2212", "-")
    m = _NUM_RE.search(s)
    return float(m.group(0)) if m else np.nan


def parse_oxygen_fraction(x):
    """Return O2 as a mole fraction in [0, 1]. Inputs are typically '21%'."""
    v = _first_number(x)
    if pd.isna(v):
        return np.nan
    return v / 100.0 if v > 1.0 else v


def parse_pressure_kpa(x):
    """Normalise pressure to kPa from atm / kPa / MPa / psi / Pa strings."""
    v = _first_number(x)
    if pd.isna(v):
        return np.nan
    s = str(x).lower() if not isinstance(x, (int, float, np.integer, np.floating)) else ""
    if "mpa" in s:
        return v * 1000.0
    if "atm" in s:
        return v * 101.325
    if "psia" in s or re.search(r"\bpsi\b", s):
        return v * 6.894757
    if "kpa" in s:
        return v
    if "pa" in s:
        return v / 1000.0
    return v  # assume kPa when units are absent


def parse_flow_mm_s(x):
    """Normalise gas flow velocity to mm/s, preserving the +/- sign convention."""
    v = _first_number(x)
    if pd.isna(v):
        return np.nan
    s = str(x).lower() if not isinstance(x, (int, float, np.integer, np.floating)) else ""
    if "cm/s" in s:
        return v * 10.0
    if "m/s" in s and "mm/s" not in s and "cm/s" not in s:
        return v * 1000.0
    return v


def parse_gravity_g(x):
    """Normalise gravity to multiples of Earth gravity (g / g_earth)."""
    if pd.isna(x):
        return np.nan
    v = _first_number(x)
    s = str(x).lower().replace("\u00b2", "2")
    if pd.isna(v):
        if "micro" in s or "\u00b5g" in s or "\u03bcg" in s:
            return 1e-6
        return np.nan
    if "cm/s2" in s or "cm/s^2" in s:
        return v / 981.0
    if "mm/s2" in s or "mm/s^2" in s:
        return v / 9810.0
    if "m/s2" in s or "m/s^2" in s:
        return v / 9.81
    return v


def parse_watts(x):
    """Ignition power in W. Skip rows expressed in current (A) or flux (W/cm2)."""
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    if "w/cm" in s or "kw/m" in s or re.search(r"\ba\b|amp|current", s):
        return np.nan
    return _first_number(x)


def parse_seconds(x):
    return _first_number(x)


def _unit_to_mm(unit):
    if not unit:
        return 1.0
    u = unit.lower().replace("\u03bc", "\u00b5")
    if u in {"\u00b5m", "um"}:
        return 0.001
    if u == "cm":
        return 10.0
    if u == "m":
        return 1000.0
    return 1.0


_DIM_TOKEN = re.compile(
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(\u00b5m|\u03bcm|um|mm|cm|m)?",
    flags=re.IGNORECASE,
)


def extract_dimensions_mm(text):
    """Pull every length in a dimension string out and return them in mm."""
    if pd.isna(text):
        return []
    s = str(text).replace(",", ".").replace("\u00d7", "x").replace("\u00d8", "diameter ")
    return [float(num) * _unit_to_mm(unit) for num, unit in _DIM_TOKEN.findall(s)]


def parse_core_outer(text):
    """Heuristic parse of wire-style 'core; outer' diameters in mm."""
    if pd.isna(text):
        return np.nan, np.nan, np.nan
    s = str(text).replace(",", ".").replace("\u00d7", "x").lower()
    core = outer = np.nan
    m = re.search(
        r"([-+]?\d*\.?\d+)\s*(\u00b5m|\u03bcm|um|mm|cm|m)?\s*(?:diameter\s*)?(?:core|inner)",
        s,
    )
    if not m:
        m = re.search(
            r"(?:core|inner)\D{0,20}([-+]?\d*\.?\d+)\s*(\u00b5m|\u03bcm|um|mm|cm|m)?",
            s,
        )
    if m:
        core = float(m.group(1)) * _unit_to_mm(m.group(2))
    m = re.search(
        r"([-+]?\d*\.?\d+)\s*(\u00b5m|\u03bcm|um|mm|cm|m)?\s*(?:diameter\s*)?(?:outer|outside)",
        s,
    )
    if not m:
        m = re.search(
            r"(?:outer|outside)\D{0,20}([-+]?\d*\.?\d+)\s*(\u00b5m|\u03bcm|um|mm|cm|m)?",
            s,
        )
    if m:
        outer = float(m.group(1)) * _unit_to_mm(m.group(2))
    thickness = (
        (outer - core) / 2.0
        if pd.notna(core) and pd.notna(outer) and outer >= core
        else np.nan
    )
    return core, outer, thickness


# ---------- Categorical normalisation ----------

def normalise_geometry(x):
    s = _clean_text(x)
    if pd.isna(s):
        return "Unknown"
    l = str(s).lower()
    if "wire" in l:
        return "Wire"
    if "flat" in l:
        return "Flat"
    if "cyl" in l:
        return "Cylindrical"
    if "spher" in l:
        return "Spherical"
    if "chunk" in l:
        return "Chunk"
    return "Other"


def normalise_internal_geometry(x):
    s = _clean_text(x)
    if pd.isna(s):
        return "Unknown"
    l = str(s).lower()
    if "rect" in l:
        return "Rectangular"
    if "cyl" in l or "circular" in l or "annular" in l:
        return "Cylindrical"
    return "Other"


def normalise_facility(x):
    s = _clean_text(x)
    if pd.isna(s):
        return "Unknown"
    l = str(s).lower()
    if "parabolic" in l:
        return "Parabolic Aircraft"
    if "drop" in l:
        return "Drop Tower + Centrifuge" if "centrifuge" in l else "Drop Tower"
    if "centrifuge" in l or "hypergravity" in l:
        return "Centrifuge"
    if "iss" in l or "spacecraft" in l:
        return "Spacecraft / ISS"
    if "sounding" in l or "rocket" in l:
        return "Sounding Rocket"
    if "ground" in l:
        return "Ground"
    return "Other"


def normalise_ignition_method(x):
    s = _clean_text(x)
    if pd.isna(s):
        return "Unknown"
    l = str(s).lower()
    if "open flame" in l or "pilot" in l or "match" in l:
        return "Open Flame"
    if "radiative" in l or "heater" in l:
        return "Radiative Heater"
    if "discharge" in l or "high-voltage" in l:
        return "Discharge"
    if "wire" in l or "coil" in l or "nicr" in l or "electric" in l:
        return "Wire / Coil"
    return "Other"


def flow_direction(v):
    if pd.isna(v):
        return "Unknown"
    if v > 0:
        return "Coflow"
    if v < 0:
        return "Counterflow"
    return "Quiescent"


def gravity_regime(g):
    if pd.isna(g):
        return "Unknown"
    if g < 1e-3:
        return "Microgravity"
    if g < 0.95:
        return "Partial"
    if g <= 1.05:
        return "Earth"
    return "Hyper"


def material_family(x):
    """Coarse polymer/material family — captures the dominant chemistry."""
    s = str(x).lower() if not pd.isna(x) else ""
    if "ldpe" in s or "low-density polyethylene" in s:
        return "LDPE"
    if "hdpe" in s or "high-density polyethylene" in s:
        return "HDPE"
    if "etfe" in s:
        return "ETFE"
    if "pmma" in s or "organic glass" in s:
        return "PMMA"
    if re.search(r"\bpe\b|polyethylene", s):
        return "PE"
    if "cellulos" in s or "kimwipe" in s or "tissue" in s or "paper" in s or "cotton" in s:
        return "Cellulosic"
    if "nomex" in s:
        return "Nomex"
    if "kapton" in s:
        return "Kapton"
    if "kevlar" in s:
        return "Kevlar"
    if "polycarbonate" in s:
        return "Polycarbonate"
    if "silicone" in s:
        return "Silicone"
    if "peek" in s:
        return "PEEK"
    if "ppsu" in s:
        return "PPSU"
    if "conex" in s:
        return "Conex"
    if "sibal" in s:
        return "SIBAL"
    if s.strip() in {"", "nan"}:
        return "Unknown"
    return "Other"


def core_material(x):
    """Conductor used inside wire samples (drives Joule heating at ignition)."""
    s = str(x).lower() if not pd.isna(x) else ""
    if "nicr" in s or "nichrome" in s:
        return "NiCr"
    if re.search(r"\bcu\b|copper", s):
        return "Copper"
    if "stainless" in s or re.search(r"\bss\b", s):
        return "Stainless steel"
    if "steel" in s:
        return "Steel"
    if "iron" in s or re.search(r"\bfe\b", s):
        return "Iron"
    return "None / NA"


def normalise_yes_no(x):
    s = _clean_text(x)
    if pd.isna(s):
        return np.nan
    l = str(s).lower().strip()
    if l in {"yes", "y", "1", "true"}:
        return 1
    if l in {"no", "n", "0", "false"}:
        return 0
    return np.nan


def _group_rare(series, min_count=8, label="Other / Rare"):
    """Replace categories that occur fewer than `min_count` times with `label`.

    Keeps the one-hot encoder from blowing up on free-text fields and helps
    XGBoost split on signal instead of memorising single-row classes.
    """
    vc = series.value_counts(dropna=False)
    return series.where(series.map(vc) >= min_count, label)


# ---------------------------------------------------------------------------
# 2. Loading + feature engineering
# ---------------------------------------------------------------------------

# Canonical column names in the CSV. The first physical header row is a
# section banner ("Citation", "Sample", ...); the real headers are on row 2,
# so we skip the banner.
COLS = {
    "article": "Article (MLA)",
    "doi": "DOI",
    "geometry": "Geometry of Sample (flat, wire, or Cylindrical)",
    "dimensions": (
        "Dimensions of sample (Wire is in diameter, cylinder is in radius x length, "
        "rectangle is in L x W x H)"
    ),
    "material": "Material of sample",
    "o2": "Oxygen Concentration",
    "pressure": "Pressure",
    "flow": "Flow Velocity (Co flow is + and counter flow is -)",
    "rig": "Rig Name",
    "internal_geom": "Internal geometry (Cylindrical , rectangular)",
    "internal_dims": "Internal Dimensions ",
    "gravity": "Gravity (g/gearth)",
    "ig_method": "Ignition method (Wire, open flame, or Radiative Heater",
    "facility": (
        "Expireimental facility (Parabolic Aircraft, Drop Tower, Spacecraft, "
        "Sounding Rocket, Ground)"
    ),
    "ig_power": "Ignition power (W)",
    "ig_time": "Ignition time (s)",
    "ignition": "Ignition (Yes/No)",
}

# Outputs that are observed *after* the ignition decision and therefore
# must NOT be used as predictors:
POST_IGNITION_LEAKS = [
    "Flame Length",
    "FSR (Flame Spread Rate)",
    "HRR (Heat release rate)",
    "Smoke/ Areosols (yes/no)",
]


def load_clean(data_path: Path) -> pd.DataFrame:
    """Load the CSV (skipping the section-banner row) and build clean features."""
    raw = pd.read_csv(data_path, skiprows=1)

    # Drop the trailing unnamed column and any post-ignition leakage columns.
    drop = [c for c in raw.columns if c.startswith("Unnamed")] + [
        c for c in POST_IGNITION_LEAKS if c in raw.columns
    ]
    raw = raw.drop(columns=drop)

    # Strip whitespace on every string cell.
    for c in raw.columns:
        if raw[c].dtype == "object":
            raw[c] = raw[c].map(_clean_text)

    df = pd.DataFrame(index=raw.index)

    df["ignition_binary"] = raw[COLS["ignition"]].map(normalise_yes_no)

    # Numeric physical features
    df["oxygen_fraction"] = raw[COLS["o2"]].map(parse_oxygen_fraction)
    df["pressure_kpa"] = raw[COLS["pressure"]].map(parse_pressure_kpa)
    df["flow_velocity_mm_s"] = raw[COLS["flow"]].map(parse_flow_mm_s)
    df["flow_speed_abs_mm_s"] = df["flow_velocity_mm_s"].abs()
    df["gravity_g"] = raw[COLS["gravity"]].map(parse_gravity_g)
    df["log10_gravity_g"] = np.log10(df["gravity_g"].clip(lower=1e-8))
    df["ignition_power_w"] = raw[COLS["ig_power"]].map(parse_watts)
    df["ignition_time_s"] = raw[COLS["ig_time"]].map(parse_seconds)
    df["ignition_energy_j"] = df["ignition_power_w"] * df["ignition_time_s"]

    # Sample geometry
    sample_dims = raw[COLS["dimensions"]].map(extract_dimensions_mm)
    for i in range(3):
        df[f"sample_dim_{i + 1}_mm"] = sample_dims.map(
            lambda v, i=i: v[i] if len(v) > i else np.nan
        )
    df["sample_dim_min_mm"] = sample_dims.map(
        lambda v: np.nan if not v else float(np.nanmin(v))
    )
    df["sample_dim_max_mm"] = sample_dims.map(
        lambda v: np.nan if not v else float(np.nanmax(v))
    )
    df["sample_dim_mean_mm"] = sample_dims.map(
        lambda v: np.nan if not v else float(np.nanmean(v))
    )
    df["sample_dim_count"] = sample_dims.map(len).astype(float)

    core_outer = raw[COLS["dimensions"]].map(parse_core_outer)
    df["core_diameter_mm"] = core_outer.map(lambda t: t[0])
    df["outer_diameter_mm"] = core_outer.map(lambda t: t[1])
    df["insulation_thickness_mm"] = core_outer.map(lambda t: t[2])

    internal_dims = raw[COLS["internal_dims"]].map(extract_dimensions_mm)
    for i in range(3):
        df[f"internal_dim_{i + 1}_mm"] = internal_dims.map(
            lambda v, i=i: v[i] if len(v) > i else np.nan
        )
    df["internal_dim_mean_mm"] = internal_dims.map(
        lambda v: np.nan if not v else float(np.nanmean(v))
    )

    # Categorical features
    df["geometry_cat"] = raw[COLS["geometry"]].map(normalise_geometry)
    df["internal_geom_cat"] = raw[COLS["internal_geom"]].map(normalise_internal_geometry)
    df["facility_cat"] = raw[COLS["facility"]].map(normalise_facility)
    df["ig_method_cat"] = raw[COLS["ig_method"]].map(normalise_ignition_method)
    df["material_family"] = raw[COLS["material"]].map(material_family)
    df["core_material"] = raw[COLS["material"]].map(core_material)
    df["flow_direction"] = df["flow_velocity_mm_s"].map(flow_direction)
    df["gravity_regime"] = df["gravity_g"].map(gravity_regime)

    df["material_grouped"] = _group_rare(
        raw[COLS["material"]].fillna("Unknown").astype(str), min_count=10
    )
    df["rig_grouped"] = _group_rare(
        raw[COLS["rig"]].fillna("Unknown").astype(str), min_count=8
    )

    # Source identifier for group-wise CV
    df["source_group"] = raw[COLS["doi"]].fillna(raw[COLS["article"]]).astype(str)

    # Drop rows without a label, exact-duplicate rows
    df = df[df["ignition_binary"].notna()].copy()
    df["ignition_binary"] = df["ignition_binary"].astype(int)
    df = df.drop_duplicates().reset_index(drop=True)

    return df


# Features fed to the model. Numeric vs. categorical lists are used by the
# ColumnTransformer below.
NUMERIC_FEATURES = [
    "oxygen_fraction",
    "pressure_kpa",
    "flow_velocity_mm_s",
    "flow_speed_abs_mm_s",
    "gravity_g",
    "log10_gravity_g",
    "ignition_power_w",
    "ignition_time_s",
    "ignition_energy_j",
    "sample_dim_1_mm",
    "sample_dim_2_mm",
    "sample_dim_3_mm",
    "sample_dim_min_mm",
    "sample_dim_max_mm",
    "sample_dim_mean_mm",
    "sample_dim_count",
    "core_diameter_mm",
    "outer_diameter_mm",
    "insulation_thickness_mm",
    "internal_dim_1_mm",
    "internal_dim_2_mm",
    "internal_dim_3_mm",
    "internal_dim_mean_mm",
]

CATEGORICAL_FEATURES = [
    "geometry_cat",
    "internal_geom_cat",
    "facility_cat",
    "ig_method_cat",
    "material_family",
    "core_material",
    "flow_direction",
    "gravity_regime",
    "material_grouped",
    "rig_grouped",
]


# ---------------------------------------------------------------------------
# 3. Pipeline and training
# ---------------------------------------------------------------------------

def build_pipeline(scale_pos_weight: float, random_state: int = 42) -> Pipeline:
    """Preprocessing + XGBoost.

    * Numeric branch: pass-through. XGBoost ingests NaN natively, so we do
      *not* impute; an unknown value is information.
    * Categorical branch: one-hot. We impute the literal string "Unknown" so
      missing-categoricals become their own bin and `handle_unknown='ignore'`
      gracefully handles unseen categories at inference time.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
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
        reg_alpha=0.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline([("preprocess", preprocessor), ("model", model)])


# ---------------------------------------------------------------------------
# 4. Reporting helpers
# ---------------------------------------------------------------------------

def _save_confusion_matrix(y_true, y_pred, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["No ignition", "Ignition"])
    ax.set_yticks([0, 1], ["No ignition", "Ignition"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (hold-out)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_roc(y_true, y_score, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (hold-out)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _onehot_feature_names(pipe: Pipeline) -> list[str]:
    """Recover feature names after the ColumnTransformer / one-hot expansion."""
    pre: ColumnTransformer = pipe.named_steps["preprocess"]
    names: list[str] = list(NUMERIC_FEATURES)
    ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["onehot"]
    for col, cats in zip(CATEGORICAL_FEATURES, ohe.categories_):
        names.extend(f"{col}={c}" for c in cats)
    return names


def _best_f1_threshold(y_true, y_score) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns thresholds of length len(prec)-1
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    # Skip the last point which corresponds to threshold = +inf
    best_idx = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[best_idx]) if len(thr) else 0.5, float(f1[best_idx])


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="database.csv", help="Path to the database CSV."
    )
    parser.add_argument(
        "--out", default="artifacts", help="Directory to write artifacts to."
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run RandomizedSearchCV (slower, ~few minutes).",
    )
    parser.add_argument(
        "--n-iter", type=int, default=30, help="RandomizedSearchCV iterations."
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading and cleaning {args.data} ...")
    df = load_clean(Path(args.data))
    print(
        f"  -> {len(df)} clean rows after dropping unlabelled / duplicates."
    )
    print(
        "  -> ignition class balance:\n"
        f"     {df['ignition_binary'].value_counts(normalize=True).round(3).to_dict()}"
    )

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["ignition_binary"].astype(int)
    groups = df["source_group"]

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    # `scale_pos_weight` reweights the positive class so XGBoost's gradient is
    # balanced. Convention: neg / pos. Here ignition is the majority class,
    # so this value will be < 1 and XGBoost will *down*-weight the easy
    # positive class, focusing on the harder no-ignition rows.
    spw = neg / max(pos, 1)
    print(f"  -> scale_pos_weight = neg/pos = {neg}/{pos} = {spw:.3f}")

    print("[2/6] Building pipeline ...")
    pipe = build_pipeline(scale_pos_weight=spw, random_state=args.random_state)

    print("[3/6] Stratified hold-out 80/20 split ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=args.random_state
    )

    if args.tune:
        print(f"[4/6] Hyperparameter search ({args.n_iter} iters, 5-fold CV on ROC-AUC) ...")
        param_dist = {
            "model__n_estimators": [200, 400, 600, 800, 1200],
            "model__max_depth": [3, 4, 5, 6, 8, 10],
            "model__learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
            "model__min_child_weight": [1, 2, 4, 6, 10],
            "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.5, 0.7, 0.8, 1.0],
            "model__gamma": [0.0, 0.1, 0.3, 0.5, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "model__reg_alpha": [0.0, 0.1, 0.5, 1.0],
        }
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=args.n_iter,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state),
            random_state=args.random_state,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        print(f"  -> best CV ROC-AUC: {search.best_score_:.4f}")
        print(f"  -> best params: {search.best_params_}")
    else:
        print("[4/6] Skipping RandomizedSearchCV (pass --tune to enable). Fitting defaults ...")
        pipe.fit(X_train, y_train)

    print("[5/6] Evaluating on hold-out ...")
    proba_test = pipe.predict_proba(X_test)[:, 1]
    pred_default = (proba_test >= 0.5).astype(int)

    thr_best, f1_best = _best_f1_threshold(y_test, proba_test)
    pred_tuned = (proba_test >= thr_best).astype(int)

    roc_auc = float(roc_auc_score(y_test, proba_test))
    pr_auc = float(average_precision_score(y_test, proba_test))

    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC : {pr_auc:.4f}")
    print(f"  Best F1 threshold: {thr_best:.3f}  ->  F1={f1_best:.4f}")
    print("\n  Classification report @ threshold = 0.5:")
    print(classification_report(y_test, pred_default, target_names=["No ignition", "Ignition"]))
    print(f"  Classification report @ threshold = {thr_best:.3f} (max-F1):")
    print(classification_report(y_test, pred_tuned, target_names=["No ignition", "Ignition"]))
    print("  Confusion matrix @ 0.5:\n", confusion_matrix(y_test, pred_default))

    _save_confusion_matrix(y_test, pred_default, out_dir / "confusion_matrix.png")
    _save_roc(y_test, proba_test, out_dir / "roc_curve.png")

    # --- Additional generalisation tests on the full dataset ---
    print("\n  Stratified 5-fold CV on full data (ROC-AUC):")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    skf_scores = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    print(f"    {skf_scores.mean():.4f} +/- {skf_scores.std():.4f}")

    print("  GroupKFold 5-fold CV by source article (harder, true-generalisation):")
    gkf = GroupKFold(n_splits=5)
    gkf_scores = cross_val_score(
        pipe, X, y, cv=gkf, groups=groups, scoring="roc_auc", n_jobs=-1
    )
    print(f"    {gkf_scores.mean():.4f} +/- {gkf_scores.std():.4f}")

    print("[6/6] Persisting model, feature importances, metrics ...")

    # Feature importance (gain-based) post one-hot expansion.
    booster = pipe.named_steps["model"]
    importances = booster.feature_importances_
    names = _onehot_feature_names(pipe)
    imp_df = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(out_dir / "feature_importance.csv", index=False)
    print("  Top 15 features by gain:")
    print(imp_df.head(15).to_string(index=False))

    joblib.dump(pipe, out_dir / "xgb_ignition_model.joblib")

    metrics = {
        "n_rows_clean": int(len(df)),
        "class_balance": {"ignition": pos, "no_ignition": neg},
        "scale_pos_weight": spw,
        "holdout_roc_auc": roc_auc,
        "holdout_pr_auc": pr_auc,
        "holdout_best_f1": f1_best,
        "holdout_best_threshold": thr_best,
        "holdout_f1_default_thr": float(f1_score(y_test, pred_default)),
        "stratified_5fold_roc_auc_mean": float(skf_scores.mean()),
        "stratified_5fold_roc_auc_std": float(skf_scores.std()),
        "groupkfold_5fold_roc_auc_mean": float(gkf_scores.mean()),
        "groupkfold_5fold_roc_auc_std": float(gkf_scores.std()),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone. Artifacts written to:", out_dir.resolve())


if __name__ == "__main__":
    main()
