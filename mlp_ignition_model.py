# MLP ignition classifier for Microgravity_Database.xlsx
# -----------------------------------------------------
# This script cleans the database, selects pre-ignition features, scales numerical
# features inside the ML pipeline, one-hot encodes categorical features, and trains
# an MLPClassifier to predict ignition_binary: Yes=1, No=0.

from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATA_PATH = Path("Microgravity_Database.xlsx")

# ---------- Cleaning utilities ----------
def clean_text(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.replace("\u00a0", " ").strip()
        s = re.sub(r"\s+", " ", s)
        return s if s not in {"", "-", "nan", "None"} else np.nan
    return x

_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def first_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).replace(",", ".").replace("−", "-")
    m = _num_re.search(s)
    return float(m.group(0)) if m else np.nan

def parse_oxygen_fraction(x):
    v = first_number(x)
    if pd.isna(v):
        return np.nan
    return v / 100.0 if v > 1.0 else v

def parse_pressure_kpa(x):
    v = first_number(x)
    if pd.isna(v):
        return np.nan
    s = str(x).lower() if not isinstance(x, (int, float, np.integer, np.floating)) else ""
    if "mpa" in s:
        return v * 1000.0
    if "atm" in s:
        return v * 101.325
    if "psia" in s or re.search(r"\bpsi\b", s):
        return v * 6.894757
    if "pa" in s and "kpa" not in s and "mpa" not in s:
        return v / 1000.0
    return v

def parse_flow_mm_s(x):
    v = first_number(x)
    if pd.isna(v):
        return np.nan
    s = str(x).lower() if not isinstance(x, (int, float, np.integer, np.floating)) else ""
    if "cm/s" in s or "cm s" in s:
        return v * 10.0
    if "m/s" in s and "mm/s" not in s and "cm/s" not in s:
        return v * 1000.0
    return v  # database convention: numeric values are already signed mm/s

def parse_gravity_g(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().replace("²", "2").replace("−", "-")
    v = first_number(x)
    if pd.isna(v):
        if "micro" in s or "µg" in s or "μg" in s:
            return 1e-6
        return np.nan
    if "cm/s2" in s or "cm/s^2" in s or "cm/s" in s:
        return v / 981.0
    if "mm/s2" in s or "mm/s^2" in s:
        return v / 9810.0
    if "m/s2" in s or "m/s^2" in s:
        return v / 9.81
    return v

def parse_watts(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    if "w/cm" in s or "kw/m" in s or re.search(r"\ba\b|amp|current", s):
        return np.nan
    return first_number(x)

def parse_seconds(x):
    return first_number(x)

def unit_factor_to_mm(unit):
    if unit is None or unit == "":
        return 1.0
    u = unit.lower().replace("μ", "µ")
    if u in ["µm", "um"]:
        return 0.001
    if u == "cm":
        return 10.0
    if u == "m":
        return 1000.0
    return 1.0

_dim_token = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(µm|μm|um|mm|cm|m)?", flags=re.I)

def extract_dimensions_mm(text):
    if pd.isna(text):
        return []
    s = str(text).replace(",", ".").replace("×", "x").replace("Ø", "diameter ")
    vals = []
    for num, unit in _dim_token.findall(s):
        vals.append(float(num) * unit_factor_to_mm(unit))
    return vals

def parse_core_outer_dims(text):
    if pd.isna(text):
        return (np.nan, np.nan, np.nan)
    s = str(text).replace(",", ".").replace("×", "x").lower()
    core = np.nan
    outer = np.nan
    m = re.search(r"([-+]?\d*\.?\d+)\s*(µm|μm|um|mm|cm|m)?\s*(?:diameter\s*)?(?:core|inner)", s)
    if not m:
        m = re.search(r"(?:core|inner)\D{0,20}([-+]?\d*\.?\d+)\s*(µm|μm|um|mm|cm|m)?", s)
    if m:
        core = float(m.group(1)) * unit_factor_to_mm(m.group(2))
    m = re.search(r"([-+]?\d*\.?\d+)\s*(µm|μm|um|mm|cm|m)?\s*(?:diameter\s*)?(?:outer|outside)", s)
    if not m:
        m = re.search(r"(?:outer|outside)\D{0,20}([-+]?\d*\.?\d+)\s*(µm|μm|um|mm|cm|m)?", s)
    if m:
        outer = float(m.group(1)) * unit_factor_to_mm(m.group(2))
    thickness = (outer - core) / 2.0 if pd.notna(core) and pd.notna(outer) and outer >= core else np.nan
    return core, outer, thickness

def normalize_geometry(x):
    s = clean_text(x)
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
    return str(s).title()

def normalize_internal_geometry(x):
    s = clean_text(x)
    if pd.isna(s):
        return "Unknown"
    l = str(s).lower()
    if "rect" in l:
        return "Rectangular"
    if "cyl" in l or "circular" in l or "annular" in l:
        return "Cylindrical/Circular"
    return str(s).title()

def normalize_facility(x):
    s = clean_text(x)
    if pd.isna(s):
        return "Unknown"
    l = str(s).lower()
    if "parabolic" in l:
        return "Parabolic Aircraft"
    if "drop" in l or "free-fall" in l or "free falling" in l:
        return "Drop Tower + Centrifuge" if "centrifuge" in l else "Drop Tower"
    if "centrifuge" in l or "hypergravity" in l:
        return "Centrifuge/Hypergravity"
    if "iss" in l or "spacecraft" in l:
        return "Spacecraft/ISS"
    if "sounding" in l or "rocket" in l:
        return "Sounding Rocket"
    if "ground" in l:
        return "Ground"
    if "numerical" in l:
        return "Numerical"
    return str(s).title()

def normalize_ignition_method(x):
    s = clean_text(x)
    if pd.isna(s):
        return "Unknown"
    l = str(s).lower()
    if "open flame" in l or "pilot" in l or "match" in l:
        return "Open Flame"
    if "radiative" in l or "heater" in l:
        return "Radiative Heater"
    if "high-voltage" in l or "discharge" in l:
        return "High-voltage Discharge"
    if "numerical" in l:
        return "Numerical Ignition Patch"
    if "wire" in l or "coil" in l or "nicr" in l or "electric" in l or "resistively" in l:
        return "Wire/Coil"
    return str(s).title()

def normalize_yes_no(x):
    s = clean_text(x)
    if pd.isna(s):
        return np.nan
    l = str(s).lower().strip()
    if l in ["yes", "y", "1", "true"]:
        return 1
    if l in ["no", "n", "0", "false"]:
        return 0
    return np.nan

def flow_direction(v):
    if pd.isna(v):
        return "Unknown"
    if v > 0:
        return "Coflow/Concurrent"
    if v < 0:
        return "Counterflow/Opposed"
    return "No Flow"

def gravity_regime(g):
    if pd.isna(g):
        return "Unknown"
    if g < 1e-3:
        return "Microgravity"
    if g < 0.95:
        return "Partial gravity"
    if g <= 1.05:
        return "Normal gravity"
    return "Hypergravity"

def material_family(x):
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
    if "cellulos" in s or "kimwipe" in s or "tissue" in s or "paper" in s:
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
    if s.strip() in ["", "nan"]:
        return "Unknown"
    return "Other material"

def core_material(x):
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
    return "Unknown/Not wire"

def group_rare(series, min_count=8):
    vc = series.value_counts(dropna=False)
    return series.where(series.map(vc) >= min_count, "Other/Rare")

# ---------- Load and clean ----------
def load_clean_database(path):
    df = pd.read_excel(path, sheet_name="Sheet2", header=[0, 1])
    # Flatten two-row headers and make all column names unique.
    cols = []
    for top, sub in df.columns:
        if pd.isna(sub) or str(sub).startswith("Unnamed"):
            name = str(top).strip()
        else:
            name = str(sub).strip()
        cols.append(name)
    seen = {}
    final_cols = []
    for name in cols:
        if name == "Info":
            seen[name] = seen.get(name, 0) + 1
            final_cols.append("Info" if seen[name] == 1 else f"Info_{seen[name]}")
        else:
            if name in seen:
                seen[name] += 1
                final_cols.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 1
                final_cols.append(name)
    df.columns = final_cols
    df.insert(0, "Original Excel Row", np.arange(3, 3 + len(df)))

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].map(clean_text)
    df = df.loc[~df.drop(columns=["Original Excel Row"]).duplicated(keep="first")].copy()

    df["ignition_binary"] = df["Ignition (Yes/No)"].map(normalize_yes_no)
    df["oxygen_fraction"] = df["Oxygen Concentration"].map(parse_oxygen_fraction)
    df["pressure_kpa"] = df["Pressure"].map(parse_pressure_kpa)
    df["flow_velocity_mm_s"] = df["Flow Velocity (Co flow is + and counter flow is -)"].map(parse_flow_mm_s)
    df["gravity_g"] = df["Gravity (g/gearth)"].map(parse_gravity_g)
    df["ignition_power_w"] = df["Ignition power (W)"].map(parse_watts)
    df["ignition_time_s"] = df["Ignition time (s)"].map(parse_seconds)

    sample_dims = df["Dimensions of sample (Wire is in diameter, cylinder is in radius x length, rectangle is in L x W x H)"].map(extract_dimensions_mm)
    for i in range(3):
        df[f"sample_dim_{i+1}_mm"] = sample_dims.map(lambda vals, i=i: vals[i] if len(vals) > i else np.nan)
    df["sample_dim_min_mm"] = sample_dims.map(lambda vals: np.nan if len(vals) == 0 else float(np.nanmin(vals)))
    df["sample_dim_max_mm"] = sample_dims.map(lambda vals: np.nan if len(vals) == 0 else float(np.nanmax(vals)))
    df["sample_dim_mean_mm"] = sample_dims.map(lambda vals: np.nan if len(vals) == 0 else float(np.nanmean(vals)))
    core_outer = df["Dimensions of sample (Wire is in diameter, cylinder is in radius x length, rectangle is in L x W x H)"].map(parse_core_outer_dims)
    df["core_diameter_mm"] = core_outer.map(lambda t: t[0])
    df["outer_diameter_mm"] = core_outer.map(lambda t: t[1])
    df["insulation_thickness_mm"] = core_outer.map(lambda t: t[2])

    internal_dims = df["Internal Dimensions"].map(extract_dimensions_mm)
    for i in range(3):
        df[f"internal_dim_{i+1}_mm"] = internal_dims.map(lambda vals, i=i: vals[i] if len(vals) > i else np.nan)
    df["internal_dim_mean_mm"] = internal_dims.map(lambda vals: np.nan if len(vals) == 0 else float(np.nanmean(vals)))

    df["geometry_category"] = df["Geometry of Sample (flat, wire, or Cylindrical)"].map(normalize_geometry)
    df["internal_geometry_category"] = df["Internal geometry (Cylindrical , rectangular)"].map(normalize_internal_geometry)
    df["experimental_facility_category"] = df["Expireimental facility (Parabolic Aircraft, Drop Tower, Spacecraft, Sounding Rocket, Ground)"].map(normalize_facility)
    df["ignition_method_category"] = df["Ignition method (Wire, open flame, or Radiative Heater"].map(normalize_ignition_method)
    df["material_family"] = df["Material of sample"].map(material_family)
    df["core_material"] = df["Material of sample"].map(core_material)
    df["flow_direction_category"] = df["flow_velocity_mm_s"].map(flow_direction)
    df["gravity_regime"] = df["gravity_g"].map(gravity_regime)
    df["sample_material_clean"] = df["Material of sample"].fillna("Unknown").astype(str).str.strip()
    df["sample_material_grouped"] = group_rare(df["sample_material_clean"], min_count=8)
    df["rig_name_grouped"] = group_rare(df["Rig Name"].fillna("Unknown").astype(str).str.strip(), min_count=8)
    df["source_group"] = df["DOI"].fillna(df["Article (MLA)"]).astype(str).str.strip()

    df = df[df["ignition_binary"].notna()].copy()
    df["ignition_binary"] = df["ignition_binary"].astype(int)
    return df

# ---------- Features ----------
numeric_features = [
    "oxygen_fraction", "pressure_kpa", "flow_velocity_mm_s", "gravity_g",
    "ignition_power_w", "ignition_time_s",
    "sample_dim_1_mm", "sample_dim_2_mm", "sample_dim_3_mm",
    "sample_dim_min_mm", "sample_dim_max_mm", "sample_dim_mean_mm",
    "core_diameter_mm", "outer_diameter_mm", "insulation_thickness_mm",
    "internal_dim_1_mm", "internal_dim_2_mm", "internal_dim_3_mm", "internal_dim_mean_mm",
]

categorical_features = [
    "geometry_category", "material_family", "core_material", "sample_material_grouped",
    "flow_direction_category", "gravity_regime", "internal_geometry_category",
    "experimental_facility_category", "ignition_method_category", "rig_name_grouped",
]

df = load_clean_database(DATA_PATH)
X = df[numeric_features + categorical_features]
y = df["ignition_binary"]
groups = df["source_group"]  # useful later for GroupKFold / leave-source-out validation

# ---------- Preprocessing and MLP ----------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_features),
    ],
    remainder="drop",
)

mlp = MLPClassifier(
    hidden_layer_sizes=(100,),  # paper used one hidden layer; use 64 neurons for a fast/stable starter. Try (100,) for exact paper-style size.
    activation="relu",
    alpha=0.01,
    learning_rate_init=1e-3,
    max_iter=100,
    early_stopping=False,
    random_state=42,
)

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", mlp),
])

# Stratified split keeps the ignition/no-ignition ratio similar in train/test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# MLPClassifier does not provide a stable class_weight argument across all scikit-learn versions.
# The class imbalance here is moderate, so we train directly and evaluate precision/recall/F1/ROC-AUC.
pipe.fit(X_train, y_train)

# Evaluation
pred = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1]

print("Confusion matrix:")
print(confusion_matrix(y_test, pred))
print("\nClassification report:")
print(classification_report(y_test, pred, target_names=["No ignition", "Ignition"]))
print("ROC-AUC:", roc_auc_score(y_test, proba))

# Cross-validation estimate. This still uses random folds; for true source transferability,
# replace StratifiedKFold with GroupKFold and pass groups=groups.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
print("\n3-fold CV metrics:")
for metric in scoring:
    vals = cv_results[f"test_{metric}"]
    print(f"{metric}: {vals.mean():.3f} ± {vals.std():.3f}")