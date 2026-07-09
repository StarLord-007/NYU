"""
fable_common.py
===============
Shared data loading, cleaning, feature engineering and grouping logic for the
Fable pipeline (fable_audit.py / fable_train.py / fable_eval.py).

This module fixes three data-integrity problems found during the audit of
``xgb_ignition_model_2.py`` + ``database_xgb.csv``:

1. **Corrupted DOI groups.** In ``database_xgb.csv`` the DOI cell of at least
   one 400+-row paper was Excel-autofilled (``...103989, ...103990, ...``),
   creating one fake "paper" per row. Group-by-DOI CV run on that file
   scattered rows of the same paper across folds, so the previously reported
   grouped ROC-AUC (0.678) was still partially contaminated.

2. **Citation-string aliases.** Six physical papers appear under two
   different citation strings (curly vs straight quotes, "et al." vs full
   author lists). Grouping by Article+Authors+DOI therefore splits one
   physical paper into two groups. We canonicalise the paper identifier:
   normalised DOI when present, otherwise a normalised article string.

3. **Encoding.** The latest CSV is cp1252-encoded; the old loader assumed
   UTF-8 and crashes on it.

Feature engineering mirrors the v2 script (so results remain comparable) but
every feature is registered with a *role* tag:

* ``physics``    — variables with absolute physical meaning that should
                   transfer across papers (O2 fraction, pressure, flow, ...).
* ``apparatus``  — descriptors of the experimental rig / campaign that are
                   highly paper-specific and are candidates for removal in
                   the domain-generalisation ablation (chamber dimensions,
                   facility, ignition hardware settings, ...).

Nothing in this module touches model training; it is pure data logic.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parsing utilities (carried over from xgb_ignition_model_2.py, unchanged
# physics, with small robustness fixes)
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _clean_text(x):
    if pd.isna(x):
        return np.nan
    if not isinstance(x, str):
        return x
    s = x.replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s if s.lower() not in {"", "-", "n/a", "na", "nan", "none"} else np.nan


def _first_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).replace(",", ".").replace("\u2212", "-")
    m = _NUM_RE.search(s)
    return float(m.group(0)) if m else np.nan


def parse_oxygen_fraction(x):
    v = _first_number(x)
    if pd.isna(v):
        return np.nan
    return v / 100.0 if v > 1.0 else v


def parse_pressure_kpa(x):
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
    return v


def parse_flow_mm_s(x):
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
    if pd.isna(text):
        return []
    s = str(text).replace(",", ".").replace("\u00d7", "x").replace("\u00d8", "diameter ")
    return [float(num) * _unit_to_mm(unit) for num, unit in _DIM_TOKEN.findall(s)]


def parse_core_outer(text):
    if pd.isna(text):
        return np.nan, np.nan, np.nan
    s = str(text).replace(",", ".").replace("\u00d7", "x").lower()
    core = outer = np.nan
    m = re.search(
        r"([-+]?\d*\.?\d+)\s*(\u00b5m|\u03bcm|um|mm|cm|m)?\s*(?:diameter\s*)?(?:core|inner)", s
    )
    if not m:
        m = re.search(
            r"(?:core|inner)\D{0,20}([-+]?\d*\.?\d+)\s*(\u00b5m|\u03bcm|um|mm|cm|m)?", s
        )
    if m:
        core = float(m.group(1)) * _unit_to_mm(m.group(2))
    m = re.search(
        r"([-+]?\d*\.?\d+)\s*(\u00b5m|\u03bcm|um|mm|cm|m)?\s*(?:diameter\s*)?(?:outer|outside)", s
    )
    if not m:
        m = re.search(
            r"(?:outer|outside)\D{0,20}([-+]?\d*\.?\d+)\s*(\u00b5m|\u03bcm|um|mm|cm|m)?", s
        )
    if m:
        outer = float(m.group(1)) * _unit_to_mm(m.group(2))
    thickness = (
        (outer - core) / 2.0 if pd.notna(core) and pd.notna(outer) and outer >= core else np.nan
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


# ---------------------------------------------------------------------------
# Canonical paper identity
# ---------------------------------------------------------------------------

def canonical_doi(x) -> str | float:
    """Normalise a DOI string so formatting variants collapse to one token."""
    s = _clean_text(x)
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    s = s.rstrip("/. ")
    return s if s else np.nan


def canonical_article(x) -> str | float:
    """Normalise a citation string: case, whitespace, quote style, punctuation."""
    s = _clean_text(x)
    if pd.isna(s):
        return np.nan
    s = str(s).lower()
    # unify curly quotes/dashes, drop all punctuation, collapse whitespace
    s = (
        s.replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u2013", "-").replace("\u2014", "-")
    )
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else np.nan


# ---------------------------------------------------------------------------
# Column registry for the latest CSV
# ---------------------------------------------------------------------------

COLS = {
    "article": "Article (MLA)",
    "authors": "Authors",
    "doi": "DOI",
    "geometry": "Geometry of Sample (Flat, wire, or Cylindrical)",
    "dimensions": (
        "Dimensions of sample (Wire is in diameter, Cylindrical is in length x radius x "
        "thickness, Rectangle is in length x width x height, Shperical)"
    ),
    "fuel_density": "fuel_density_kg_m3",
    "fuel_thermal_conductivity": "fuel_k_W_mK",
    "fuel_heat_capacity": "fuel_cp_J_kgK",
    "fuel_pyrolysis_temperature": "fuel_pyrolysis_T_K",
    "core_density": "core_density_kg_m3",
    "core_heat_capacity": "core_cp_J_kgK",
    "core_thermal_conductivity": "core_k_W_mK",
    "o2": "Oxygen Concentration",
    "diluent_gas": "diluent",
    "gas_molar_mass": "gas_M",
    "gas_specific_heat": "gas_cp_mass",
    "gas_thermal_conductivity": "gas_k",
    "gas_density": "gas_density_kg_m3",
    "gas_thermal_diffusivity": "gas_alpha_m2_s",
    "gas_kinematic_viscosity": "gas_nu_m2_s",
    "pressure": "Pressure",
    "flow": "Flow Velocity (Co flow is + and counter flow is -)",
    "internal_geom": "Internal geometry (Cylindrical , rectangular)",
    "internal_dims": "Internal Dimensions",
    "gravity": "Gravity (g/gearth)",
    "facility": (
        "Expireimental facility (Parabolic Aircraft, Drop Tower, Spacecraft, "
        "Sounding Rocket, Ground)"
    ),
    "ig_method": "Ignition method (Wire, open flame, or Radiative Heater",
    "ig_power": "Ignition power (W)",
    "ig_time": "Ignition time (s)",
    "ignition": "Ignition (Yes/No)",
    # ---- optional identity / apparatus columns (see OPTIONAL_COLS) ----
    # ``rig`` and ``material`` are present in some historical spreadsheets but
    # NOT in ``Microgravity_Database_reduced.csv`` (which encodes the material
    # implicitly through the fuel_* thermophysical columns). They are declared
    # here only so the resolver can pick them up when they *do* exist.
    "rig": "Rig Name",
    "material": "Material of sample",
}

# Logical keys that are allowed to be absent from a given CSV. When a key in
# this set cannot be resolved, ``_resolve_columns`` maps it to ``None`` instead
# of raising, and ``load_clean`` degrades gracefully (fills the derived column
# with NaN / a placeholder). This is what lets the *same* module load both the
# legacy workbook (with Rig Name / Material of sample) and the new reduced CSV
# (without them).
OPTIONAL_COLS = {"rig", "material"}

POST_IGNITION_LEAKS = [
    "Flame Length",
    "FSR (Flame Spread Rate)",
    "HRR (Heat release rate)",
    "Smoke/ Areosols (yes/no)",
]

# Default database shipped alongside this module (used when no path is given).
DEFAULT_DATA = "Microgravity_Database_reduced.csv"


def _read_csv_any_encoding(path: Path) -> pd.DataFrame:
    """Read the two-row-header microgravity CSV under any common encoding.

    The file uses a *section banner* first row (Citation / Sample / Flow / ...)
    above the real field names, so we skip the first row. Windows-exported CSVs
    carry ``\\r\\n`` line endings and cp1252 bytes; we try UTF-8 first, then the
    Windows/Latin fallbacks. Fully-empty trailing ``Unnamed`` columns produced
    by stray trailing commas are dropped here so downstream logic never sees
    them.
    """
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, skiprows=1, encoding=enc)
            # Strip trailing whitespace from every header (the reduced CSV has
            # e.g. "Internal Dimensions " with a trailing space).
            df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
            return df
        except UnicodeDecodeError as exc:  # try the next encoding
            last_err = exc
            continue
    raise UnicodeDecodeError(
        "all", b"", 0, 1, f"Could not decode {path} ({last_err})"
    )


def _resolve_columns(raw: pd.DataFrame) -> dict:
    """Map the logical column registry onto whatever header variant the file has.

    Optional columns (``OPTIONAL_COLS``) resolve to ``None`` when absent rather
    than raising, so one module can load both the legacy workbook and the new
    reduced CSV.
    """
    cols = {}
    available = {c.lower().strip(): c for c in raw.columns if isinstance(c, str)}
    fallbacks = {
        "article": ["article (mla)", "article"],
        "dimensions": None,  # matched by prefix below
    }
    for key, name in COLS.items():
        if name in raw.columns:
            cols[key] = name
            continue
        # exact match modulo case / surrounding whitespace
        if name.lower().strip() in available:
            cols[key] = available[name.lower().strip()]
            continue
        if key in fallbacks and fallbacks[key]:
            for cand in fallbacks[key]:
                if cand in available:
                    cols[key] = available[cand]
                    break
            if key in cols:
                continue
        # prefix matching for the long free-text headers
        prefix = name.split("(")[0].strip().lower()
        hits = [c for c in raw.columns if isinstance(c, str) and c.lower().startswith(prefix)]
        if len(hits) == 1:
            cols[key] = hits[0]
        elif key == "authors" and "authors" in available:
            cols[key] = available["authors"]
        elif key in OPTIONAL_COLS:
            cols[key] = None  # allowed to be missing
        else:
            raise KeyError(f"Cannot resolve column for '{key}' (wanted: {name!r})")
    return cols


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------
# role: "physics"    -> absolute physical meaning, expected to transfer
#       "apparatus"  -> rig / campaign descriptor, paper-fingerprint candidate
NUMERIC_FEATURES = {
    "oxygen_fraction": "physics",
    "pressure_kpa": "physics",
    "flow_velocity_mm_s": "physics",
    "flow_speed_abs_mm_s": "physics",
    "gravity_g": "physics",
    "log10_gravity_g": "physics",
    "ignition_power_w": "apparatus",
    "ignition_time_s": "apparatus",
    "ignition_energy_j": "apparatus",
    "sample_dim_1_mm": "physics",
    "sample_dim_2_mm": "physics",
    "sample_dim_3_mm": "physics",
    "sample_dim_min_mm": "physics",
    "sample_dim_max_mm": "physics",
    "sample_dim_mean_mm": "physics",
    "sample_dim_count": "apparatus",
    "core_diameter_mm": "physics",
    "outer_diameter_mm": "physics",
    "insulation_thickness_mm": "physics",
    "internal_dim_1_mm": "apparatus",
    "internal_dim_2_mm": "apparatus",
    "internal_dim_3_mm": "apparatus",
    "internal_dim_mean_mm": "apparatus",
    "fuel_density": "physics",
    "fuel_heat_capacity": "physics",
    "fuel_thermal_conductivity": "physics",
    "fuel_pyrolysis_temperature": "physics",
    "core_density": "physics",
    "core_heat_capacity": "physics",
    "core_thermal_conductivity": "physics",
    "gas_molar_mass": "physics",
    "gas_density": "physics",
    "gas_specific_heat": "physics",
    "gas_thermal_conductivity": "physics",
    "gas_thermal_diffusivity": "physics",
    "gas_kinematic_viscosity": "physics",
}

CATEGORICAL_FEATURES = {
    "geometry_cat": "physics",
    "diluent_gas_cat": "physics",
    "internal_geom_cat": "apparatus",
    "facility_cat": "apparatus",
    "ig_method_cat": "apparatus",
    "flow_direction": "physics",
    "gravity_regime": "physics",
}

ALL_NUMERIC = list(NUMERIC_FEATURES)
ALL_CATEGORICAL = list(CATEGORICAL_FEATURES)
PHYSICS_NUMERIC = [k for k, v in NUMERIC_FEATURES.items() if v == "physics"]
PHYSICS_CATEGORICAL = [k for k, v in CATEGORICAL_FEATURES.items() if v == "physics"]


def _group_rare(series, min_count=10, label="Other / Rare"):
    vc = series.value_counts(dropna=False)
    return series.where(series.map(vc) >= min_count, label)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def locate_data_file(user_path: str | Path | None = None) -> Path:
    """Resolve the database path robustly, independent of the cwd.

    Search order: (1) the path exactly as given, (2) the default filename next
    to this module, (3) the default filename in the module's parent directories.
    When ``user_path`` is ``None`` the module default (``DEFAULT_DATA``) is used.
    """
    here = Path(__file__).resolve().parent
    given = Path(user_path) if user_path is not None else Path(DEFAULT_DATA)
    if given.exists():
        return given
    name = given.name
    candidates = [here / name] + [parent / name for parent in here.parents]
    for cand in candidates:
        if cand.exists():
            return cand
    return given  # return original so the caller can emit a clear error


def load_clean(data_path: str | Path | None = None, dedupe: bool = True) -> pd.DataFrame:
    """Load the latest CSV and return a clean feature frame.

    Returns a DataFrame with engineered features, the binary target
    ``ignition_binary``, and identity columns:

    * ``paper_id``       — canonical physical-paper identifier (normalised DOI,
                           falling back to normalised citation string).
    * ``paper_label``    — short human-readable label for reporting.
    * ``raw_citation``   — original Article string (audit only, never a feature).
    * ``rig_name``       — original rig name (audit only, never a feature).
    """
    data_path = locate_data_file(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"database not found: {data_path}")
    raw = _read_csv_any_encoding(data_path)
    raw.columns = [c.strip() if isinstance(c, str) else c for c in raw.columns]
    cols = _resolve_columns(raw)

    drop = [c for c in raw.columns if str(c).startswith("Unnamed")] + [
        c for c in POST_IGNITION_LEAKS if c in raw.columns
    ]
    raw = raw.drop(columns=drop)
    for c in raw.columns:
        if raw[c].dtype == "object":
            raw[c] = raw[c].map(_clean_text)

    df = pd.DataFrame(index=raw.index)
    df["ignition_binary"] = raw[cols["ignition"]].map(normalise_yes_no)

    df["oxygen_fraction"] = raw[cols["o2"]].map(parse_oxygen_fraction)
    df["pressure_kpa"] = raw[cols["pressure"]].map(parse_pressure_kpa)
    df["flow_velocity_mm_s"] = raw[cols["flow"]].map(parse_flow_mm_s)
    df["flow_speed_abs_mm_s"] = df["flow_velocity_mm_s"].abs()
    df["gravity_g"] = raw[cols["gravity"]].map(parse_gravity_g)
    df["log10_gravity_g"] = np.log10(df["gravity_g"].clip(lower=1e-8))
    df["ignition_power_w"] = raw[cols["ig_power"]].map(parse_watts)
    df["ignition_time_s"] = raw[cols["ig_time"]].map(parse_seconds)
    df["ignition_energy_j"] = df["ignition_power_w"] * df["ignition_time_s"]
    df["gas_molar_mass"] = raw[cols["gas_molar_mass"]].map(_first_number)
    df["gas_density"] = raw[cols["gas_density"]].map(_first_number)
    df["gas_specific_heat"] = raw[cols["gas_specific_heat"]].map(_first_number)
    df["gas_thermal_conductivity"] = raw[cols["gas_thermal_conductivity"]].map(_first_number)
    df["gas_thermal_diffusivity"] = raw[cols["gas_thermal_diffusivity"]].map(_first_number)
    df["gas_kinematic_viscosity"] = raw[cols["gas_kinematic_viscosity"]].map(_first_number)
    df["diluent_gas_cat"] = _group_rare(raw[cols["diluent_gas"]].fillna("Unknown"))

    # ---- fuel & core thermophysical properties -------------------------------
    # The reduced database encodes the sample material implicitly, through these
    # per-row thermophysical columns, instead of a free-text "Material" column.
    # They are declared physics features in NUMERIC_FEATURES, so they MUST be
    # materialised here (the previous version registered but never assigned them,
    # which broke the drop_duplicates key). Missing / placeholder cells become
    # NaN and are median-imputed downstream by the modelling pipelines.
    df["fuel_density"] = raw[cols["fuel_density"]].map(_first_number)
    df["fuel_heat_capacity"] = raw[cols["fuel_heat_capacity"]].map(_first_number)
    df["fuel_thermal_conductivity"] = raw[cols["fuel_thermal_conductivity"]].map(_first_number)
    df["fuel_pyrolysis_temperature"] = raw[cols["fuel_pyrolysis_temperature"]].map(_first_number)
    df["core_density"] = raw[cols["core_density"]].map(_first_number)
    df["core_heat_capacity"] = raw[cols["core_heat_capacity"]].map(_first_number)
    df["core_thermal_conductivity"] = raw[cols["core_thermal_conductivity"]].map(_first_number)

    sample_dims = raw[cols["dimensions"]].map(extract_dimensions_mm)
    for i in range(3):
        df[f"sample_dim_{i + 1}_mm"] = sample_dims.map(
            lambda v, i=i: v[i] if len(v) > i else np.nan
        )
    df["sample_dim_min_mm"] = sample_dims.map(lambda v: np.nan if not v else float(np.nanmin(v)))
    df["sample_dim_max_mm"] = sample_dims.map(lambda v: np.nan if not v else float(np.nanmax(v)))
    df["sample_dim_mean_mm"] = sample_dims.map(lambda v: np.nan if not v else float(np.nanmean(v)))
    df["sample_dim_count"] = sample_dims.map(len).astype(float)

    core_outer = raw[cols["dimensions"]].map(parse_core_outer)
    df["core_diameter_mm"] = core_outer.map(lambda t: t[0])
    df["outer_diameter_mm"] = core_outer.map(lambda t: t[1])
    df["insulation_thickness_mm"] = core_outer.map(lambda t: t[2])

    internal_dims = raw[cols["internal_dims"]].map(extract_dimensions_mm)
    for i in range(3):
        df[f"internal_dim_{i + 1}_mm"] = internal_dims.map(
            lambda v, i=i: v[i] if len(v) > i else np.nan
        )
    df["internal_dim_mean_mm"] = internal_dims.map(
        lambda v: np.nan if not v else float(np.nanmean(v))
    )

    df["geometry_cat"] = raw[cols["geometry"]].map(normalise_geometry)
    df["internal_geom_cat"] = raw[cols["internal_geom"]].map(normalise_internal_geometry)
    df["facility_cat"] = raw[cols["facility"]].map(normalise_facility)
    df["ig_method_cat"] = raw[cols["ig_method"]].map(normalise_ignition_method)
    df["flow_direction"] = df["flow_velocity_mm_s"].map(flow_direction)
    df["gravity_regime"] = df["gravity_g"].map(gravity_regime)

    # ---- canonical paper identity ----
    doi_c = raw[cols["doi"]].map(canonical_doi)
    art_c = raw[cols["article"]].map(canonical_article)
    df["paper_id"] = doi_c.fillna("article::" + art_c.astype(str))
    df["raw_citation"] = raw[cols["article"]]
    # ``Rig Name`` is optional and absent from the reduced CSV; fall back to NaN
    # so the audit code that references ``rig_name`` keeps working.
    df["rig_name"] = raw[cols["rig"]] if cols.get("rig") else np.nan

    # Short label: first author surname + year-ish token + doi tail
    def _short_label(cit, pid):
        c = str(cit) if pd.notna(cit) else str(pid)
        first = re.split(r"[,.]", c)[0].strip()[:18]
        m = re.search(r"\((\d{4})\)|\b(19|20)\d{2}\b", c)
        year = m.group(0).strip("()") if m else "????"
        tail = str(pid)[-12:]
        return f"{first} {year} [{tail}]"

    df["paper_label"] = [
        _short_label(c, p) for c, p in zip(df["raw_citation"], df["paper_id"])
    ]

    # ---- final filtering ----
    df = df[df["ignition_binary"].notna()].copy()
    df["ignition_binary"] = df["ignition_binary"].astype(int)

    if dedupe:
        # Drop exact duplicates of (features + label) *within* a canonical
        # paper. Replicate experiments with identical settings and identical
        # outcome carry no extra ranking information but inflate that paper's
        # weight; conflicting-outcome replicates (stochastic ignition
        # boundary) are kept.
        key_cols = ALL_NUMERIC + ALL_CATEGORICAL + ["ignition_binary", "paper_id"]
        df = df.drop_duplicates(subset=key_cols).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def feature_lists(feature_set: str = "all") -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature lists for a named feature set.

    * ``all``      — every engineered feature (v2-script parity).
    * ``physics``  — physics-role features only (apparatus descriptors removed).
    """
    if feature_set == "all":
        return list(ALL_NUMERIC), list(ALL_CATEGORICAL)
    if feature_set == "physics":
        return list(PHYSICS_NUMERIC), list(PHYSICS_CATEGORICAL)
    raise ValueError(f"unknown feature_set {feature_set!r}")


# ---------------------------------------------------------------------------
# Paper-level weighting strategies
# ---------------------------------------------------------------------------

def paper_weights(paper_ids: pd.Series, strategy: str, beta: float = 0.999) -> np.ndarray:
    """Per-row weights that re-balance the influence of papers.

    Strategies (each normalised to mean weight = 1 so that the effective
    learning rate / regularisation trade-off of XGBoost is unchanged):

    * ``none``       — w = 1.
    * ``inverse``    — w = 1 / N_paper        (every paper counts equally).
    * ``sqrt``       — w = 1 / sqrt(N_paper)  (compromise; large campaigns
                       still count more, but sub-linearly).
    * ``effective``  — class-balanced "effective number of samples"
                       (Cui et al., CVPR 2019) applied at the paper level:
                       w = (1 - beta) / (1 - beta**N_paper). For beta -> 1
                       this approaches ``inverse``; for beta -> 0, ``none``.
    * ``log``        — w = 1 / (1 + ln N_paper). Statistically motivated by
                       treating within-paper rows as exchangeable draws from
                       a campaign-level cluster: the information content of a
                       cluster grows roughly logarithmically once rows are
                       strongly correlated.
    """
    n = paper_ids.map(paper_ids.value_counts()).to_numpy(dtype=float)
    if strategy == "none":
        w = np.ones_like(n)
    elif strategy == "inverse":
        w = 1.0 / n
    elif strategy == "sqrt":
        w = 1.0 / np.sqrt(n)
    elif strategy == "effective":
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
    elif strategy == "log":
        w = 1.0 / (1.0 + np.log(n))
    else:
        raise ValueError(f"unknown paper weighting strategy {strategy!r}")
    return w / w.mean()


def class_weights(y: pd.Series | np.ndarray) -> np.ndarray:
    """Balanced per-row class weights, normalised to mean 1."""
    y = np.asarray(y)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    n = len(y)
    w = np.where(y == 1, n / (2.0 * pos), n / (2.0 * neg)).astype(float)
    return w / w.mean()


def combined_weights(
    y: pd.Series | np.ndarray,
    paper_ids: pd.Series,
    paper_strategy: str = "sqrt",
    use_class: bool = True,
    beta: float = 0.999,
) -> np.ndarray:
    """weight = paper_weight x class_weight, normalised to mean 1."""
    w = paper_weights(paper_ids, paper_strategy, beta=beta)
    if use_class:
        w = w * class_weights(y)
    return w / w.mean()


# ---------------------------------------------------------------------------
# Self-test / smoke test
# ---------------------------------------------------------------------------
# Running ``python fable_common.py [path/to/database.csv]`` loads the database,
# builds the feature frame and prints a short integrity report. This is a quick
# way to confirm the module works against a new CSV before running the full
# audit / train / eval pipeline.

def _self_test(data_path: str | Path | None = None) -> None:
    path = locate_data_file(data_path)
    print(f"[fable_common] loading: {path}")
    df = load_clean(path)
    num, cat = feature_lists("all")
    pnum, pcat = feature_lists("physics")
    print(f"[fable_common] rows (after clean/dedupe) : {len(df)}")
    print(f"[fable_common] unique papers             : {df['paper_id'].nunique()}")
    print(f"[fable_common] ignition positive rate    : {df['ignition_binary'].mean():.3f}")
    print(f"[fable_common] numeric features   (all)  : {len(num)}")
    print(f"[fable_common] categorical features (all): {len(cat)}")
    print(f"[fable_common] physics-only features     : {len(pnum)} numeric + {len(pcat)} categorical")

    missing = [c for c in num + cat if c not in df.columns]
    if missing:
        raise RuntimeError(f"engineered features missing from frame: {missing}")
    print("[fable_common] OK: every registered feature is present in the frame.")

    # A couple of feature-coverage sanity numbers so a new CSV can be eyeballed.
    print("\n[fable_common] non-null coverage of a few key features:")
    for c in ["oxygen_fraction", "pressure_kpa", "gravity_g",
              "fuel_density", "gas_density", "geometry_cat", "facility_cat"]:
        if c in df.columns:
            print(f"    {c:<28s}: {df[c].notna().mean() * 100:5.1f}%")


if __name__ == "__main__":  # pragma: no cover
    import sys

    _self_test(sys.argv[1] if len(sys.argv) > 1 else None)
