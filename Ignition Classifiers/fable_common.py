"""Canonical data loading and feature engineering for ignition classification."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_VERSION = "fable-data-v3"
ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_DIM = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(µm|μm|um|mm|cm|m)?", re.I)

COLUMNS = {
    "article": "Article (MLA)", "authors": "Authors", "doi": "DOI",
    "geometry": "Geometry of Sample", "dimensions": "Dimensions of sample",
    "fuel_density": "fuel_density_kg_m3", "fuel_k": "fuel_k_W_mK",
    "fuel_cp": "fuel_cp_J_kgK", "fuel_pyrolysis": "fuel_pyrolysis_T_K",
    "fuel_alpha": "fuel_alpha_m2_s", "core_density": "core_density_kg_m3",
    "core_k": "core_k_W_mK", "core_cp": "core_cp_J_kgK",
    "oxygen": "Oxygen Concentration", "diluent": "diluent", "gas_m": "gas_M",
    "gas_cp": "gas_cp_mass", "gas_k": "gas_k", "gas_density": "gas_density_kg_m3",
    "gas_alpha": "gas_alpha_m2_s", "gas_nu": "gas_nu_m2_s", "pressure": "Pressure",
    "flow": "Flow Velocity", "internal_geometry": "Internal geometry",
    "internal_dimensions": "Internal Dimensions", "gravity": "Gravity",
    "facility": "Expireimental facility", "ignition_method": "Ignition method",
    "ignition_power": "Ignition power", "ignition_time": "Ignition time",
    "target": "Ignition (Yes/No)",
}
POST_OUTCOME_COLUMNS = ("Flame Length", "FSR (Flame Spread Rate)", "HRR (Heat release rate)",
                        "Smoke/ Areosols (yes/no)")

NUMERIC_FEATURES = {
    "fuel_density_kg_m3": "physics", "fuel_k_w_mk": "physics",
    "fuel_cp_j_kgk": "physics", "fuel_pyrolysis_t_k": "physics",
    "fuel_alpha_m2_s": "physics", "core_density_kg_m3": "physics",
    "core_k_w_mk": "physics", "core_cp_j_kgk": "physics",
    "oxygen_fraction": "physics", "gas_molar_mass": "physics",
    "gas_cp_j_kgk": "physics", "gas_k_w_mk": "physics",
    "gas_density_kg_m3": "physics", "gas_alpha_m2_s": "physics",
    "gas_nu_m2_s": "physics", "pressure_kpa": "physics",
    "flow_velocity_mm_s": "physics", "flow_speed_abs_mm_s": "physics",
    "gravity_g": "physics", "log10_gravity_g": "physics",
    "ignition_power_w": "apparatus", "ignition_time_s": "apparatus",
    "ignition_energy_j": "apparatus", "sample_dim_1_mm": "apparatus",
    "sample_dim_2_mm": "apparatus", "sample_dim_3_mm": "apparatus",
    "sample_dim_min_mm": "apparatus", "sample_dim_max_mm": "apparatus",
    "sample_dim_mean_mm": "apparatus", "sample_dim_count": "apparatus",
    "core_diameter_mm": "apparatus", "outer_diameter_mm": "apparatus",
    "insulation_thickness_mm": "apparatus", "internal_dim_1_mm": "apparatus",
    "internal_dim_2_mm": "apparatus", "internal_dim_3_mm": "apparatus",
    "internal_dim_mean_mm": "apparatus",
}
CATEGORICAL_FEATURES = {
    "geometry_cat": "physics", "diluent_cat": "physics",
    "flow_direction_cat": "physics", "gravity_regime_cat": "physics",
    "internal_geometry_cat": "apparatus", "facility_cat": "apparatus",
    "ignition_method_cat": "apparatus",
}


def _text(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan
    value = re.sub(r"\s+", " ", str(value).replace("\xa0", " ").strip())
    return np.nan if value.lower() in {"", "-", "na", "n/a", "nan", "none"} else value


def _number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    match = _NUM.search(str(value).replace(",", ".").replace("−", "-"))
    return float(match.group()) if match else np.nan


def _factor(unit: str) -> float:
    return {"µm": .001, "μm": .001, "um": .001, "cm": 10., "m": 1000.}.get(
        (unit or "mm").lower(), 1.)


def dimensions_mm(value: Any) -> list[float]:
    if pd.isna(value):
        return []
    text = str(value).replace(",", ".").replace("×", "x").replace("Ø", " diameter ")
    return [float(number) * _factor(unit) for number, unit in _DIM.findall(text)]


def pressure_kpa(value: Any) -> float:
    number, text = _number(value), str(value).lower()
    if pd.isna(number):
        return np.nan
    if "mpa" in text: return number * 1000
    if "atm" in text: return number * 101.325
    if "psi" in text: return number * 6.894757
    if "pa" in text and "kpa" not in text and "mpa" not in text: return number / 1000
    return number


def flow_mm_s(value: Any) -> float:
    number, text = _number(value), str(value).lower()
    if pd.isna(number):
        return np.nan
    if "cm/s" in text: return number * 10
    if "m/s" in text and "mm/s" not in text and "cm/s" not in text: return number * 1000
    return number


def gravity_g(value: Any) -> float:
    text, number = str(value).lower().replace("²", "2"), _number(value)
    if pd.isna(number):
        return 1e-6 if any(x in text for x in ("micro", "µg", "μg")) else np.nan
    if "cm/s2" in text or "cm/s^2" in text: return number / 981
    if "mm/s2" in text or "mm/s^2" in text: return number / 9810
    if "m/s2" in text or "m/s^2" in text: return number / 9.81
    return number


def oxygen_fraction(value: Any) -> float:
    number = _number(value)
    return number / 100 if pd.notna(number) and number > 1 else number


def watts(value: Any) -> float:
    text = str(value).lower()
    return np.nan if any(x in text for x in ("w/cm", "kw/m", "amp", "current")) else _number(value)


def _category(value: Any, mapping: dict[str, tuple[str, ...]]) -> str:
    text = str(_text(value)).lower()
    if text == "nan":
        return "Unknown"
    for label, words in mapping.items():
        if any(word in text for word in words):
            return label
    return "Other"


def canonical_doi(value: Any) -> str | float:
    """Return a normalized true DOI; non-DOI URLs fall back to citation identity."""
    value = _text(value)
    if pd.isna(value):
        return np.nan
    text = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", str(value).lower()).rstrip("/. ")
    match = re.search(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", text, re.I)
    return match.group().rstrip(".,;") if match else np.nan


def canonical_citation(value: Any) -> str | float:
    value = _text(value)
    if pd.isna(value):
        return np.nan
    text = str(value).lower().translate(str.maketrans({"“": '"', "”": '"', "’": "'", "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text)).strip()


def _resolve(raw: pd.DataFrame, require_target: bool) -> dict[str, str]:
    found: dict[str, str] = {}
    inference_optional = {"article", "authors", "doi", "target"}
    for key, wanted in COLUMNS.items():
        hits = [column for column in raw.columns if str(column).strip().lower().startswith(wanted.lower())]
        if hits:
            found[key] = hits[0]
        elif require_target or key not in inference_optional:
            raise ValueError(
                f"Required database column {key!r} ({wanted!r}) is missing. "
                f"Available columns: {list(raw.columns)!r}"
            )
    return found


def read_raw(path: str | Path) -> tuple[pd.DataFrame, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    errors = []
    for encoding in ENCODINGS:
        try:
            raw = pd.read_csv(path, skiprows=1, encoding=encoding)
            raw.columns = [str(c).strip() for c in raw.columns]
            return raw, encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError(f"Could not decode {path} with {ENCODINGS}: {'; '.join(errors)}")


def dataset_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_lists(feature_set: str) -> tuple[list[str], list[str]]:
    if feature_set not in {"all", "physics"}:
        raise ValueError("feature_set must be 'all' or 'physics'")
    numeric = [name for name, role in NUMERIC_FEATURES.items()
               if feature_set == "all" or role == "physics"]
    categorical = [name for name, role in CATEGORICAL_FEATURES.items()
                   if feature_set == "all" or role == "physics"]
    return numeric, categorical


def feature_manifest() -> list[dict[str, str]]:
    return ([{"feature": k, "kind": "numeric", "role": v} for k, v in NUMERIC_FEATURES.items()] +
            [{"feature": k, "kind": "categorical", "role": v} for k, v in CATEGORICAL_FEATURES.items()])


def load_data(path: str | Path, require_target: bool = True, deduplicate: bool = True
              ) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw, encoding = read_raw(path)
    original_rows = len(raw)
    blank_mask = raw.replace(r"^\s*$", np.nan, regex=True).isna().all(axis=1)
    blank_source_rows = (raw.index[blank_mask] + 3).astype(int).tolist()
    raw = raw.loc[~blank_mask].copy()
    columns = _resolve(raw, require_target=require_target)
    raw = raw.drop(columns=[c for c in raw if c.startswith("Unnamed")], errors="ignore")
    for column in raw.select_dtypes(include="object"):
        raw[column] = raw[column].map(_text)
    df = pd.DataFrame(index=raw.index)
    df["source_row_number"] = raw.index + 3
    df["row_id"] = [f"row-{i:07d}" for i in df["source_row_number"]]
    empty = pd.Series(np.nan, index=raw.index)
    article_source = raw[columns["article"]] if "article" in columns else empty
    authors_source = raw[columns["authors"]] if "authors" in columns else empty
    doi_source = raw[columns["doi"]] if "doi" in columns else empty
    df["raw_citation"] = article_source
    df["raw_authors"] = authors_source
    df["raw_doi"] = doi_source
    doi = doi_source.map(canonical_doi)
    citation = article_source.map(canonical_citation)
    missing_identity = doi.isna() & citation.isna()
    if require_target and missing_identity.any():
        rows = df.loc[missing_identity, "source_row_number"].tolist()[:20]
        raise ValueError(f"Rows lack both canonical DOI and citation identity: {rows}")
    df["paper_id"] = doi.map(lambda x: f"doi::{x}" if pd.notna(x) else np.nan)
    df["paper_id"] = df["paper_id"].fillna(citation.map(lambda x: f"citation::{x}"))
    df.loc[df["paper_id"].isna(), "paper_id"] = df.loc[
        df["paper_id"].isna(), "row_id"].map(lambda x: f"unknown::{x}")
    df["paper_label"] = df["raw_citation"].fillna(df["paper_id"]).astype(str).str.slice(0, 100)

    source_to_feature = {
        "fuel_density": "fuel_density_kg_m3", "fuel_k": "fuel_k_w_mk",
        "fuel_cp": "fuel_cp_j_kgk", "fuel_pyrolysis": "fuel_pyrolysis_t_k",
        "fuel_alpha": "fuel_alpha_m2_s", "core_density": "core_density_kg_m3",
        "core_k": "core_k_w_mk", "core_cp": "core_cp_j_kgk",
        "gas_m": "gas_molar_mass", "gas_cp": "gas_cp_j_kgk", "gas_k": "gas_k_w_mk",
        "gas_density": "gas_density_kg_m3", "gas_alpha": "gas_alpha_m2_s",
        "gas_nu": "gas_nu_m2_s",
    }
    for source, feature in source_to_feature.items():
        df[feature] = pd.to_numeric(raw[columns[source]], errors="coerce")
    df["oxygen_fraction"] = raw[columns["oxygen"]].map(oxygen_fraction)
    df["pressure_kpa"] = raw[columns["pressure"]].map(pressure_kpa)
    df["flow_velocity_mm_s"] = raw[columns["flow"]].map(flow_mm_s)
    df["flow_speed_abs_mm_s"] = df["flow_velocity_mm_s"].abs()
    df["gravity_g"] = raw[columns["gravity"]].map(gravity_g)
    df["log10_gravity_g"] = np.log10(df["gravity_g"].clip(lower=1e-8))
    df["ignition_power_w"] = raw[columns["ignition_power"]].map(watts)
    df["ignition_time_s"] = raw[columns["ignition_time"]].map(_number)
    df["ignition_energy_j"] = df["ignition_power_w"] * df["ignition_time_s"]

    sample_dims = raw[columns["dimensions"]].map(dimensions_mm)
    internal_dims = raw[columns["internal_dimensions"]].map(dimensions_mm)
    for prefix, values in (("sample", sample_dims), ("internal", internal_dims)):
        for i in range(3):
            df[f"{prefix}_dim_{i + 1}_mm"] = values.map(lambda x, i=i: x[i] if len(x) > i else np.nan)
    df["sample_dim_min_mm"] = sample_dims.map(lambda x: min(x) if x else np.nan)
    df["sample_dim_max_mm"] = sample_dims.map(lambda x: max(x) if x else np.nan)
    df["sample_dim_mean_mm"] = sample_dims.map(lambda x: float(np.mean(x)) if x else np.nan)
    df["sample_dim_count"] = sample_dims.map(len).astype(float)
    df["internal_dim_mean_mm"] = internal_dims.map(lambda x: float(np.mean(x)) if x else np.nan)
    dimension_text = raw[columns["dimensions"]].fillna("").astype(str).str.lower()
    core = dimension_text.str.extract(r"(?:core|inner)\D{0,20}(" + _NUM.pattern + r")")[0].astype(float)
    outer = dimension_text.str.extract(r"(?:outer|outside)\D{0,20}(" + _NUM.pattern + r")")[0].astype(float)
    df["core_diameter_mm"], df["outer_diameter_mm"] = core, outer
    df["insulation_thickness_mm"] = (outer - core).where(outer >= core) / 2

    df["geometry_cat"] = raw[columns["geometry"]].map(lambda x: _category(
        x, {"Wire": ("wire",), "Flat": ("flat",), "Cylindrical": ("cyl",), "Spherical": ("spher",)}))
    df["internal_geometry_cat"] = raw[columns["internal_geometry"]].map(lambda x: _category(
        x, {"Rectangular": ("rect",), "Cylindrical": ("cyl", "circular", "annular")}))
    df["ignition_method_cat"] = raw[columns["ignition_method"]].map(lambda x: _category(
        x, {"Open Flame": ("open flame", "pilot", "match"), "Radiative Heater": ("radiative", "heater"),
            "Discharge": ("discharge", "high-voltage"), "Wire / Coil": ("wire", "coil", "nicr", "electric")}))
    df["facility_cat"] = raw[columns["facility"]].fillna("Unknown").astype(str)
    df["diluent_cat"] = raw[columns["diluent"]].fillna("Unknown").astype(str)
    df["flow_direction_cat"] = np.select(
        [df["flow_velocity_mm_s"] > 0, df["flow_velocity_mm_s"] < 0,
         df["flow_velocity_mm_s"] == 0], ["Coflow", "Counterflow", "Quiescent"], "Unknown")
    df["gravity_regime_cat"] = np.select(
        [df["gravity_g"] < .001, df["gravity_g"] < .95, df["gravity_g"] <= 1.05],
        ["Microgravity", "Partial", "Earth"], "Hyper / Unknown")

    missing_target_rows: list[int] = []
    if "target" in columns:
        normalized = raw[columns["target"]].astype(str).str.strip().str.lower()
        df["ignition_binary"] = normalized.map(
            {"yes": 1, "y": 1, "1": 1, "true": 1, "no": 0, "n": 0, "0": 0, "false": 0})
        missing_target_rows = df.loc[df["ignition_binary"].isna(), "source_row_number"].tolist()
        if require_target and missing_target_rows:
            raise ValueError(
                f"{len(missing_target_rows)} rows have missing/unrecognized target labels; "
                f"source rows include {missing_target_rows[:20]}. No rows were silently dropped."
            )
        if require_target:
            df["ignition_binary"] = df["ignition_binary"].astype(int)
    elif require_target:
        raise ValueError("Target column is required for training/evaluation")

    features = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    duplicate_mask = df.duplicated(subset=features + (["ignition_binary"] if "ignition_binary" in df else []) +
                                   ["paper_id"], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if deduplicate:
        df = df.loc[~duplicate_mask].reset_index(drop=True)
    report = {
        "data_version": DATA_VERSION, "dataset_path": str(Path(path).resolve()),
        "dataset_sha256": dataset_hash(path), "encoding": encoding,
        "raw_row_count": original_rows, "row_count": len(df),
        "blank_row_count": len(blank_source_rows),
        "blank_source_rows": blank_source_rows,
        "label_counts": ({str(k): int(v) for k, v in df["ignition_binary"].value_counts().items()}
                         if "ignition_binary" in df else {}),
        "missing_target_count": len(missing_target_rows),
        "paper_count": int(df["paper_id"].nunique()), "duplicate_count": duplicate_count,
        "missingness_by_model_feature": {c: int(df[c].isna().sum()) for c in features},
        "feature_availability": {c: bool(df[c].notna().any()) for c in features},
        "post_outcome_columns_excluded": [c for c in POST_OUTCOME_COLUMNS
                                          if any(str(x).startswith(c) for x in raw.columns)],
        "feature_manifest": feature_manifest(),
    }
    return df, report


def write_validation_report(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def paper_weights(papers: pd.Series, strategy: str = "sqrt", beta: float = .999) -> np.ndarray:
    counts = papers.map(papers.value_counts()).to_numpy(float)
    if strategy == "none": weights = np.ones(len(papers))
    elif strategy == "inverse": weights = 1 / counts
    elif strategy == "sqrt": weights = 1 / np.sqrt(counts)
    elif strategy == "effective": weights = (1 - beta) / (1 - beta ** counts)
    elif strategy == "log": weights = 1 / (1 + np.log(counts))
    else: raise ValueError(f"Unknown paper weighting strategy: {strategy}")
    return weights / weights.mean()


def combined_weights(y: np.ndarray, papers: pd.Series, paper_strategy: str,
                     class_weight: bool) -> np.ndarray:
    weights = paper_weights(papers, paper_strategy)
    if class_weight:
        y = np.asarray(y)
        counts = np.bincount(y, minlength=2)
        class_weights = np.array([len(y) / (2 * max(counts[i], 1)) for i in range(2)])
        weights *= class_weights[y]
    return weights / weights.mean()
