#!/usr/bin/env python3
"""Generate a physics-enriched microgravity ignition database.

The output intentionally keeps the source file's first two header/classification
rows. Data corrections follow the Fable audit:

* canonical paper identifiers are added so DOI/citation aliases collapse;
* the known DOI-autofill pattern from the old file is repaired if encountered;
* exact duplicate labelled measurements within a canonical paper are removed.

Material properties are representative literature values for material families,
not grade-specific measurements. Ambiguous materials are left blank rather than
filled with false precision.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Microgravity_Database_Latest.csv"
DEFAULT_OUTPUT = ROOT / "physics_db.csv"

AIR_DYNAMIC_VISCOSITY_PA_S = 1.85e-5
AIR_THERMAL_DIFFUSIVITY_M2_S_AT_1ATM = 2.2e-5
REFERENCE_AIR_TEMP_K = 293.15
G0_M_S2 = 9.80665
AIR_O2_FRACTION = 0.2095
AIR_GAS_CONSTANT_J_KG_K = 287.05


@dataclass(frozen=True)
class MaterialProps:
    family: str
    density_kg_m3: float
    thermal_conductivity_w_m_k: float
    heat_capacity_j_kg_k: float
    pyrolysis_temp_c: float
    limiting_oxygen_index_fraction: float | None
    source: str
    notes: str

    @property
    def thermal_diffusivity_m2_s(self) -> float:
        return self.thermal_conductivity_w_m_k / (
            self.density_kg_m3 * self.heat_capacity_j_kg_k
        )

    @property
    def thermal_inertia(self) -> float:
        return math.sqrt(
            self.thermal_conductivity_w_m_k
            * self.density_kg_m3
            * self.heat_capacity_j_kg_k
        )


MATERIALS: dict[str, MaterialProps] = {
    "LDPE": MaterialProps(
        "LDPE",
        921.0,
        0.34,
        2300.0,
        450.0,
        0.17,
        "Qenos PE thermal/electrical guide; polyolefin thermal-lens study",
        "Representative LDPE at room temperature; pyrolysis onset/range varies by heating rate.",
    ),
    "HDPE": MaterialProps(
        "HDPE",
        960.0,
        0.44,
        1900.0,
        450.0,
        0.17,
        "Qenos PE thermal/electrical guide; polyolefin thermal-lens study",
        "Representative HDPE at room temperature.",
    ),
    "PE": MaterialProps(
        "PE",
        940.0,
        0.40,
        2100.0,
        450.0,
        0.17,
        "Qenos PE thermal/electrical guide; polyolefin thermal-lens study",
        "Generic polyethylene fallback when LDPE/HDPE grade is not specified.",
    ),
    "ETFE": MaterialProps(
        "ETFE",
        1700.0,
        0.23,
        900.0,
        392.5,
        0.30,
        "NETZSCH ETFE data sheet",
        "ETFE decomposition reported around 385-400 C; LOI is representative for fluoropolymer insulation.",
    ),
    "PMMA": MaterialProps(
        "PMMA",
        1190.0,
        0.193,
        1466.0,
        350.0,
        0.17,
        "MIT PMMA material database; Bangs Labs PMMA literature summary; PMMA ignition/pyrolysis literature",
        "PMMA pyrolysis is commonly treated over about 250-450 C; value is midpoint-like representative.",
    ),
    "Cellulosic": MaterialProps(
        "Cellulosic",
        1530.0,
        0.24,
        1209.0,
        325.0,
        0.19,
        "Biomass combustion model; birch cellulose thermal-characteristics study; cellulose pyrolysis literature",
        "Uses dense/cell-wall cellulose properties; paper/wipe apparent density and conductivity can be much lower.",
    ),
    "SIBAL": MaterialProps(
        "SIBAL",
        573.0,
        0.10,
        1325.0,
        325.0,
        0.19,
        "NASA/Saffire and BASS literature on SIBAL fabric",
        "SIBAL is 75 wt% cotton and 25 wt% fiberglass; density estimated from 18.2 mg/cm2 area density and 0.3175 mm thickness.",
    ),
    "Cellulose acetate": MaterialProps(
        "Cellulose acetate",
        1300.0,
        0.20,
        1500.0,
        300.0,
        0.20,
        "Representative engineering-polymer tables; cellulose-acetate thermal-decomposition literature",
        "Representative cellulose-acetate plastic values.",
    ),
    "Nomex": MaterialProps(
        "Nomex",
        1380.0,
        0.13,
        1200.0,
        427.0,
        0.29,
        "DuPont Nomex technical guide; AZoM Nomex property table",
        "Nomex fiber/paper density and conductivity vary with construction; degradation accelerates above about 427 C.",
    ),
    "Kevlar": MaterialProps(
        "Kevlar",
        1440.0,
        0.04,
        1420.0,
        500.0,
        0.29,
        "Aramid-fiber property literature; manufacturer datasheets",
        "Para-aramid representative values; exact fabric construction can dominate effective conductivity.",
    ),
    "Aramid": MaterialProps(
        "Aramid",
        1400.0,
        0.10,
        1300.0,
        450.0,
        0.29,
        "Nomex/Kevlar aramid property literature",
        "Generic aramid fallback for Conex/Arimid/Nomeks aliases.",
    ),
    "Kapton": MaterialProps(
        "Kapton",
        1420.0,
        0.20,
        1090.0,
        520.0,
        0.37,
        "DuPont Kapton HN data sheet",
        "Kapton thermal conductivity is anisotropic; cross-plane film value used.",
    ),
    "Polycarbonate": MaterialProps(
        "Polycarbonate",
        1220.0,
        0.20,
        1200.0,
        500.0,
        0.25,
        "NETZSCH PC data sheet; FSRI polycarbonate material database; PC pyrolysis kinetics literature",
        "Representative PC; decomposition commonly reported across about 480-570 C.",
    ),
    "Silicone": MaterialProps(
        "Silicone",
        970.0,
        0.15,
        1460.0,
        330.0,
        0.26,
        "MIT PDMS material database; PDMS thermal-aging/decomposition literature",
        "PDMS/silicone representative values; degradation begins above about 300 C.",
    ),
    "PPSU": MaterialProps(
        "PPSU",
        1290.0,
        0.30,
        1130.0,
        450.0,
        0.35,
        "Ensinger/Eagle PPSU datasheets; GEHR PPSU property table",
        "Representative PPSU/polyphenylsulfone; decomposition often reported above 400-450 C.",
    ),
    "PEEK": MaterialProps(
        "PEEK",
        1320.0,
        0.25,
        1410.0,
        600.0,
        0.35,
        "NETZSCH PEEK data sheet; MatWeb PEEK summary; MakeItFrom PEEK",
        "Unfilled PEEK representative values.",
    ),
    "PET": MaterialProps(
        "PET",
        1390.0,
        0.155,
        1170.0,
        400.0,
        0.21,
        "Mylar/PET film datasheets; ESA PETP materials listing",
        "PET/Mylar film representative values; decomposition depends on atmosphere and heating rate.",
    ),
    "Polyurethane foam": MaterialProps(
        "Polyurethane foam",
        50.0,
        0.050,
        2440.0,
        250.0,
        0.18,
        "Rigid/flexible polyurethane foam thermophysical studies; PU thermal-decomposition literature",
        "Foam apparent properties; density and conductivity depend strongly on cell structure.",
    ),
    "Polyurethane": MaterialProps(
        "Polyurethane",
        1200.0,
        0.22,
        1500.0,
        250.0,
        0.18,
        "Polyurethane foam/melt thermophysical studies; PU thermal-decomposition literature",
        "Solid/polymer fallback for polyurethane when foam morphology is not specified.",
    ),
    "Phenolic foam": MaterialProps(
        "Phenolic foam",
        50.0,
        0.035,
        1400.0,
        400.0,
        0.30,
        "Phenolic-foam engineering data and fire-property literature",
        "Representative phenolic open-cell foam; exact formulation unknown.",
    ),
    "ABS": MaterialProps(
        "ABS",
        1040.0,
        0.17,
        1300.0,
        380.0,
        0.18,
        "ABS engineering-polymer property tables; ABS thermal-degradation literature",
        "Representative ABS plastic.",
    ),
    "Polystyrene": MaterialProps(
        "Polystyrene",
        1050.0,
        0.13,
        1300.0,
        400.0,
        0.18,
        "Bulk polystyrene/PMMA property summaries; polymer thermal-degradation literature",
        "Representative high-impact/general polystyrene fallback.",
    ),
    "Nylon": MaterialProps(
        "Nylon",
        1140.0,
        0.25,
        1700.0,
        430.0,
        0.22,
        "Nylon engineering-polymer property tables; polyamide degradation literature",
        "Capron/polyamide representative values.",
    ),
    "Neoprene": MaterialProps(
        "Neoprene",
        1230.0,
        0.19,
        2000.0,
        350.0,
        0.25,
        "Representative chloroprene-rubber engineering data",
        "Representative neoprene rubber values.",
    ),
    "Paraffin": MaterialProps(
        "Paraffin",
        900.0,
        0.25,
        2100.0,
        300.0,
        0.20,
        "Paraffin thermophysical-property literature",
        "Representative paraffin wax values.",
    ),
    "PEI": MaterialProps(
        "PEI",
        1270.0,
        0.22,
        1100.0,
        520.0,
        0.47,
        "Ultem/PEI engineering datasheets",
        "Representative polyetherimide / Ultem 1000 values.",
    ),
    "PTFE/FEP": MaterialProps(
        "PTFE/FEP",
        2150.0,
        0.25,
        1000.0,
        500.0,
        0.95,
        "Fluoropolymer engineering-property tables",
        "Generic fluoroplastic-film fallback; high LOI indicates near-nonflammability in normal air.",
    ),
    "PP": MaterialProps(
        "PP",
        900.0,
        0.18,
        1800.0,
        447.0,
        0.18,
        "NOAA CAMEO polypropylene data; NIST/polymer thermophysical-property literature",
        "Representative polypropylene; conductivity and heat capacity vary with crystallinity and temperature.",
    ),
    "POM": MaterialProps(
        "POM",
        1410.0,
        0.31,
        1400.0,
        350.0,
        0.15,
        "POM/Delrin engineering-polymer property tables; NIST polymer fire-property literature",
        "Representative polyoxymethylene / Delrin values.",
    ),
    "Epoxy": MaterialProps(
        "Epoxy",
        1200.0,
        0.20,
        1100.0,
        350.0,
        0.20,
        "Epoxy-resin engineering-property tables and thermal-degradation literature",
        "Representative unfilled epoxy resin; filled laminates can differ substantially.",
    ),
    "Polyisobutylene": MaterialProps(
        "Polyisobutylene",
        920.0,
        0.13,
        1900.0,
        380.0,
        0.18,
        "Polyisobutylene engineering-property and thermal-degradation literature",
        "Representative PIB rubber/polymer values.",
    ),
    "Fiberglass laminate": MaterialProps(
        "Fiberglass laminate",
        1900.0,
        0.30,
        800.0,
        450.0,
        None,
        "Glass-fiber/epoxy-laminate engineering data",
        "Composite fallback; combustible binder fraction is not specified.",
    ),
}


def clean_text(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return "" if text.lower() in {"", "-", "n/a", "na", "nan", "none"} else text


def first_number(value: str | None) -> float | None:
    text = clean_text(value).replace(",", ".").replace("\u2212", "-")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else None


def fmt(value: float | int | str | None, digits: int = 8) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return ""
    text = f"{value:.{digits}g}"
    return "0" if text == "-0" else text


def parse_oxygen_fraction(value: str | None) -> float | None:
    number = first_number(value)
    if number is None:
        return None
    return number / 100.0 if number > 1.0 else number


def parse_pressure_kpa(value: str | None) -> float | None:
    number = first_number(value)
    if number is None:
        return None
    text = clean_text(value).lower()
    if "mpa" in text:
        return number * 1000.0
    if "atm" in text:
        return number * 101.325
    if "psia" in text or re.search(r"\bpsi\b", text):
        return number * 6.894757
    if "kpa" in text:
        return number
    if re.search(r"\bpa\b", text):
        return number / 1000.0
    return number


def parse_flow_mm_s(value: str | None) -> float | None:
    number = first_number(value)
    if number is None:
        return None
    text = clean_text(value).lower()
    if "cm/s" in text:
        return number * 10.0
    if "m/s" in text and "mm/s" not in text and "cm/s" not in text:
        return number * 1000.0
    return number


def parse_gravity_g(value: str | None) -> float | None:
    text = clean_text(value).lower().replace("\u00b2", "2")
    number = first_number(text)
    if number is None:
        if "micro" in text or "\u00b5g" in text or "\u03bcg" in text:
            return 1e-6
        return None
    if "cm/s2" in text or "cm/s^2" in text:
        return number / 981.0
    if "mm/s2" in text or "mm/s^2" in text:
        return number / 9810.0
    if "m/s2" in text or "m/s^2" in text:
        return number / G0_M_S2
    return number


def parse_watts(value: str | None) -> float | None:
    text = clean_text(value).lower()
    if "w/cm" in text or "kw/m" in text or re.search(r"\ba\b|amp|current", text):
        return None
    return first_number(value)


def unit_to_mm(unit: str | None) -> float:
    normalized = (unit or "").lower().replace("\u03bc", "\u00b5")
    if normalized in {"\u00b5m", "um"}:
        return 0.001
    if normalized == "cm":
        return 10.0
    if normalized == "m":
        return 1000.0
    return 1.0


DIM_TOKEN = re.compile(
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(\u00b5m|\u03bcm|um|mm|cm|m)?",
    flags=re.IGNORECASE,
)


def extract_dimensions_mm(value: str | None) -> list[float]:
    text = clean_text(value).replace(",", ".").replace("\u00d7", "x").replace("\u00d8", "diameter ")
    return [float(num) * unit_to_mm(unit) for num, unit in DIM_TOKEN.findall(text)]


def parse_core_outer_mm(value: str | None) -> tuple[float | None, float | None, float | None]:
    text = clean_text(value).replace(",", ".").replace("\u00d7", "x").lower()
    core = None
    outer = None
    core_patterns = [
        r"([-+]?\d*\.?\d+)\s*(\u00b5m|\u03bcm|um|mm|cm|m)?\s*(?:diameter\s*)?(?:core|inner)",
        r"(?:core|inner)\D{0,20}([-+]?\d*\.?\d+)\s*(\u00b5m|\u03bcm|um|mm|cm|m)?",
    ]
    outer_patterns = [
        r"([-+]?\d*\.?\d+)\s*(\u00b5m|\u03bcm|um|mm|cm|m)?\s*(?:diameter\s*)?(?:outer|outside)",
        r"(?:outer|outside)\D{0,20}([-+]?\d*\.?\d+)\s*(\u00b5m|\u03bcm|um|mm|cm|m)?",
    ]
    for pattern in core_patterns:
        match = re.search(pattern, text)
        if match:
            core = float(match.group(1)) * unit_to_mm(match.group(2))
            break
    for pattern in outer_patterns:
        match = re.search(pattern, text)
        if match:
            outer = float(match.group(1)) * unit_to_mm(match.group(2))
            break
    thickness = None
    if core is not None and outer is not None and outer >= core:
        thickness = (outer - core) / 2.0
    return core, outer, thickness


def canonical_doi(value: str | None, article: str | None = None) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    text = text.rstrip("/. ")

    # Fable found that the older working file had Excel-autofilled DOI suffixes
    # for the Konno ISO-4589-4 paper. Repair that exact physical paper if it is
    # encountered in an input file derived from the old sheet.
    article_text = clean_text(article).lower()
    if (
        "iso 4589" in article_text
        and "konno" in article_text
        and re.match(r"10\.1016/j\.firesaf\.2023\.10\d+$", text)
    ):
        return "10.1016/j.firesaf.2023.103989"

    return text


def canonical_article(value: str | None) -> str:
    text = clean_text(value).lower()
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def paper_id(article: str | None, doi: str | None) -> tuple[str, str, str]:
    doi_c = canonical_doi(doi, article)
    article_c = canonical_article(article)
    return doi_c, article_c, doi_c if doi_c else f"article::{article_c}"


def simple_category(value: str | None, rules: list[tuple[str, str]]) -> str:
    text = clean_text(value).lower()
    for pattern, result in rules:
        if re.search(pattern, text):
            return result
    return "Unknown" if not text else "Other"


def normalise_geometry(value: str | None) -> str:
    return simple_category(
        value,
        [
            ("wire", "Wire"),
            ("flat", "Flat"),
            ("cyl", "Cylindrical"),
            ("spher", "Spherical"),
            ("chunk", "Chunk"),
        ],
    )


def normalise_internal_geometry(value: str | None) -> str:
    return simple_category(value, [("rect", "Rectangular"), ("cyl|circular|annular", "Cylindrical")])


def normalise_facility(value: str | None) -> str:
    text = clean_text(value).lower()
    if not text:
        return "Unknown"
    if "parabolic" in text:
        return "Parabolic Aircraft"
    if "drop" in text:
        return "Drop Tower + Centrifuge" if "centrifuge" in text else "Drop Tower"
    if "centrifuge" in text or "hypergravity" in text:
        return "Centrifuge"
    if "iss" in text or "spacecraft" in text:
        return "Spacecraft / ISS"
    if "sounding" in text or "rocket" in text:
        return "Sounding Rocket"
    if "ground" in text:
        return "Ground"
    return "Other"


def normalise_ignition_method(value: str | None) -> str:
    return simple_category(
        value,
        [
            ("open flame|pilot|match", "Open Flame"),
            ("radiative|heater", "Radiative Heater"),
            ("discharge|high-voltage", "Discharge"),
            ("wire|coil|nicr|electric", "Wire / Coil"),
        ],
    )


def normalise_ignition(value: str | None) -> str:
    text = clean_text(value).lower()
    if text in {"yes", "y", "1", "true"}:
        return "Yes"
    if text in {"no", "n", "0", "false"}:
        return "No"
    return ""


def flow_direction(flow_mm_s: float | None) -> str:
    if flow_mm_s is None:
        return "Unknown"
    if flow_mm_s > 0:
        return "Coflow"
    if flow_mm_s < 0:
        return "Counterflow"
    return "Quiescent"


def gravity_regime(gravity_g: float | None) -> str:
    if gravity_g is None:
        return "Unknown"
    if gravity_g < 1e-3:
        return "Microgravity"
    if gravity_g < 0.95:
        return "Partial"
    if gravity_g <= 1.05:
        return "Earth"
    return "Hyper"


def match_material_family(value: str | None) -> str:
    text = clean_text(value).lower()
    if not text:
        return "Unknown"
    ordered_rules: list[tuple[str, str]] = [
        ("ldpe|low-density polyethylene", "LDPE"),
        ("hdpe|high-density polyethylene", "HDPE"),
        ("etfe", "ETFE"),
        ("pmma|polymethyl methacrylate|organic glass|acrylic/pmma", "PMMA"),
        ("polycarbonate|\\bpc\\b", "Polycarbonate"),
        ("polydimethylsiloxane|pdms|silicone", "Silicone"),
        ("peek", "PEEK"),
        ("ppsu|polyphenylsulfone", "PPSU"),
        ("kapton|polyimide", "Kapton"),
        ("kevlar", "Kevlar"),
        ("nomex|nomeks|conex|aramid|arimid", "Nomex"),
        ("cellulose acetate", "Cellulose acetate"),
        ("sibal", "SIBAL"),
        ("cellulos|kimwipe|whatman|filter paper|tengu|bond paper|paper|cotton|cheesecloth|silkspan|manila|tissue|tspacecraft|felt|leather", "Cellulosic"),
        ("polyethylene-terephthalate|polyethyleneterephthalate|mylar|dacron|\\bpet\\b", "PET"),
        ("polyurethane foam|foam fuel|dacron-wrapped polyurethane", "Polyurethane foam"),
        ("polyurethane", "Polyurethane"),
        ("polyphenolic|phenolic", "Phenolic foam"),
        ("ultem|pei", "PEI"),
        ("abs|acrylbutylstyrene", "ABS"),
        ("polystyrene|\\bps\\b|styrene", "Polystyrene"),
        ("capron|nylon", "Nylon"),
        ("neoprene", "Neoprene"),
        ("paraffin", "Paraffin"),
        ("fluoroplastic|ptfe|fep|teflon", "PTFE/FEP"),
        ("\\bpp\\b|polypropylene", "PP"),
        ("polyoxymethylene|delrin|\\bpom\\b", "POM"),
        ("epoxide|epoxy", "Epoxy"),
        ("polyisobutylene", "Polyisobutylene"),
        ("glass-fiber|glass fiber|fiberglass|stef|p-213", "Fiberglass laminate"),
        ("\\bpe\\b|polyethylene", "PE"),
    ]
    for pattern, family in ordered_rules:
        if re.search(pattern, text):
            return family
    return "Unknown"


def match_core_family(value: str | None) -> str:
    text = clean_text(value).lower()
    if re.search(r"\bcu\b|copper", text):
        return "Copper"
    if "nicr" in text or "nichrome" in text:
        return "NiCr"
    if "stainless" in text or re.search(r"\bss\b", text):
        return "Stainless steel"
    if "steel" in text:
        return "Steel"
    if "iron" in text or re.search(r"\bfe\b", text):
        return "Iron"
    return ""


CORE_PROPS = {
    "Copper": (8960.0, 401.0, 385.0),
    "NiCr": (8400.0, 11.3, 450.0),
    "Stainless steel": (8000.0, 16.0, 500.0),
    "Steel": (7850.0, 45.0, 470.0),
    "Iron": (7870.0, 80.0, 450.0),
}


def header_index(headers: list[str], wanted: str) -> int:
    wanted_norm = wanted.strip().lower()
    for i, header in enumerate(headers):
        if header.strip().lower() == wanted_norm:
            return i
    for i, header in enumerate(headers):
        if header.strip().lower().startswith(wanted_norm):
            return i
    raise KeyError(f"missing header: {wanted}")


def row_value(row: list[str], index: int) -> str:
    return row[index] if index < len(row) else ""


def surface_area_to_volume_1_m(geometry: str, dims_mm: list[float], outer_mm: float | None) -> float | None:
    positive = [d for d in dims_mm if d > 0]
    if geometry == "Wire" and outer_mm and outer_mm > 0:
        return 4.0 / (outer_mm / 1000.0)
    if geometry == "Spherical" and positive:
        diameter_m = max(positive) / 1000.0
        return 6.0 / diameter_m if diameter_m > 0 else None
    if geometry == "Flat" and positive:
        thickness_m = min(positive) / 1000.0
        return 2.0 / thickness_m if thickness_m > 0 else None
    if len(positive) >= 3:
        a, b, c = sorted(positive[:3])
        volume = a * b * c
        area = 2.0 * (a * b + a * c + b * c)
        return (area / volume) * 1000.0 if volume > 0 else None
    return None


def characteristic_length_m(geometry: str, dims_mm: list[float], outer_mm: float | None) -> float | None:
    positive = [d for d in dims_mm if d > 0]
    if geometry == "Wire" and outer_mm and outer_mm > 0:
        return outer_mm / 1000.0
    if not positive:
        return None
    if geometry in {"Flat", "Cylindrical"}:
        return min(positive) / 1000.0
    return max(positive) / 1000.0


def dedupe_key(
    row: list[str],
    idx: dict[str, int],
    canonical_id: str,
    dims: list[float],
    internal_dims: list[float],
    grouped_material: str,
) -> tuple:
    o2 = parse_oxygen_fraction(row_value(row, idx["o2"]))
    pressure = parse_pressure_kpa(row_value(row, idx["pressure"]))
    flow = parse_flow_mm_s(row_value(row, idx["flow"]))
    gravity = parse_gravity_g(row_value(row, idx["gravity"]))
    power = parse_watts(row_value(row, idx["ig_power"]))
    ig_time = first_number(row_value(row, idx["ig_time"]))
    core, outer, insulation = parse_core_outer_mm(row_value(row, idx["dimensions"]))
    sample_dim_key = dimension_feature_key(dims, include_min_max=True)
    internal_dim_key = dimension_feature_key(internal_dims, include_min_max=False)
    return (
        canonical_id,
        normalise_geometry(row_value(row, idx["geometry"])),
        sample_dim_key,
        match_material_family(row_value(row, idx["material"])),
        grouped_material,
        match_core_family(row_value(row, idx["material"])),
        round(o2, 8) if o2 is not None else None,
        round(pressure, 6) if pressure is not None else None,
        round(flow, 6) if flow is not None else None,
        round(gravity, 8) if gravity is not None else None,
        normalise_facility(row_value(row, idx["facility"])),
        normalise_internal_geometry(row_value(row, idx["internal_geom"])),
        internal_dim_key,
        normalise_ignition_method(row_value(row, idx["ig_method"])),
        round(power, 6) if power is not None else None,
        round(ig_time, 6) if ig_time is not None else None,
        round(core, 6) if core is not None else None,
        round(outer, 6) if outer is not None else None,
        round(insulation, 6) if insulation is not None else None,
        normalise_ignition(row_value(row, idx["ignition"])),
    )


def dimension_feature_key(dims: list[float], include_min_max: bool) -> tuple:
    values: list[float | None | int] = []
    for pos in range(3):
        values.append(round(dims[pos], 6) if pos < len(dims) else None)
    if include_min_max:
        values.append(round(min(dims), 6) if dims else None)
        values.append(round(max(dims), 6) if dims else None)
        values.append(round(sum(dims) / len(dims), 6) if dims else None)
        values.append(len(dims))
    else:
        values.append(round(sum(dims) / len(dims), 6) if dims else None)
    return tuple(values)


NEW_COLUMNS: list[tuple[str, str]] = [
    ("Cleaned", "canonical_doi"),
    ("Cleaned", "canonical_article"),
    ("Cleaned", "canonical_paper_id"),
    ("Cleaned", "dedupe_status"),
    ("Cleaned", "normalized_ignition"),
    ("Material Properties", "fuel_material_family"),
    ("Material Properties", "core_material_family"),
    ("Material Properties", "material_density_kg_m3"),
    ("Material Properties", "material_thermal_conductivity_w_m_k"),
    ("Material Properties", "material_heat_capacity_j_kg_k"),
    ("Material Properties", "material_thermal_diffusivity_m2_s"),
    ("Material Properties", "material_pyrolysis_temp_c"),
    ("Material Properties", "material_thermal_inertia_j_m2_k_sqrt_s"),
    ("Material Properties", "limiting_oxygen_index_fraction"),
    ("Material Properties", "oxygen_margin_vs_loi_fraction"),
    ("Material Properties", "core_density_kg_m3"),
    ("Material Properties", "core_thermal_conductivity_w_m_k"),
    ("Material Properties", "core_heat_capacity_j_kg_k"),
    ("Material Properties", "core_thermal_diffusivity_m2_s"),
    ("Material Properties", "material_property_source"),
    ("Material Properties", "material_property_notes"),
    ("Derived Physics", "oxygen_fraction_clean"),
    ("Derived Physics", "pressure_kpa_clean"),
    ("Derived Physics", "pressure_atm"),
    ("Derived Physics", "oxygen_partial_pressure_kpa"),
    ("Derived Physics", "oxygen_partial_pressure_atm"),
    ("Derived Physics", "oxygen_enrichment_ratio_vs_air"),
    ("Derived Physics", "flow_velocity_m_s"),
    ("Derived Physics", "flow_speed_abs_m_s"),
    ("Derived Physics", "flow_direction_clean"),
    ("Derived Physics", "gravity_g_clean"),
    ("Derived Physics", "gravity_m_s2"),
    ("Derived Physics", "gravity_regime_clean"),
    ("Derived Physics", "sample_dim_min_mm"),
    ("Derived Physics", "sample_dim_max_mm"),
    ("Derived Physics", "sample_dim_mean_mm"),
    ("Derived Physics", "core_diameter_mm"),
    ("Derived Physics", "outer_diameter_mm"),
    ("Derived Physics", "insulation_thickness_mm"),
    ("Derived Physics", "characteristic_length_m"),
    ("Derived Physics", "surface_area_to_volume_1_m"),
    ("Derived Physics", "air_density_kg_m3_approx"),
    ("Derived Physics", "air_kinematic_viscosity_m2_s_approx"),
    ("Derived Physics", "reynolds_number_approx"),
    ("Derived Physics", "peclet_number_approx"),
    ("Derived Physics", "froude_number_approx"),
    ("Derived Physics", "richardson_number_approx"),
    ("Derived Physics", "wire_fuel_area_mm2"),
    ("Derived Physics", "wire_fuel_load_kg_m"),
    ("Derived Physics", "flat_fuel_load_kg_m2"),
    ("Derived Physics", "ignition_power_w_clean"),
    ("Derived Physics", "ignition_time_s_clean"),
    ("Derived Physics", "ignition_energy_j_clean"),
    ("Derived Physics", "ignition_energy_per_fuel_mass_j_kg_proxy"),
]


def enrich_row(row: list[str], idx: dict[str, int], dedupe_status: str) -> list[str]:
    article = row_value(row, idx["article"])
    doi = row_value(row, idx["doi"])
    doi_c, article_c, canonical_id = paper_id(article, doi)

    raw_material = row_value(row, idx["material"])
    family = match_material_family(raw_material)
    core_family = match_core_family(raw_material)
    props = MATERIALS.get(family)
    core_tuple = CORE_PROPS.get(core_family)

    o2 = parse_oxygen_fraction(row_value(row, idx["o2"]))
    pressure = parse_pressure_kpa(row_value(row, idx["pressure"]))
    pressure_atm = pressure / 101.325 if pressure is not None else None
    o2_pp_kpa = o2 * pressure if o2 is not None and pressure is not None else None
    o2_pp_atm = o2_pp_kpa / 101.325 if o2_pp_kpa is not None else None
    o2_enrichment = o2 / AIR_O2_FRACTION if o2 is not None else None

    flow_mm_s = parse_flow_mm_s(row_value(row, idx["flow"]))
    flow_m_s = flow_mm_s / 1000.0 if flow_mm_s is not None else None
    flow_abs_m_s = abs(flow_m_s) if flow_m_s is not None else None
    gravity = parse_gravity_g(row_value(row, idx["gravity"]))
    gravity_m_s2 = gravity * G0_M_S2 if gravity is not None else None

    dims = extract_dimensions_mm(row_value(row, idx["dimensions"]))
    positive_dims = [d for d in dims if d > 0]
    dim_min = min(positive_dims) if positive_dims else None
    dim_max = max(positive_dims) if positive_dims else None
    dim_mean = sum(positive_dims) / len(positive_dims) if positive_dims else None
    core_mm, outer_mm, insulation_mm = parse_core_outer_mm(row_value(row, idx["dimensions"]))
    geometry = normalise_geometry(row_value(row, idx["geometry"]))
    char_len_m = characteristic_length_m(geometry, positive_dims, outer_mm)
    sa_v = surface_area_to_volume_1_m(geometry, positive_dims, outer_mm)

    air_density = None
    air_nu = None
    reynolds = None
    peclet = None
    froude = None
    richardson = None
    if pressure is not None:
        air_density = (pressure * 1000.0) / (AIR_GAS_CONSTANT_J_KG_K * REFERENCE_AIR_TEMP_K)
        air_nu = AIR_DYNAMIC_VISCOSITY_PA_S / air_density if air_density > 0 else None
    if flow_abs_m_s is not None and char_len_m is not None and air_density is not None:
        reynolds = air_density * flow_abs_m_s * char_len_m / AIR_DYNAMIC_VISCOSITY_PA_S
    if flow_abs_m_s is not None and char_len_m is not None and pressure is not None:
        air_alpha = AIR_THERMAL_DIFFUSIVITY_M2_S_AT_1ATM * (101.325 / pressure)
        peclet = flow_abs_m_s * char_len_m / air_alpha if air_alpha > 0 else None
    if (
        flow_abs_m_s is not None
        and char_len_m is not None
        and gravity_m_s2 is not None
        and gravity_m_s2 > 0
        and char_len_m > 0
    ):
        froude = flow_abs_m_s / math.sqrt(gravity_m_s2 * char_len_m)
    if (
        flow_abs_m_s is not None
        and flow_abs_m_s > 0
        and char_len_m is not None
        and gravity_m_s2 is not None
    ):
        richardson = gravity_m_s2 * char_len_m / (flow_abs_m_s**2)

    wire_fuel_area_mm2 = None
    wire_fuel_load = None
    flat_fuel_load = None
    if props is not None:
        if outer_mm is not None and core_mm is not None and outer_mm >= core_mm:
            wire_fuel_area_mm2 = math.pi * (outer_mm**2 - core_mm**2) / 4.0
            wire_fuel_load = props.density_kg_m3 * wire_fuel_area_mm2 * 1e-6
        elif geometry == "Flat" and dim_min is not None:
            if family == "SIBAL":
                # Literature reports 18.2 mg/cm2 for the full cotton/fiberglass fabric.
                flat_fuel_load = 0.182
            elif len(positive_dims) >= 3:
                flat_fuel_load = props.density_kg_m3 * (dim_min / 1000.0)

    ignition_power = parse_watts(row_value(row, idx["ig_power"]))
    ignition_time = first_number(row_value(row, idx["ig_time"]))
    ignition_energy = (
        ignition_power * ignition_time
        if ignition_power is not None and ignition_time is not None
        else None
    )
    ignition_energy_per_mass = None
    if ignition_energy is not None and wire_fuel_load is not None and wire_fuel_load > 0:
        ignition_energy_per_mass = ignition_energy / wire_fuel_load

    loi = props.limiting_oxygen_index_fraction if props is not None else None
    o2_margin = o2 - loi if o2 is not None and loi is not None else None
    core_density = core_k = core_cp = core_alpha = None
    if core_tuple is not None:
        core_density, core_k, core_cp = core_tuple
        core_alpha = core_k / (core_density * core_cp)

    material_values = {
        "canonical_doi": doi_c,
        "canonical_article": article_c,
        "canonical_paper_id": canonical_id,
        "dedupe_status": dedupe_status,
        "normalized_ignition": normalise_ignition(row_value(row, idx["ignition"])),
        "fuel_material_family": "" if family == "Unknown" else family,
        "core_material_family": core_family,
        "material_density_kg_m3": fmt(props.density_kg_m3) if props else "",
        "material_thermal_conductivity_w_m_k": fmt(props.thermal_conductivity_w_m_k) if props else "",
        "material_heat_capacity_j_kg_k": fmt(props.heat_capacity_j_kg_k) if props else "",
        "material_thermal_diffusivity_m2_s": fmt(props.thermal_diffusivity_m2_s) if props else "",
        "material_pyrolysis_temp_c": fmt(props.pyrolysis_temp_c) if props else "",
        "material_thermal_inertia_j_m2_k_sqrt_s": fmt(props.thermal_inertia) if props else "",
        "limiting_oxygen_index_fraction": fmt(loi),
        "oxygen_margin_vs_loi_fraction": fmt(o2_margin),
        "core_density_kg_m3": fmt(core_density),
        "core_thermal_conductivity_w_m_k": fmt(core_k),
        "core_heat_capacity_j_kg_k": fmt(core_cp),
        "core_thermal_diffusivity_m2_s": fmt(core_alpha),
        "material_property_source": props.source if props else "",
        "material_property_notes": props.notes if props else "",
        "oxygen_fraction_clean": fmt(o2),
        "pressure_kpa_clean": fmt(pressure),
        "pressure_atm": fmt(pressure_atm),
        "oxygen_partial_pressure_kpa": fmt(o2_pp_kpa),
        "oxygen_partial_pressure_atm": fmt(o2_pp_atm),
        "oxygen_enrichment_ratio_vs_air": fmt(o2_enrichment),
        "flow_velocity_m_s": fmt(flow_m_s),
        "flow_speed_abs_m_s": fmt(flow_abs_m_s),
        "flow_direction_clean": flow_direction(flow_mm_s),
        "gravity_g_clean": fmt(gravity),
        "gravity_m_s2": fmt(gravity_m_s2),
        "gravity_regime_clean": gravity_regime(gravity),
        "sample_dim_min_mm": fmt(dim_min),
        "sample_dim_max_mm": fmt(dim_max),
        "sample_dim_mean_mm": fmt(dim_mean),
        "core_diameter_mm": fmt(core_mm),
        "outer_diameter_mm": fmt(outer_mm),
        "insulation_thickness_mm": fmt(insulation_mm),
        "characteristic_length_m": fmt(char_len_m),
        "surface_area_to_volume_1_m": fmt(sa_v),
        "air_density_kg_m3_approx": fmt(air_density),
        "air_kinematic_viscosity_m2_s_approx": fmt(air_nu),
        "reynolds_number_approx": fmt(reynolds),
        "peclet_number_approx": fmt(peclet),
        "froude_number_approx": fmt(froude),
        "richardson_number_approx": fmt(richardson),
        "wire_fuel_area_mm2": fmt(wire_fuel_area_mm2),
        "wire_fuel_load_kg_m": fmt(wire_fuel_load),
        "flat_fuel_load_kg_m2": fmt(flat_fuel_load),
        "ignition_power_w_clean": fmt(ignition_power),
        "ignition_time_s_clean": fmt(ignition_time),
        "ignition_energy_j_clean": fmt(ignition_energy),
        "ignition_energy_per_fuel_mass_j_kg_proxy": fmt(ignition_energy_per_mass),
    }
    return [material_values[name] for _, name in NEW_COLUMNS]


def generate(input_path: Path, output_path: Path) -> tuple[int, int]:
    with input_path.open("r", encoding="cp1252", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"{input_path} must contain the two Fable header rows")

    section_row = list(rows[0])
    header_row = list(rows[1])
    data_rows = rows[2:]

    idx = {
        "article": header_index(header_row, "Article (MLA)"),
        "authors": header_index(header_row, "Authors"),
        "doi": header_index(header_row, "DOI"),
        "geometry": header_index(header_row, "Geometry of Sample"),
        "dimensions": header_index(header_row, "Dimensions of sample"),
        "material": header_index(header_row, "Material of sample"),
        "o2": header_index(header_row, "Oxygen Concentration"),
        "pressure": header_index(header_row, "Pressure"),
        "flow": header_index(header_row, "Flow Velocity"),
        "rig": header_index(header_row, "Rig Name"),
        "internal_geom": header_index(header_row, "Internal geometry"),
        "internal_dims": header_index(header_row, "Internal Dimensions"),
        "gravity": header_index(header_row, "Gravity"),
        "facility": header_index(header_row, "Expireimental facility"),
        "ig_method": header_index(header_row, "Ignition method"),
        "ig_power": header_index(header_row, "Ignition power"),
        "ig_time": header_index(header_row, "Ignition time"),
        "ignition": header_index(header_row, "Ignition (Yes/No)"),
    }

    while len(section_row) < len(header_row):
        section_row.append("")
    section_row = section_row[: len(header_row)]
    output_rows = [
        section_row + [section for section, _ in NEW_COLUMNS],
        header_row + [name for _, name in NEW_COLUMNS],
    ]

    material_counts: dict[str, int] = {}
    for row in data_rows:
        material = clean_text(row_value(row, idx["material"]))
        material_counts[material] = material_counts.get(material, 0) + 1

    seen: set[tuple] = set()
    dropped = 0
    kept = 0
    for row in data_rows:
        if not any(clean_text(value) for value in row):
            continue
        padded = row + [""] * (len(header_row) - len(row))
        doi_c, _article_c, canonical_id = paper_id(
            row_value(padded, idx["article"]), row_value(padded, idx["doi"])
        )
        dims = extract_dimensions_mm(row_value(padded, idx["dimensions"]))
        internal_dims = extract_dimensions_mm(row_value(padded, idx["internal_dims"]))
        ignition = normalise_ignition(row_value(padded, idx["ignition"]))
        material = clean_text(row_value(padded, idx["material"]))
        grouped_material = (
            material if material_counts.get(material, 0) >= 10 else "Other / Rare"
        )
        status = "kept"
        if ignition:
            key = dedupe_key(
                padded, idx, canonical_id, dims, internal_dims, grouped_material
            )
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            status = "kept_deduplicated_labelled"
        else:
            status = "kept_unlabelled"
        output_rows.append(padded[: len(header_row)] + enrich_row(padded, idx, status))
        kept += 1

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(output_rows)

    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    kept, dropped = generate(args.input, args.output)
    print(f"wrote {args.output}")
    print(f"kept rows: {kept}")
    print(f"dropped duplicate labelled rows: {dropped}")


if __name__ == "__main__":
    main()
