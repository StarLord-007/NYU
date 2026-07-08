# Physics_DB — Redesigned Microgravity Combustion Dataset

**Source:** `Microgravity_Database.csv` (5,057 usable rows, 87 papers)
**Output:** `Physics_DB.csv` — 57 physics-based features + `group_id` + 2 targets (`target_ignition`, `target_fsr_mm_s`)

**Design principle:** every feature must be a physical quantity a combustion model could
use, so that a model trained on some papers transfers to unseen ones. All paper-, lab-,
and rig-identity information is removed from the feature space; `group_id` is retained
**only** as the Leave-One-Paper-Out (LOPO) split key and must never be used as a feature.

---

## 1. Features removed (leakage sources)

| Removed | Reason |
|---|---|
| `Article`, `Authors`, `DOI` | Pure paper identity → memorization under random splits |
| `Rig Name` | Rig identity is a proxy for the paper; replaced by physical confinement scale |
| `Material of sample` (name string) | Categorical names don't transfer; replaced by thermophysical properties |
| `Experimental facility` | Proxy for lab; the physics is already carried by `g_ratio` and flow variables |
| Free-text notes, figure references | Non-physical metadata (physical content extracted first, see §3) |
| `Flame Length`, `HRR`, `Smoke` | 96–99% missing; auxiliary outputs, not features |

## 2. Features transformed / parsed

- **Multi-row header** collapsed (row 2 used as header, unnamed trailing columns folded into notes parsing).
- **Dimensions** (free text, mixed units/formats) → numeric `length_mm`, `width_mm`, `thickness_mm`,
  `diameter_mm`, `core_dia_mm`, `coating_mm`, `hollow` flag. Handles flat sheets, rods, tubes,
  spheres, and coated-wire patterns ("0.5 mm core; 0.8 mm outer", "4.0 mm outer; 2.5 mm Cu core; …").
- **Composite (insulated-wire) samples** split into **core** vs **fuel coating**:
  metal core properties (`core_density_kg_m3`, `core_k_W_mK`, `core_cp_J_kgK`) taken from
  Ni-Cr / Cu / stainless tables; hollow "mix-Air-metal" cores use the effective properties
  stated in the source notes. Flame-retarded LDPE grades encoded as `fr_type` (AP/EG) +
  `fr_mass_fraction` rather than as separate material names.
- **Chamber**: rig identity replaced by `chamber_min_dimension_mm` (physical confinement scale
  parsed from internal dimensions).
- **Ignition source**: method normalized to `hot_wire` / `pilot_flame` / `radiative`;
  power × time → `ignition_energy_J`.
- **Notes mined for physics**: diluent gas (N₂/He/Ar/CO₂), external radiant flux (kW/m²),
  initial sample temperature (K), orientation, spread direction.

## 3. Features added

### Fuel thermophysical properties (replace material name)
`fuel_density_kg_m3`, `fuel_k_W_mK`, `fuel_cp_J_kgK`, `fuel_pyrolysis_T_K`,
`fuel_heat_of_combustion_MJ_kg` — mapped from a ~50-entry lookup covering PMMA, LDPE/HDPE,
ETFE, PTFE, cellulosics (tissue/filter/bond paper/cotton/SIBAL/cheesecloth), silicones,
PC, Nomex/Kevlar/polyimide, PPSU/PEEK/PEI/PSU, PET, PU and phenolic foams, paraffin, PS/ABS,
nylon, POM, epoxy, GFRP, carbon fabric, etc. 100% of rows are mapped.

### Environment physics
- `pO2_kPa` = X(O₂) × P — the true oxidizer availability variable (flammability limits collapse
  onto pO₂ far better than onto X(O₂) or P separately).
- Diluent-aware **gas mixture properties**: molar mass, mass-specific cₚ, conductivity,
  density (ideal gas at T_init), thermal diffusivity, kinematic viscosity. This lets the
  model treat "21% O₂ in He" and "21% O₂ in CO₂" as physically different atmospheres —
  exactly what governs quenching/enhancement in diluent-swap experiments.

### Flow + gravity unified
- `buoyant_velocity_mm_s` = (g·ν)^{1/3} — the characteristic buoyant-induction velocity;
  puts drop-tower, parabolic, centrifuge and ground data on **one velocity axis**.
- `effective_velocity_mm_s` = |forced flow| + buoyant velocity (dominant oxidizer-supply speed).
- `flow_is_opposed` (opposed vs concurrent spread — different physical regimes).
- Dimensionless groups: `Re_flow`, `Gr_flow`, and the mixed-convection parameter `Gr/Re²`.

### Solid-phase derived physics
- `fuel_thermal_inertia_krc` (kρc), `fuel_thermal_diffusivity_m2_s`.
- `fuel_half_thickness_mm` — characteristic conduction depth (half sheet thickness, wire
  coating thickness, rod radius, tube wall).
- `fuel_areal_heat_capacity_J_m2K` = ρcδ — the thermally-thin fuel controlling parameter.
- `thermal_thickness_ratio` = δ / (α_g/U) — thermally-thin vs thermally-thick regime indicator.
- `deRis_thin_FSR_mm_s` — de Ris thermally-thin closed-form spread rate
  V_f = √2·λ_g·(T_f−T_p) / (ρcδ·(T_p−T∞)), with a flame temperature estimate
  `flame_temp_est_K` linear in X(O₂). This physics prior alone achieves log-correlation
  ≈ 0.63 with measured FSR across all 87 papers — a strong transferable anchor feature.
- `core_heat_sink_ratio` (ρcA of core / ρcA of fuel) and `core_conductance_ratio` (k_core/k_fuel)
  — capture the two competing wire-core effects (heat sink vs conductive preheating) with
  dimensionless numbers instead of core material names.

## 4. Physical justification (summary)

Ignition and opposed-flow flame spread over solids are controlled by: oxidizer supply
(pO₂, effective velocity, diluent transport properties), gas-phase heat feedback (λ_g, T_f),
solid heat-up (kρc or ρcδ depending on thermal thickness), pyrolysis threshold (T_p), and
heat losses/sinks (core, radiation, confinement). The redesigned feature set expresses each
row **only** in these terms, so two experiments from different labs with the same physics
now have (nearly) the same feature vector — the property required for LOPO generalization.

## 5. Assumptions

- Material properties are **nominal room-temperature literature values** (SFPE Handbook,
  polymer datasheets, NASA/NIST fire-safety reports); grade-to-grade variation is ignored.
- Fabrics/tissues use *effective* (bulk) density, not fiber density.
- Pyrolysis temperatures are representative onset values (±30–50 K uncertainty).
- Flame temperature estimate: T_f ≈ 1400 K + 2500·(X_O₂ − 0.21), clipped to [1200, 3000] K.
- Kinematic viscosity from μ_air ≈ 1.85×10⁻⁵ Pa·s (mixture μ variation neglected); Gr uses βΔT ≈ 4.
- Diluent defaults to N₂ when unstated. Initial temperature defaults to 293 K when unstated.
- Missing forced-flow velocity treated as 0 (quiescent) when computing effective velocity;
  the raw `flow_velocity_mm_s` retains NaN so models can distinguish "unreported."

## 6. Material property sources

SFPE Handbook of Fire Protection Engineering (5th ed., thermophysical property tables);
Tewarson heat-of-combustion compilations; NASA STD-6001/White Sands materials test reports
(Nomex, Kevlar, Kapton, PTFE, ETFE); polymer supplier datasheets (PMMA, LDPE, HDPE, PC,
PEEK, PPSU, Ultem); FLARE/JAXA and NASA GRC wire-insulation papers for NiCr/Cu core and
effective hollow-core properties (the latter taken directly from this database's own notes).

## 7. Remaining limitations

- Single nominal property set per material class; no temperature dependence, char/melt
  behavior, or in-depth radiation absorption is represented.
- `flame_temp_est_K` / `deRis_thin_FSR_mm_s` use a crude flame-temperature correlation;
  they are priors, not ground truth.
- ~4% of rows lack a parseable characteristic thickness; ignition power/time are 55–81% missing
  (reflected as NaN, imputation left to the modeling stage).
- Sample orientation/spread direction are sparsely reported (~1%) — relevant only at 1 g.
- FSR units are as-reported (mm/s assumed); a few extreme pressures (up to 30 atm OI tests,
  one 6,343 kPa entry) are retained un-curated — screen outliers before regression.
- Residual paper–condition confounding is unavoidable: some physics regions (e.g., He diluent,
  centrifuge hypergravity) exist in only one paper, so LOPO on those groups tests true
  extrapolation.

## 8. Usage note

- **Grouping:** use `group_id` for GroupKFold / Leave-One-Paper-Out. **Never feed it to the model.**
- **Targets:** `target_ignition` (1/0, n = 5,057) and `target_fsr_mm_s` (n = 2,590).
- Recommended: log-transform FSR, Re, Gr, kρc, and areal heat capacity before training.
