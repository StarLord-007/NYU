# Ignition Classifier Feature Engineering


## Atmosphere features

### Oxygen fraction and oxygen partial pressure

Convert volumetric oxygen percent to mole/volume fraction:


$$
X_{O_2}=\frac{\text{Oxygen Concentration}}{100}
$$
Convert pressure from kPa to Pa:


$$
P=1000\,P_{\mathrm{kPa}}
$$
Then calculate oxygen partial pressure:


$$
p_{O_2}=X_{O_2}P
$$
**Implementation:** create `X_O2` and `p_O2_Pa`. Keep both, as the classifier may learn different effects of oxygen fraction and total pressure.

### Oxygen mass fraction


$$
Y_{O_2}=\frac{X_{O_2}M_{O_2}}{M_{\mathrm{gas}}}
$$
where $M_{O_2}=0.031998\ \mathrm{kg/mol}$ and $M_{\mathrm{gas}}$ is `gas_M`.

**Implementation:** create `Y_O2`. Use it in mass-based oxygen transport features.

### Gas-property consistency feature

The database already stores `gas_alpha_m2_s`. Optionally check it using:


$$
\alpha_g^*=\frac{k_g}{\rho_gc_{p,g}}
$$
where $k_g$ is `gas_k`, $\rho_g$ is `gas_density_kg_m3`, and $c_{p,g}$ is `gas_cp_mass`.

**Implementation:** use `gas_alpha_m2_s` as the model input; use $\alpha_g^*$ only for quality control.

### Dynamic viscosity


$$
\mu_g=\rho_g\nu_g
$$
where $\nu_g$ is `gas_nu_m2_s`.

**Implementation:** create `gas_mu_Pa_s` for Reynolds number.

---

## Fuel thermal features

### Volumetric heat capacity


$$
C_{\mathrm{fuel,vol}}=\rho_sc_{p,s}
$$
where $\rho_s$ is `fuel_density_kg_m3` and $c_{p,s}$ is `fuel_cp_J_kgK`.

**Implementation:** create `fuel_volumetric_heat_capacity_J_m3K`.

### Fuel thermal diffusivity check


$$
\alpha_s^*=\frac{k_s}{\rho_sc_{p,s}}
$$
where $k_s$ is `fuel_k_W_mK`.

**Implementation:** use `fuel_alpha_m2_s` as the primary feature. Use $\alpha_s^*$ only to identify possible property or unit problems.

### Pyrolysis temperature rise

Choose a fixed reference ambient temperature if individual ambient temperatures are unavailable, for example $T_\infty=298.15\ \mathrm{K}$:


$$
\Delta T_{\mathrm{pyro}}=T_{\mathrm{pyro}}-T_\infty
$$
where $T_{\mathrm{pyro}}$ is `fuel_pyrolysis_T_K`.

**Implementation:** create `delta_T_pyro_K`.

### Fuel heating demand


$$
H_{\mathrm{fuel,vol}}=\rho_sc_{p,s}(T_{\mathrm{pyro}}-T_\infty)
$$
This gives the sensible energy requirement per unit volume to reach pyrolysis temperature.

**Implementation:** create `fuel_heating_demand_J_m3`. This is a high-value feature because it combines density, heat capacity, and pyrolysis temperature.

---

## Flow and transport features

### Flow-direction variables

Since your convention is negative = opposed and positive = concurrent:


$$
U_{\mathrm{opp}}=\max(-U,0)
$$
$$
U_{\mathrm{co}}=\max(U,0)
$$
$$
I_{\mathrm{opp}}=\begin{cases}1,&U<0\\0,&U\ge0\end{cases}
\qquad
I_{\mathrm{co}}=\begin{cases}1,&U>0\\0,&U\le0\end{cases}
$$
**Implementation:** retain raw signed `Flow Velocity` and add `U_opp`, `U_co`, `I_opp`, and `I_co`.

### Characteristic length

Use a geometry-specific characteristic size $L_c$:

| Geometry | Use for $L_c$ |
|---|---|
| Flat | Specimen length aligned with flow/spread |
| Wire | Outer diameter $d_o$ |
| Cylinder | Outer diameter $d$ |
| Sphere | Diameter $d$ |

### Reynolds number


$$
Re=\frac{\rho_g|U|L_c}{\mu_g}=\frac{|U|L_c}{\nu_g}
$$
**Implementation:** create `Re`. It combines velocity, gas momentum transport, and specimen scale.

### Thermal Péclet number


$$
Pe=\frac{|U|L_c}{\alpha_g}
$$
**Implementation:** create `Pe`. It quantifies advection relative to thermal diffusion.

### Prandtl number


$$
Pr=\frac{\nu_g}{\alpha_g}
$$
**Implementation:** create `Pr`.

### Gas preheat length


$$
L_g=\frac{\alpha_g}{\max(|U|,U_{\min})}
$$
Use $U_{\min}=0.001\ \mathrm{m/s}$, or another declared small cutoff, to avoid divergence at nominal zero flow.

**Implementation:** create `gas_preheat_length_m`. Do not interpret it as an exact physical flame length at zero flow; it is a bounded regime feature.

---

## Gravity features

### Normalized gravity

The database already stores:


$$
G_{\mathrm{norm}}=\text{Gravity}
$$
To obtain SI acceleration only when needed:


$$
g=9.81G_{\mathrm{norm}}
$$
**Implementation:** use raw `Gravity` as `G_norm`; do not divide it by 9.81 again.

### Buoyancy velocity scale

If you select a fixed representative flame temperature $T_f$ and ambient temperature $T_\infty$, calculate:


$$
U_{\mathrm{NC}}\sim\left[\frac{g\alpha_g(T_f-T_\infty)}{T_\infty}\right]^{1/3}
$$
**Implementation:** treat `U_NC` as optional. It is a physics-guided proxy, not a measured velocity. Avoid using it unless your assumed temperatures are documented consistently.

### Gravity-flow interaction


$$
\Pi_{gU}=G_{\mathrm{norm}}Pe
$$
**Implementation:** create `gravity_Pe_interaction`. It distinguishes, for example, a high-flow microgravity condition from an equivalent high-flow 1g condition.

---

## Flat-sample equations

Apply only when `Geometry of Sample = Flat`. Parse the dimensions as $L\times H\times W$ and convert mm to m.

### Half thickness


$$
\delta=\frac{H}{2}
$$
### Surface area and volume


$$
A_{\mathrm{flat}}=2(LW+LH+WH)
$$
$$
V_{\mathrm{flat}}=LHW
$$
### Surface-to-volume ratio


$$
\Pi_{\mathrm{SV,flat}}=\frac{A_{\mathrm{flat}}}{V_{\mathrm{flat}}}
$$
### Aspect ratios


$$
\Pi_{H/L}=\frac{H}{L}
$$
$$
\Pi_{H/W}=\frac{H}{W}
$$
### Through-thickness thermal diffusion time


$$
t_{\mathrm{diff,flat}}=\frac{\delta^2}{\alpha_s}
$$
### Whole-specimen heating scale


$$
E_{\mathrm{heat,flat}}=\rho_sc_{p,s}V_{\mathrm{flat}}(T_{\mathrm{pyro}}-T_\infty)
$$
**Implement for flat samples:** `delta_m`, `A_flat_m2`, `V_flat_m3`, `SV_flat_m_inv`, `H_over_L`, `H_over_W`, `t_diff_flat_s`, and `E_heat_flat_J`.

---

## Wire equations

Apply only when `Geometry of Sample = Wire`. Parse `Dimensions of sample` to obtain core diameter $d_c$ and outer diameter $d_o$; convert both from mm to m.

### Radii and insulation thickness


$$
r_c=\frac{d_c}{2}
$$
$$
r_o=\frac{d_o}{2}
$$
$$
\delta_{\mathrm{ins}}=r_o-r_c
$$
### Core and insulation cross-sectional areas


$$
A_{\mathrm{core}}=\pi r_c^2
$$
$$
A_{\mathrm{ins}}=\pi(r_o^2-r_c^2)
$$
### Surface-to-volume ratio of insulation


$$
\Pi_{\mathrm{SV,wire}}=\frac{2r_o}{r_o^2-r_c^2}
$$
### Core-to-insulation ratio


$$
\Pi_{\mathrm{core/ins}}=\frac{r_c^2}{r_o^2-r_c^2}
$$
### Core volumetric heat capacity


$$
C_{\mathrm{core,vol}}=\rho_cc_{p,c}
$$
### Core axial conduction proxy


$$
G_{\mathrm{core}}=k_cA_{\mathrm{core}}
$$
### Dimensionless core-heating proxy

Choose $L_{\mathrm{ref}}=d_o$:


$$
\Pi_{\mathrm{core,heat}}=\frac{k_cA_{\mathrm{core}}}{k_g(2\pi r_o)L_{\mathrm{ref}}}
$$
### Insulation heating demand per unit length


$$
H'_{\mathrm{ins}}=\rho_sc_{p,s}\pi(r_o^2-r_c^2)(T_{\mathrm{pyro}}-T_\infty)
$$
### Curvature/preheat ratio


$$
\Pi_{\mathrm{curv,wire}}=\frac{L_g}{r_o}
$$
**Implement for wire samples:** `r_core_m`, `r_outer_m`, `insulation_thickness_m`, `A_core_m2`, `A_ins_m2`, `SV_wire_m_inv`, `core_insulation_ratio`, `core_volumetric_heat_capacity_J_m3K`, `core_conduction_proxy_WmK`, `core_heat_ratio`, `insulation_heating_demand_per_m_J_m`, and `curvature_ratio_wire`.

---

## Cylinder and sphere equations

### Cylinder

For radius $r_s$:


$$
A'_{\mathrm{cyl}}=2\pi r_s
$$
$$
V'_{\mathrm{cyl}}=\pi r_s^2
$$
$$
\Pi_{\mathrm{SV,cyl}}=\frac{2}{r_s}
$$
### Sphere

For radius $r_s$:


$$
A_{\mathrm{sph}}=4\pi r_s^2
$$
$$
V_{\mathrm{sph}}=\frac{4}{3}\pi r_s^3
$$
$$
\Pi_{\mathrm{SV,sph}}=\frac{3}{r_s}
$$
### Solid diffusion time


$$
t_{\mathrm{diff}}=\frac{r_s^2}{\alpha_s}
$$
### Curvature/preheat ratio


$$
\Pi_{\mathrm{curv}}=\frac{L_g}{r_s}
$$
---

## Recommended feature blocks

### Use for every classifier

```text
Geometry of Sample
X_O2
p_O2_Pa
Y_O2
diluent
Pressure (kPa)
gas_M
gas_cp_mass
gas_k
gas_density_kg_m3
gas_alpha_m2_s
gas_nu_m2_s
fuel_density_kg_m3
fuel_k_W_mK
fuel_cp_J_kgK
fuel_pyrolysis_T_K
fuel_alpha_m2_s
fuel_volumetric_heat_capacity_J_m3K
delta_T_pyro_K
fuel_heating_demand_J_m3
Flow Velocity
U_opp
U_co
I_opp
I_co
Re
Pe
Pr
gas_preheat_length_m
Gravity
gravity_Pe_interaction
Facility
Internal Geometry
Internal Dimensions
Ignition Method
```

### Add only for flat samples

```text
delta_m
SV_flat_m_inv
H_over_L
H_over_W
t_diff_flat_s
E_heat_flat_J
```

### Add only for wire samples

```text
insulation_thickness_m
SV_wire_m_inv
core_insulation_ratio
core_volumetric_heat_capacity_J_m3K
core_conduction_proxy_WmK
core_heat_ratio
insulation_heating_demand_per_m_J_m
curvature_ratio_wire
```

### Avoid using in the initial ignition classifier

```text
Ignition Power (W)
Ignition Time (s)
Flame Length
FSR
HRR
Smoke/ Areosols
```

Ignition power and time are excluded because they are too incomplete in the current database. Flame length, FSR, HRR, and smoke are excluded because they occur after the ignition outcome.

---

## Missingness and geometry handling

### Core properties

Core properties are not applicable to flat, spherical, and solid-cylinder samples. Do not set them to zero without a geometry flag.

Create:


$$
I_{\mathrm{wire}}=\begin{cases}1,&\text{Geometry = Wire}\\0,&\text{otherwise}\end{cases}
$$
Keep wire-only variables missing or impute them after adding `I_wire`.

### Geometry-specific models

A practical setup is:

1. Train one global classifier using universal features plus geometry labels.
2. Train a separate wire-only model if wire records are numerous enough.
3. Train a flat-only model if flat records are numerous enough.
4. Compare global versus geometry-specific validation performance.

---

## Model-specific implementation

### XGBoost

Use the complete engineered feature set. It is the strongest starting baseline because it naturally learns nonlinear thresholds and interactions.

Recommended interactions to add explicitly:


$$
p_{O_2}\times H_{\mathrm{fuel,vol}}^{-1}
$$
$$
G_{\mathrm{norm}}\times Pe
$$
$$
U_{\mathrm{opp}}\times \Pi_{\mathrm{SV}}
$$
### Decision Tree

Use a compact feature set for interpretability:

```text
p_O2_Pa
fuel_heating_demand_J_m3
delta_m or insulation_thickness_m
SV ratio
Re
Pe
Gravity
Geometry of Sample
diluent
```

### KNN

Standardize all numerical features before fitting:


$$
x_{\mathrm{scaled}}=\frac{x-\mu_x}{\sigma_x}
$$
Fit $\mu_x$ and $\sigma_x$ using only training data in each split.

Use a compact, low-redundancy set such as `p_O2_Pa`, `fuel_heating_demand_J_m3`, surface-to-volume ratio, `Re`, `Pe`, `Gravity`, `fuel_alpha_m2_s`, and `gas_alpha_m2_s`.

### MLP

Standardize all numerical features and one-hot encode categorical features. Use geometry, diluent, facility, internal geometry, and ignition method as categorical inputs.

### SVM

Standardize numerical features and use a compact feature set to avoid unstable kernels from many correlated variables. Start with `p_O2_Pa`, `fuel_heating_demand_J_m3`, surface-to-volume ratio, `Re`, `Pe`, `Gravity`, `fuel_alpha_m2_s`, and `gas_alpha_m2_s`.

---

## Validation protocol

Use grouped cross-validation, not a fully random row split, because many rows come from the same study and experimental campaign.

Recommended grouping keys:

```text
DOI
Citation
Facility
Rig Name
```

Report at least:


$$
\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$
$$
\mathrm{Precision}=\frac{TP}{TP+FP}
$$
$$
\mathrm{Recall}=\frac{TP}{TP+FN}
$$
$$
F1=\frac{2(\mathrm{Precision})(\mathrm{Recall})}{\mathrm{Precision}+\mathrm{Recall}}
$$
Also report ROC-AUC; if no-ignition cases are relatively scarce, report precision-recall AUC as well.

---

## Implementation checklist

- [ ] Exclude post-ignition outputs: Flame Length, FSR, HRR, Smoke/Aerosols
- [ ] Convert oxygen from volumic percent to $X_{O_2}$
- [ ] Calculate $p_{O_2}$ and $Y_{O_2}$
- [ ] Preserve signed raw flow and add opposed/coflow decomposition
- [ ] Use `Gravity` directly as $g/g_{\oplus}$
- [ ] Parse dimensions by geometry
- [ ] Calculate geometry-specific features only in their relevant branch
- [ ] Exclude ignition power/time from the initial model due to sparsity
- [ ] One-hot encode geometry, diluent, facility, internal geometry, and ignition method
- [ ] Scale numeric features for KNN, MLP, and SVM
- [ ] Use grouped cross-validation by DOI, citation, facility, or rig
