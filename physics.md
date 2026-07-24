<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# @Google Drive Here is my database, look at it (especially the features set I have) and give all the equations that I can implement.

Your table supports a **regime-gated flame-spread model**, not one universal FSR equation. You have the key predictors needed to implement this: geometry, dimensions, material, oxygen concentration, pressure, signed flow velocity, gravity, apparatus/enclosure descriptors, ignition inputs, and ignition outcome/FSR.[^1]

The important practical split is:

1. **Ignition/sustained-flame classifier:** `Ignition` as Yes/No
2. **Conditional FSR regressor:** predict FSR only for `Ignition = Yes`
3. **Separate model families:** flat solids, wires, and cylindrical samples should not share the same analytical closure without geometry-specific terms. Your database explicitly contains all three geometries.[^1]

## Database variables

The file’s core columns appear to be:

$$
\{\text{geometry},\ \text{dimensions},\ \text{material},\ X_{O_2},\ P,\ U,\ \text{chamber geometry},\ \text{chamber dimensions},\ g,\ \text{facility},\ \text{ignition method},\ Q_{\mathrm{ign}},\ t_{\mathrm{ign}}\}
$$

with outputs:

$$
\{\text{Ignition},\ L_f,\ V_f,\ \dot Q,\ \text{smoke/aerosols}\}
$$

where $V_f$ is FSR. The database includes, for example, flat LDPE/HDPE samples, LDPE-insulated wires with Cu and NiCr cores, varying oxygen levels, pressure, flow, partial gravity, and ignition outcomes.[^1]

### First transformations

Use SI units internally:

$$
X_{O_2}=\frac{\text{Oxygen Concentration}}{100}
$$

$$
P[\mathrm{Pa}]
=
1000\,P[\mathrm{kPa}]
$$

$$
U[\mathrm{m/s}]
=
\frac{U[\mathrm{mm/s}]}{1000}
$$

$$
d[\mathrm{m}]
=
\frac{d[\mathrm{mm}]}{1000}
$$

$$
\delta[\mathrm{m}]
=
\frac{\text{flat-sample thickness [mm]}}{2000}
$$

For a flat specimen, $\delta$ is the half-thickness; for a wire, use the outer insulation radius and core radius separately:

$$
r_o=\frac{d_o}{2},
\qquad
r_c=\frac{d_c}{2},
\qquad
\delta_{\mathrm{ins}}=r_o-r_c
$$

For a solid cylinder:

$$
r_s=\frac{d}{2}
$$

Your `Dimensions of sample` field contains the needed geometry parameters, but it needs parsing into standardized numeric columns before any equation can be implemented reliably.[^1]

***

## Output equations

Use these as the target definitions, depending on what raw data you retain.

### Flame-spread rate

$$
V_f=\frac{dx_f}{dt}
$$

This is the primary FSR target. If the database contains only a reported FSR, retain it as the target. If it later includes flame-front position versus time, calculate $V_f$ from the fitted slope after ignition transient.[^2][^3]

### Ignition energy

$$
E_{\mathrm{ign}}=Q_{\mathrm{ign}}t_{\mathrm{ign}}
$$

- $Q_{\mathrm{ign}}$: ignition power, W
- $t_{\mathrm{ign}}$: ignition duration, s
- $E_{\mathrm{ign}}$: delivered ignition energy, J

Your database has both ignition power and ignition time, so this is immediately implementable. It should be a feature in the ignition classifier; it should not automatically be assumed to control the post-ignition steady FSR.[^1]

### FSR plus extinction formulation

For modeling all cases—including “No ignition” entries—use:

$$
\widehat{V}_{f,\mathrm{all}}
=
\widehat{p}_{\mathrm{ign}}
\,
\widehat{V}_{f\mid\mathrm{ign}}
$$

- $\widehat{p}_{\mathrm{ign}}$: predicted probability that a sustained flame occurs
- $\widehat{V}_{f\mid\mathrm{ign}}$: predicted FSR conditional on sustained ignition

This avoids falsely treating “No ignition” as a low but known continuous FSR. The database includes many near-limit cases where ignition is explicitly No, particularly at lower oxygen concentration.[^1]

***

## Shared atmosphere equations

These are implementable for **every row** once you standardize oxygen, pressure, and a representative initial temperature.

### Oxygen partial pressure

$$
p_{O_2}=X_{O_2}P
$$

This should be a primary feature for both ignition and FSR. It combines the two independent database columns that determine oxygen availability.[^1]

### Inert-gas fraction

If the atmosphere is binary oxygen plus diluent:

$$
X_{\mathrm{dil}}=1-X_{O_2}
$$

If the actual diluent identity is known or can be derived from the source paper, do **not** collapse nitrogen, helium, argon, and carbon dioxide into one category. Their thermal conductivity, heat capacity, oxygen diffusivity, Lewis number, and radiative participation differ materially. The database currently exposes oxygen concentration but not a dedicated atmosphere/diluent column in the retrieved schema, so add one.[^4][^2][^1]

### Ideal-gas density

$$
\rho_g
=
\frac{P\overline{M}}{R_uT_\infty}
$$

- $\overline{M}$: mixture molecular weight
- $R_u$: universal gas constant
- $T_\infty$: initial ambient gas temperature

This is implementable only after you add or infer diluent composition and record/assume an initial temperature. It is essential for Reynolds, buoyancy, and oxygen-flux features.

### Oxygen mass fraction

$$
Y_{O_2}
=
\frac{X_{O_2}M_{O_2}}{\overline{M}}
$$

This is preferred over mole fraction in oxygen mass-flux equations.[^5]

### Gas diffusivity scaling

A useful engineering scaling when a measured or database gas-diffusivity value is unavailable is:

$$
D_{O_2,\mathrm{mix}}
\approx
D_{\mathrm{ref}}
\left(\frac{T_\infty}{T_{\mathrm{ref}}}\right)^{n_D}
\left(\frac{P_{\mathrm{ref}}}{P}\right)
$$

A common approximate exponent is $n_D\approx1.5$ to 1.75. Treat this as a derived engineering feature, not a ground-truth material property; mixture-specific diffusion coefficients are preferable.

### Thermal diffusivity

$$
\alpha_g=\frac{k_g}{\rho_gc_{p,g}}
$$

This is needed for thermal-regime scaling, Péclet number, gas preheat length, and thin/thick transition. It requires mixture-property lookup or a property model based on atmosphere composition and temperature.[^3]

***

## Flow and gravity equations

Your signed flow field is unusually important: the data use negative values for counterflow/opposed flow and positive values for coflow. Preserve the sign; do not feed only $|U|$ into a universal model. [^1]

### Flow-direction indicators

$$
U_{\mathrm{opp}}=\max(-U,0)
$$

$$
U_{\mathrm{co}}=\max(U,0)
$$

$$
I_{\mathrm{opp}}=
\begin{cases}
1,& U<0\\
0,& U\ge0
\end{cases}
$$

$$
I_{\mathrm{co}}=
\begin{cases}
1,& U>0\\
0,& U\le0
\end{cases}
$$

Use $U_{\mathrm{opp}}$ in classical opposed-flow thermal equations. Model coflow separately or include a geometry/regime interaction, because its transport behavior is not equivalent to opposed flow.

### Effective relative flow

For a moving flame in opposed flow:

$$
U_r=U_{\mathrm{opp}}-V_f
$$

During prediction, $V_f$ is unknown, so use an iterative calculation:

$$
U_r^{(n)}=U_{\mathrm{opp}}-\widehat{V}_f^{(n-1)}
$$

or use the measured $V_f$ only for post-hoc mechanistic analysis rather than as an input feature. Takahashi et al. identify relative velocity as central to low-flow microgravity spread.[^6]

### Natural-convection speed scale

$$
U_{NC}
\sim
\left[
\frac{
g\,\alpha_g\,(T_g-T_\infty)
}
{T_\infty}
\right]^{1/3}
$$

This provides a gravity-derived flow feature when $g>0$. It is not usable at $g=0$, where the expression correctly tends toward zero.[^3]

### Effective opposed velocity

For a pragmatic cross-gravity model:

$$
U_{\mathrm{eff}}
=
U_{\mathrm{opp}}+
C_bU_{NC}
$$

where $C_b$ is learned from data or calibrated by material/geometry class.

This is a **hybrid model feature**, not a universally validated closed law. It lets the model use imposed counterflow and buoyancy in one velocity scale without forcing a parabolic-flight case and a 1g case to share an identical raw-flow interpretation.[^2][^3]

### Reynolds number

For flat samples use an axial length $L$; for wires/cylinders use outer diameter $d_o$:

$$
Re_L
=
\frac{\rho_g|U|L}{\mu_g}
$$

$$
Re_d
=
\frac{\rho_g|U|d_o}{\mu_g}
$$

### Péclet numbers

$$
Pe_g=\frac{|U|L}{\alpha_g}
$$

$$
Pe_{O_2}=\frac{|U|L}{D_{O_2,\mathrm{mix}}}
$$

These are better transfer features than raw $U$ alone when the database spans pressure, oxygen composition, gas type, and scale.

### Gravity-dependent dimensionless groups

$$
Gr_L
=
\frac{
g\beta(T_f-T_\infty)L^3
}
{\nu_g^2}
$$

$$
Ra_L
=
Gr_LPr
$$

- $\beta\approx1/T_\infty$ for ideal gases
- $\nu_g=\mu_g/\rho_g$
- $Pr=\nu_g/\alpha_g$

Use them only after establishing mixture properties and a defensible reference temperature. They are especially appropriate for rows with $g>0$, but their literal values should not be overinterpreted in partial gravity or small enclosures.

***

## Flat-sample equations

Apply these only to rows with `Geometry of Sample = Flat`.

### Flat geometric variables

For a rectangle with length $L$, width $W$, total thickness $h$:

$$
\delta=\frac{h}{2}
$$

$$
A_s=LW
$$

$$
AR=\frac{L}{W}
$$

$$
\chi_{\delta}=\frac{\delta}{L}
$$

Use the length aligned with spread/flow. Your current dimensions field stores flat samples as $L\times W\times H$, so all four variables can be parsed.[^1]

### Gas preheat length

$$
L_g=\frac{\alpha_g}{U_{\mathrm{eff}}}
$$

**Use only when** $U_{\mathrm{eff}}>0$. In nominal microgravity and near-zero flow, the predicted length diverges; cap it at an enclosure/available-specimen length:

$$
L_{g,\mathrm{cap}}
=
\min
\left(
\frac{\alpha_g}{\max(U_{\mathrm{eff}},U_{\min})},
L_{\mathrm{avail}}
\right)
$$

where $L_{\mathrm{avail}}$ may be the specimen length or a chamber length. This matters because your database includes near-zero gravity and low-flow cases.[^3][^1]

### Classical thermally thin FSR

$$
V_{f,\mathrm{thin}}
=
\frac{\alpha_g}{\rho_sc_s\delta}
\left(
\frac{T_f-T_v}{T_v-T_\infty}
\right)
$$

Implement it as a **physics-derived feature** or baseline prediction only if you can append material properties $\rho_s$, $c_s$, and a pyrolysis/vaporization temperature $T_v$. It is appropriate for thermally thin, conduction-dominated, opposed-flow spread; it does not include radiation, ignition, finite-rate chemistry, or geometry effects.[^3]

An equivalent reported form is:

$$
V_{f,\mathrm{thin}}
=
\frac{k_g}{\rho_sc_s\delta}
\left(
\frac{T_f-T_v}{T_v-T_\infty}
\right)
$$

The apparent difference reflects choices of characteristic transport quantities and notation in the referenced thermal theory. Keep one internally consistent implementation rather than mixing forms.[^6][^3]

### Classical thermally thick FSR

$$
V_{f,\mathrm{thick}}
=
\frac{\alpha_g}{c_{p,g}}
\left(
\frac{\alpha_g}
{\rho_sc_s\alpha_s}
\right)^{1/2}
U_{\mathrm{eff}}
\left(
\frac{T_f-T_v}
{T_v-T_\infty}
\right)^2
$$

Use this for flat thick fuels in the conduction-dominated opposed-flow regime. It encodes the key thick-fuel behavior:

$$
V_{f,\mathrm{thick}}\propto U_{\mathrm{eff}}
$$

It is not appropriate in the quiescent microgravity radiation-dominated regime.[^3]

### Thin/thick regime indicator

$$
\delta_{\mathrm{cr}}
=
\left(
\frac{\alpha_s\alpha_g}
{c_{p,g}U_{\mathrm{eff}}}
\right)^{1/2}
\left(
\frac{T_f-T_v}{T_v-T_\infty}
\right)
$$

$$
\Pi_{\mathrm{th}}
=
\frac{\delta}{\delta_{\mathrm{cr}}}
$$

Interpretation:

$$
\Pi_{\mathrm{th}}<1
\Rightarrow
\text{thermally thin}
$$

$$
\Pi_{\mathrm{th}}>1
\Rightarrow
\text{thermally thick}
$$

This is a powerful implementable regime feature, but only away from $U_{\mathrm{eff}}\approx0$. Use an “indeterminate/radiation candidate” label rather than forcing a thin/thick decision at very low effective velocity.[^3]

### Smooth thin-thick blend

Instead of an abrupt switch:

$$
w_{\mathrm{thin}}
=
\frac{1}
{1+\exp[a(\ln\Pi_{\mathrm{th}}-b)]}
$$

$$
\widehat V_{f,\mathrm{thermal}}
=
w_{\mathrm{thin}}V_{f,\mathrm{thin}}
+
(1-w_{\mathrm{thin}})V_{f,\mathrm{thick}}
$$

- $a$: learned transition sharpness
- $b$: learned transition location, typically initialized near zero

This is a recommended implementation formulation, not a direct published correlation. It is more stable for your heterogeneous data than hard thresholding.

***

## Radiation-controlled equations

These matter in your dataset at microgravity/partial gravity and low relative flow, where thermal-regime models alone may fail.

### Radiation loss from a flat fuel surface

$$
q_{\mathrm{rad,loss}}
=
\varepsilon_s\sigma
\left(
T_s^4-T_\infty^4
\right)
$$

- $\varepsilon_s$: solid emissivity
- $\sigma$: Stefan–Boltzmann constant
- $T_s$: surface temperature

Your current database does not show surface temperature or emissivity columns, so this requires material-property augmentation or fitted material embeddings.

### Radiation-loss parameter

$$
R_{\mathrm{rad}}
=
\frac{
\varepsilon_s\sigma(T_v^4-T_\infty^4)
}
{
k_g(T_f-T_v)
}
\frac{\alpha_g}{V_{f,\mathrm{thin}}}
$$

### Radiation-corrected thin-fuel equation

$$
\gamma+\frac{R_{\mathrm{rad}}}{\gamma}=1
$$

$$
\gamma=\frac{V_f}{V_{f,\mathrm{thin}}}
$$

Rearranged:

$$
\gamma^2-\gamma+R_{\mathrm{rad}}=0
$$

$$
\gamma
=
\frac{
1+\sqrt{1-4R_{\mathrm{rad}}}
}{2}
$$

Use the physically larger root when it exists. If:

$$
R_{\mathrm{rad}}>\frac{1}{4}
$$

the simple steady radiation-corrected model has no real solution, which should be interpreted as a candidate for extinction or strongly unsteady behavior—not as a numerical error. This is especially useful for your `Ignition = No` / near-extinction records.[^6]

### Gas-phase radiation plus conduction

$$
q_{\mathrm{feedback}}
=
K d_g
+
\frac{k_g(T_f-T_v)}{d_g}
$$

$$
K
=
4\sigma a_P
(T_f^4-T_v^4)
$$

- $d_g$: flame thickness
- $a_P$: Planck-mean absorption coefficient

This form is suitable only after you add the identity/concentration of radiatively participating species, especially carbon dioxide and water. It cannot be reliably computed from oxygen concentration alone.[^4]

### Radiation-supported thick-fuel rate

$$
V_{f,\mathrm{rad,thick}}
=
\left[
\frac{
K\alpha_g
}
{
\rho_sc_{p,s}(T_v-T_\infty)
}
\right]^{1/2}
$$

Use this as a **candidate baseline feature** for thick fuels at small $U_{\mathrm{eff}}$, particularly if the atmosphere includes radiatively active gases. It is not an appropriate default for nitrogen or helium dilution.[^4]

***

## Wire equations

Apply these only to rows with `Geometry of Sample = Wire`. Your database includes core material, core diameter, outer diameter, LDPE insulation, oxygen, pressure, signed flow, partial gravity, ignition outcome, and FSR—enough to build a useful wire-specific mechanistic feature set.[^1]

### Wire geometry

$$
r_c=\frac{d_c}{2}
$$

$$
r_o=\frac{d_o}{2}
$$

$$
\delta_{\mathrm{ins}}=r_o-r_c
$$

$$
\Lambda=\frac{r_c}{r_o}
$$

$$
A_{\mathrm{ins}}
=
\pi(r_o^2-r_c^2)
$$

$$
A_{\mathrm{core}}=\pi r_c^2
$$

$$
\frac{A_{\mathrm{core}}}{A_{\mathrm{ins}}}
=
\frac{r_c^2}{r_o^2-r_c^2}
$$

- $\Lambda$: core-to-outer-radius ratio
- $A_{\mathrm{core}}/A_{\mathrm{ins}}$: axial heat-sink/conduction feature

This matters because your data include both Cu and NiCr cores and multiple outer diameters. Core composition and core area should not be represented only as a categorical material name; append $k_c$, $\rho_c$, and $c_{p,c}$ by core material.[^1]

### Wire external area per unit length

$$
A'_{\mathrm{surf}}
=
2\pi r_o
$$

### Insulation volume per unit length

$$
V'_{\mathrm{ins}}
=
\pi(r_o^2-r_c^2)
$$

### Surface-to-volume ratio

$$
\frac{A'_{\mathrm{surf}}}{V'_{\mathrm{ins}}}
=
\frac{2r_o}{r_o^2-r_c^2}
$$

This is an excellent directly implementable geometric feature. It rises for thinner insulation and captures why nominal “outer diameter” is insufficient.

### Cylindrical conductive heat feedback

$$
Q'_{\mathrm{cond}}
=
\frac{
2\pi k_g(T_f-T_s)
}
{
\ln(d_s/r_o)
}
$$

- $Q'_{\mathrm{cond}}$: heat transfer per unit flame/preheat length
- $d_s$: flame stand-off radius/distance
- $r_o$: outer wire radius

The original cylindrical expression is written with a flame stand-off distance and gas preheat length; per-unit-length form avoids requiring $L_g$ explicitly. You cannot fully evaluate it unless flame stand-off distance is measured or modeled.[^5]

### Wire radiative loss

$$
Q'_{\mathrm{rad,loss}}
=
2\pi r_o\varepsilon_s\sigma
(T_s^4-T_\infty^4)
$$

### Net wire preheating

$$
Q'_{\mathrm{net}}
=
Q'_{\mathrm{cond}}
-
Q'_{\mathrm{rad,loss}}
$$

This is the correct physical structure for a wire energy feature, although $T_s$, $T_f$, $\varepsilon_s$, and $d_s$ must be supplied through measurements, material properties, simplified assumptions, or a latent model.[^5]

### Convective oxygen supply

$$
\dot m_{O_2,c}'
=
2\pi r_od_s
\rho_g
U_r
Y_{O_2}
$$

### Diffusive oxygen supply

$$
\dot m_{O_2,d}'
=
2\pi d_s L_g
\rho_gD_{O_2,\mathrm{mix}}
Y_{O_2}
$$

The original paper’s diffusion expression is geometry-specific; for implementation, treat this as a scaling feature rather than an exact rate unless $d_s$ and $L_g$ are available.[^5]

### Oxygen-supply proxy without unmeasured stand-off distance

Because $d_s$ is not in your database, use measurable proxies:

$$
\Phi_{O_2,\mathrm{conv}}
=
\rho_gU_{\mathrm{opp}}Y_{O_2}
$$

$$
\Phi_{O_2,\mathrm{diff}}
=
\frac{\rho_gD_{O_2,\mathrm{mix}}Y_{O_2}}{r_o}
$$

$$
\Phi_{O_2,\mathrm{total}}
=
C_c\Phi_{O_2,\mathrm{conv}}
+
C_d\Phi_{O_2,\mathrm{diff}}
$$

Learn $C_c$ and $C_d$ from the wire subset. These are implementable and preserve the convection-versus-diffusion structure without inventing a flame stand-off measurement.

### Core axial heat-conduction feature

$$
G'_{\mathrm{core}}
=
\frac{k_cA_{\mathrm{core}}}{L_{\mathrm{ref}}}
$$

- $k_c$: core thermal conductivity
- $L_{\mathrm{ref}}$: reference preheat length, such as $r_o$, specimen length, or a capped $L_g$

A dimensionless alternative:

$$
\Pi_{\mathrm{core}}
=
\frac{k_cA_{\mathrm{core}}}
{k_g(2\pi r_o)L_{\mathrm{ref}}}
$$

This is not a directly published FSR correlation, but it is physically consistent and necessary because your database varies core material between copper and NiCr. Copper’s axial thermal conduction can change the available heat reaching unburned insulation.[^5][^1]

***

## Cylindrical-sample equations

For a solid rod or tube, use a distinct geometric branch. The database labels these as cylindrical and records dimensions as radius times length.[^1]

### Cylinder geometry

$$
A'_{\mathrm{surf}}
=
2\pi r_s
$$

$$
V'_{\mathrm{solid}}
=
\pi r_s^2
$$

$$
\frac{A'_{\mathrm{surf}}}{V'_{\mathrm{solid}}}
=
\frac{2}{r_s}
$$

### Curvature ratio

$$
\Pi_{\mathrm{curv}}
=
\frac{L_g}{r_s}
$$

### Cylinder Reynolds number

$$
Re_d
=
\frac{\rho_g|U|2r_s}{\mu_g}
$$

### Cylindrical oxygen-diffusion feature

$$
\Phi_{O_2,\mathrm{diff,cyl}}
=
\frac{\rho_gD_{O_2,\mathrm{mix}}Y_{O_2}}{r_s}
$$

These variables capture the increased importance of curvature and radial diffusion for small-diameter samples. They should be used with a cylinder-specific model or geometry interactions rather than pooled blindly with planar sheets.[^5][^1]

***

## Ignition classifier equations

Your dataset has a binary `Ignition` output. This enables a separate flammability/sustained-spread boundary model.

### Logistic ignition model

$$
p_{\mathrm{ign}}
=
\frac{1}
{
1+\exp(-z)
}
$$

$$
z=
\beta_0
+
\beta_1p_{O_2}
+
\beta_2\ln P
+
\beta_3U_{\mathrm{opp}}
+
\beta_4U_{\mathrm{co}}
+
\beta_5g
+
\beta_6E_{\mathrm{ign}}
+
\beta_7\Pi_{\mathrm{geom}}
+
\beta_8\Pi_{\mathrm{core}}
+
\beta_9\mathrm{Material}
+
\beta_{10}\mathrm{Facility}
$$

This equation is a recommended implementation, not an equation claimed by the papers. It maps directly to your available fields and accommodates the observed dataset pattern: lower oxygen cases switch from sustained ignition to no ignition under otherwise similar flow and gravity conditions.[^1]

### Physics-guided ignition threshold

A more interpretable alternative is:

$$
\mathcal{M}_{\mathrm{ign}}
=
\frac{
\dot q_{\mathrm{feedback}}
+
\dot q_{\mathrm{ign,eff}}
}
{
\dot q_{\mathrm{loss}}
+
\dot q_{\mathrm{pyrolysis,req}}
}
$$

$$
\mathrm{Ignition}
=
\begin{cases}
1,& \mathcal{M}_{\mathrm{ign}}>1\\
0,& \mathcal{M}_{\mathrm{ign}}\le1
\end{cases}
$$

This is a physical template. Your present fields allow the ignition-energy part, geometry, oxygen, flow, and gravity terms, but not a full evaluation until material thermal properties and flame/surface temperature assumptions are added.

***

## Recommended implementable model

The most defensible model for the database as it currently stands is a **mixture-of-regimes model**.

### Stage 1: predict sustained ignition

$$
\widehat p_{\mathrm{ign}}
=
f_{\mathrm{class}}
\left(
p_{O_2},
P,
U_{\mathrm{opp}},
U_{\mathrm{co}},
g,
E_{\mathrm{ign}},
\text{geometry},
\text{dimensions},
\text{material},
\text{core material},
\text{facility}
\right)
$$

### Stage 2: select a geometry branch

$$
\widehat V_{f\mid\mathrm{ign}}
=
\begin{cases}
f_{\mathrm{flat}}(\mathbf{x}),& \text{flat}\\
f_{\mathrm{wire}}(\mathbf{x}),& \text{wire}\\
f_{\mathrm{cyl}}(\mathbf{x}),& \text{cylindrical}
\end{cases}
$$

### Flat branch

$$
\widehat V_{f,\mathrm{flat}}
=
w_{\mathrm{thin}}
V_{f,\mathrm{thin}}
+
(1-w_{\mathrm{thin}})
V_{f,\mathrm{thick}}
+
\Delta V_{\mathrm{rad}}
$$

where $w_{\mathrm{thin}}$ is determined by $\Pi_{\mathrm{th}}$, and $\Delta V_{\mathrm{rad}}$ is learned as a low-$U_{\mathrm{eff}}$, low-$g$ correction rather than assumed to follow a universal radiation law.

### Wire branch

$$
\widehat V_{f,\mathrm{wire}}
=
f_{\mathrm{wire}}
\left(
p_{O_2},
P,
U_{\mathrm{opp}},
g,
r_o,
r_c,
\delta_{\mathrm{ins}},
\frac{A'_{\mathrm{surf}}}{V'_{\mathrm{ins}}},
\Pi_{\mathrm{core}},
\Phi_{O_2,\mathrm{conv}},
\Phi_{O_2,\mathrm{diff}},
\text{core material}
\right)
$$

### All-case prediction

$$
\widehat V_{f,\mathrm{all}}
=
\widehat p_{\mathrm{ign}}
\widehat V_{f\mid\mathrm{ign}}
$$

This formulation uses all the major fields currently in the spreadsheet while avoiding invalid use of planar thermal theory for wires and cylinders.[^6][^3][^5][^1]

***

## Features to add

Your present table is already useful, but several missing features block direct evaluation of the strongest physical equations.


| Add / standardize | Why it is needed |
| :-- | :-- |
| Diluent gas composition | Needed for $\overline{M}$, $k_g$, $c_{p,g}$, $D_{O_2}$, radiation, Lewis number |
| Initial gas temperature | Needed for density, transport properties, buoyancy, radiative loss |
| Material density $\rho_s$ | Needed for all thermal FSR closures |
| Material $c_{p,s}$, $k_s$, $\alpha_s$ | Needed for thin/thick classification and heat balance |
| Pyrolysis temperature $T_v$ | Needed for thermal and radiation closures |
| Emissivity $\varepsilon_s$ | Needed for radiation-loss equations |
| Core $k_c,\rho_c,c_{p,c}$ | Needed for wire axial heat-sink effect |
| Flow direction relative to flame | More precise than signed apparatus flow alone |
| Flame stand-off distance | Needed for direct wire oxygen/heat-transfer equations |
| Initial/steady surface and flame temperature | Needed for radiative and buoyancy models |
| Specimen orientation | Essential for gravity effects, especially “downward” versus “upward” spread |
| Time window used for FSR | Distinguishes transient FSR from quasi-steady FSR |
| Diluent CO$_2$/H$_2$O relevance | Needed to activate gas-radiation features |

The database currently includes chamber geometry and internal dimensions; preserve them as experimental random-effect or domain-shift variables, rather than assuming they are fundamental material properties. Enclosure size can strongly affect oxygen replenishment and radiative boundary conditions.[^7][^1]

***

## Avoid these errors

- Do **not** fit a single polynomial to all flat, wire, and cylindrical cases.
- Do **not** treat `No ignition` as an FSR of zero in a plain regression; use a classifier or zero-inflated/hurdle structure.
- Do **not** calculate $L_g=\alpha_g/U$ without handling $U\rightarrow0$; cap it by specimen/chamber length.
- Do **not** use $U_{\mathrm{opp}}-V_f$ as a predictive input unless it is solved iteratively; otherwise it leaks the target.
- Do **not** calculate oxygen flux using oxygen concentration alone; convert to $Y_{O_2}$ and include density/diffusivity when atmosphere composition is known.
- Do **not** activate gas-radiation equations for all inert gases; CO$_2$ and water-vapor participation is fundamentally different from nitrogen and helium.[^4][^5]

<div align="center">⁂</div>

[^1]: Microgravity_Database_reduced.xlsx

[^2]: 1976_Andracchio_droptower_TechReport.pdf

[^3]: 2016_Bhattacharjee_ISS_FSJ.pdf

[^4]: 2002_Son_droptower_PROCI.pdf

[^5]: 2002_Fujita_droptower_PROCI.pdf

[^6]: 2002_Takahashi_droptower_PROCI.pdf

[^7]: 1996_Kashiwagi_droptower_CF.pdf

