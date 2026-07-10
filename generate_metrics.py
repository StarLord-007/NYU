#!/usr/bin/env python3
"""
Generate a comprehensive set of metric plots/charts from
Microgravity_Database_reduced.csv into the ./metrics folder.

Metrics covered:
- Number of papers, points per paper, paper sizes
- Ignition/extinction composition (overall, per paper, per facility, per geometry)
- Ranges & gaps of: fuel density, fuel conductivity, fuel cp, oxygen,
  pressure, gravity, flow velocity, FSR
- O2 vs Pressure per point (colored by ignition, gravity, diluent)
- FSR relationships (vs O2, pressure, gravity, fuel density)
- Categorical compositions (diluent, facility, geometry, ignition method)
- Data completeness / missingness
- Correlation heatmap
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COL = {
    "paper": "Article (MLA)",
    "geom": "Geometry of Sample (Flat, wire, or Cylindrical)",
    "fuel_rho": "fuel_density_kg_m3",
    "fuel_k": "fuel_k_W_mK",
    "fuel_cp": "fuel_cp_J_kgK",
    "o2": "Oxygen Concentration",
    "diluent": "diluent",
    "pressure": "Pressure",
    "flow": "Flow Velocity (Co flow is + and counter flow is -)",
    "gravity": "Gravity (g/gearth)",
    "facility": "Expireimental facility (Parabolic Aircraft, Drop Tower, Spacecraft, Sounding Rocket, Ground)",
    "ign_method": "Ignition method (Wire, open flame, or Radiative Heater",
    "ignition": "Ignition (Yes/No)",
    "fsr": "FSR (Flame Spread Rate)",
    "flame_len": "Flame Length",
    "hrr": "HRR (Heat release rate)",
    "gas_k": "gas_k",
    "gas_rho": "gas_density_kg_m3",
}

# ---------------------------------------------------------------- load
df = pd.read_csv("Microgravity_Database_reduced.csv", header=1, encoding="latin-1")
df = df.dropna(how="all")

# clean ignition label
df["ign_clean"] = df[COL["ignition"]].astype(str).str.strip().str.title()
df.loc[~df["ign_clean"].isin(["Yes", "No"]), "ign_clean"] = np.nan

# clean geometry
df["geom_clean"] = df[COL["geom"]].astype(str).str.strip().str.title()
df.loc[df["geom_clean"].isin(["Nan", ""]), "geom_clean"] = np.nan

# numeric FSR
df["fsr_num"] = pd.to_numeric(df[COL["fsr"]], errors="coerce")
df["flame_len_num"] = pd.to_numeric(df[COL["flame_len"]], errors="coerce")

# short paper id (first author + index)
first_author = df[COL["paper"]].astype(str).str.split(",").str[0].str.strip()
codes, uniques = pd.factorize(df[COL["paper"]])
df["paper_id"] = ["P%02d %s" % (c + 1, a[:18]) for c, a in zip(codes, first_author)]

IGN_COLORS = {"Yes": "#2ca02c", "No": "#d62728"}

saved = []


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved.append(name)
    print("saved", name)


# ================================================================
# 1. DATASET OVERVIEW SUMMARY CARD
# ================================================================
n_points = len(df)
n_papers = df[COL["paper"]].nunique()
n_ign = (df["ign_clean"] == "Yes").sum()
n_ext = (df["ign_clean"] == "No").sum()
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")
stats = [
    ("Total data points", f"{n_points:,}"),
    ("Number of papers", f"{n_papers}"),
    ("Median points / paper", f"{df.groupby(COL['paper']).size().median():.0f}"),
    ("Ignition (Yes)", f"{n_ign:,} ({100*n_ign/n_points:.1f}%)"),
    ("Extinction / No-ignition (No)", f"{n_ext:,} ({100*n_ext/n_points:.1f}%)"),
    ("O2 mole-fraction range", f"{df[COL['o2']].min():.3f} - {df[COL['o2']].max():.3f}"),
    ("Pressure range (kPa)", f"{df[COL['pressure']].min():.1f} - {df[COL['pressure']].max():.0f}"),
    ("Gravity range (g/g_earth)", f"{df[COL['gravity']].min():.2f} - {df[COL['gravity']].max():.2f}"),
    ("Fuel density range (kg/m3)", f"{df[COL['fuel_rho']].min():.0f} - {df[COL['fuel_rho']].max():.0f}"),
    ("Fuel conductivity range (W/mK)", f"{df[COL['fuel_k']].min():.3f} - {df[COL['fuel_k']].max():.2f}"),
    ("Points with numeric FSR", f"{df['fsr_num'].notna().sum():,}"),
    ("FSR range (mm/s)", f"{df['fsr_num'].min():.2f} - {df['fsr_num'].max():.0f}"),
]
for i, (k, v) in enumerate(stats):
    r, c = i % 6, i // 6
    ax.text(0.02 + c * 0.5, 0.92 - r * 0.16, k, fontsize=11, weight="bold", va="top")
    ax.text(0.02 + c * 0.5, 0.86 - r * 0.16, v, fontsize=11, color="#1f77b4", va="top")
ax.set_title("Dataset Overview - Microgravity Flammability Database (reduced)", fontsize=14)
save(fig, "01_dataset_overview_summary.png")

# ================================================================
# 2. POINTS PER PAPER (all papers, horizontal bar)
# ================================================================
pp = df.groupby("paper_id").size().sort_values()
fig, ax = plt.subplots(figsize=(10, 16))
pp.plot.barh(ax=ax, color="#1f77b4")
ax.set_xlabel("Number of data points")
ax.set_ylabel("Paper")
ax.set_title(f"Number of Data Points per Paper (all {n_papers} papers)")
ax.tick_params(axis="y", labelsize=7)
save(fig, "02_points_per_paper_all.png")

# 3. top 25 papers
fig, ax = plt.subplots(figsize=(10, 8))
pp.tail(25).plot.barh(ax=ax, color="#ff7f0e")
ax.set_xlabel("Number of data points")
ax.set_title("Top 25 Papers by Number of Data Points")
save(fig, "03_points_per_paper_top25.png")

# 4. histogram of paper sizes
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(pp.values, bins=30, color="#2ca02c", edgecolor="k")
ax.axvline(pp.median(), color="r", ls="--", label=f"median = {pp.median():.0f}")
ax.axvline(pp.mean(), color="b", ls="--", label=f"mean = {pp.mean():.1f}")
ax.set_xlabel("Points in paper")
ax.set_ylabel("Number of papers")
ax.set_title("Distribution of Paper Sizes (points per paper)")
ax.legend()
save(fig, "04_paper_size_histogram.png")

# 5. cumulative share of data by paper (Pareto)
fig, ax = plt.subplots(figsize=(9, 5))
cum = pp.sort_values(ascending=False).cumsum() / n_points * 100
ax.plot(range(1, len(cum) + 1), cum.values, marker="o", ms=3)
ax.axhline(80, color="r", ls="--", alpha=0.6, label="80% of data")
n80 = int((cum.values < 80).sum()) + 1
ax.axvline(n80, color="r", ls=":", alpha=0.6, label=f"{n80} papers -> 80%")
ax.set_xlabel("Number of papers (largest first)")
ax.set_ylabel("Cumulative % of all data points")
ax.set_title("Pareto Curve - Cumulative Data Share by Paper")
ax.legend()
save(fig, "05_pareto_cumulative_data_by_paper.png")

# ================================================================
# 6. IGNITION / EXTINCTION COMPOSITION (overall pie)
# ================================================================
fig, ax = plt.subplots(figsize=(7, 6))
counts = df["ign_clean"].value_counts(dropna=False)
labels = [("Unknown" if pd.isna(k) else ("Ignition (Yes)" if k == "Yes" else "No ignition / Extinction"))
          for k in counts.index]
colors = ["#2ca02c" if k == "Yes" else "#d62728" if k == "No" else "#7f7f7f" for k in counts.index]
ax.pie(counts.values, labels=[f"{l}\n{v:,} ({100*v/n_points:.1f}%)" for l, v in zip(labels, counts.values)],
       colors=colors, startangle=90, wedgeprops=dict(edgecolor="w"))
ax.set_title("Overall Composition: Ignition vs Extinction (No-Ignition)")
save(fig, "06_ignition_composition_pie.png")

# 7. ignition composition per paper (stacked, top 30 papers)
top30 = pp.tail(30).index
ct = pd.crosstab(df["paper_id"], df["ign_clean"]).reindex(top30).fillna(0)
fig, ax = plt.subplots(figsize=(10, 10))
left = np.zeros(len(ct))
for lab, col in [("Yes", "#2ca02c"), ("No", "#d62728")]:
    if lab in ct:
        ax.barh(ct.index, ct[lab], left=left, color=col, label=f"Ignition={lab}")
        left += ct[lab].values
ax.set_xlabel("Data points")
ax.set_title("Ignition vs Extinction Composition per Paper (top 30 papers)")
ax.legend()
ax.tick_params(axis="y", labelsize=7)
save(fig, "07_ignition_composition_per_paper.png")

# 8. ignition fraction per paper histogram
frac = df.groupby(COL["paper"])["ign_clean"].apply(lambda s: (s == "Yes").mean())
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(frac, bins=20, color="#9467bd", edgecolor="k")
ax.set_xlabel("Fraction of points with Ignition = Yes")
ax.set_ylabel("Number of papers")
ax.set_title("Distribution of Ignition Fraction Across Papers")
save(fig, "08_ignition_fraction_per_paper_hist.png")

# ================================================================
# CATEGORICAL COMPOSITIONS
# ================================================================
def cat_bar(series, title, fname, color="#1f77b4"):
    fig, ax = plt.subplots(figsize=(9, 5))
    vc = series.fillna("Unknown").value_counts()
    vc.plot.bar(ax=ax, color=color, edgecolor="k")
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Data points")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    save(fig, fname)

cat_bar(df[COL["facility"]], "Data Points by Experimental Facility", "09_facility_composition.png", "#17becf")
cat_bar(df["geom_clean"], "Data Points by Sample Geometry", "10_geometry_composition.png", "#bcbd22")
cat_bar(df[COL["diluent"]], "Data Points by Diluent Gas", "11_diluent_composition.png", "#e377c2")
cat_bar(df[COL["ign_method"]], "Data Points by Ignition Method", "12_ignition_method_composition.png", "#8c564b")

# 13. facility vs ignition stacked
ct = pd.crosstab(df[COL["facility"]].fillna("Unknown"), df["ign_clean"])
fig, ax = plt.subplots(figsize=(9, 5))
ct[["Yes", "No"]].plot.bar(stacked=True, ax=ax, color=["#2ca02c", "#d62728"], edgecolor="k")
ax.set_ylabel("Data points")
ax.set_title("Ignition Outcome by Experimental Facility")
plt.xticks(rotation=30, ha="right")
ax.legend(title="Ignition")
save(fig, "13_ignition_by_facility.png")

# 14. geometry vs ignition stacked (fraction)
ct = pd.crosstab(df["geom_clean"], df["ign_clean"], normalize="index") * 100
fig, ax = plt.subplots(figsize=(9, 5))
ct[["Yes", "No"]].plot.bar(stacked=True, ax=ax, color=["#2ca02c", "#d62728"], edgecolor="k")
ax.set_ylabel("% of points")
ax.set_title("Ignition Outcome Share by Sample Geometry (%)")
plt.xticks(rotation=0)
ax.legend(title="Ignition", loc="lower right")
save(fig, "14_ignition_share_by_geometry.png")

# ================================================================
# RANGE + GAP ANALYSIS helper
# ================================================================
def range_and_gaps(vals, label, unit, fname_prefix, log=False, bins=40, top_gaps=8):
    v = pd.to_numeric(vals, errors="coerce").dropna()
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
    u = np.sort(v.unique())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={"height_ratios": [3, 1.3]})
    # histogram
    if log:
        b = np.logspace(np.log10(u.min()), np.log10(u.max()), bins)
        ax1.set_xscale("log")
    else:
        b = bins
    ax1.hist(v, bins=b, color="#1f77b4", edgecolor="k", alpha=0.85)
    ax1.set_ylabel("Data points")
    ax1.set_title(f"Range of {label}: {u.min():.4g} to {u.max():.4g} {unit}  "
                  f"({len(u)} unique values, n={len(v):,})")

    # coverage strip / gaps
    ax2.eventplot(u, orientation="horizontal", colors="#1f77b4", lineoffsets=0.5,
                  linelengths=0.8, linewidths=0.8)
    if log:
        ax2.set_xscale("log")
        gaps = np.diff(np.log10(u))
    else:
        gaps = np.diff(u)
    if len(gaps):
        idx = np.argsort(gaps)[::-1][:top_gaps]
        for i in idx:
            lo, hi = u[i], u[i + 1]
            ax2.axvspan(lo, hi, color="red", alpha=0.25)
        # annotate largest gap
        i0 = idx[0]
        ax2.text(np.sqrt(u[i0] * u[i0 + 1]) if log else (u[i0] + u[i0 + 1]) / 2, 1.05,
                 f"largest gap:\n{u[i0]:.4g} - {u[i0+1]:.4g}", ha="center",
                 fontsize=8, color="darkred")
    ax2.set_ylim(0, 1.4)
    ax2.set_yticks([])
    ax2.set_xlabel(f"{label} {unit}")
    ax2.set_title(f"Coverage Strip and Largest {top_gaps} Gaps in {label} (red = untested gaps)",
                  fontsize=10)
    fig.tight_layout()
    save(fig, f"{fname_prefix}.png")


range_and_gaps(df[COL["fuel_rho"]], "Fuel Density", "[kg/m3]",
               "15_fuel_density_range_gaps", log=False)
range_and_gaps(df[COL["fuel_k"]], "Fuel Thermal Conductivity", "[W/m-K]",
               "16_fuel_conductivity_range_gaps", log=True)
range_and_gaps(df[COL["fuel_cp"]], "Fuel Specific Heat cp", "[J/kg-K]",
               "17_fuel_cp_range_gaps", log=False)
range_and_gaps(df[COL["o2"]], "Oxygen Concentration", "[mole fraction]",
               "18_oxygen_range_gaps", log=False)
range_and_gaps(df[COL["pressure"]], "Pressure", "[kPa]",
               "19_pressure_range_gaps", log=True)
range_and_gaps(df[COL["gravity"]], "Gravity Level", "[g/g_earth]",
               "20_gravity_range_gaps", log=False)
range_and_gaps(df[COL["flow"]], "Flow Velocity", "[cm/s] (+co-flow / -counter-flow)",
               "21_flow_velocity_range_gaps", log=False)
range_and_gaps(df["fsr_num"], "Flame Spread Rate (FSR)", "[mm/s]",
               "22_fsr_range_gaps", log=True)

# ================================================================
# GRAVITY REGIMES
# ================================================================
g = df[COL["gravity"]]
regime = pd.cut(g, bins=[-0.001, 0.01, 0.9, 1.1, 100],
                labels=["Microgravity (<=0.01g)", "Partial g (0.01-0.9g)",
                        "Normal g (~1g)", "Hypergravity (>1.1g)"])
fig, ax = plt.subplots(figsize=(9, 5))
vc = regime.value_counts().reindex(["Microgravity (<=0.01g)", "Partial g (0.01-0.9g)",
                                    "Normal g (~1g)", "Hypergravity (>1.1g)"])
vc.plot.bar(ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], edgecolor="k")
for i, v in enumerate(vc.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
ax.set_ylabel("Data points")
ax.set_title("Data Points by Gravity Regime")
plt.xticks(rotation=15)
save(fig, "23_gravity_regime_composition.png")

# 24. ignition share by gravity regime
ct = pd.crosstab(regime, df["ign_clean"], normalize="index") * 100
fig, ax = plt.subplots(figsize=(9, 5))
ct[["Yes", "No"]].plot.bar(stacked=True, ax=ax, color=["#2ca02c", "#d62728"], edgecolor="k")
ax.set_ylabel("% of points")
ax.set_title("Ignition Outcome Share by Gravity Regime (%)")
plt.xticks(rotation=15)
ax.legend(title="Ignition", loc="lower right")
save(fig, "24_ignition_share_by_gravity_regime.png")

# ================================================================
# O2 vs PRESSURE per point
# ================================================================
m = df[COL["o2"]].notna() & df[COL["pressure"]].notna()
sub = df[m]

# 25. colored by ignition
fig, ax = plt.subplots(figsize=(10, 7))
for lab, col in IGN_COLORS.items():
    s = sub[sub["ign_clean"] == lab]
    ax.scatter(s[COL["o2"]], s[COL["pressure"]], s=12, alpha=0.45, c=col,
               label=f"Ignition={lab} (n={len(s):,})", edgecolors="none")
ax.set_yscale("log")
ax.set_xlabel("Oxygen Concentration [mole fraction]")
ax.set_ylabel("Pressure [kPa] (log scale)")
ax.set_title("O2 vs Pressure per Data Point, Colored by Ignition Outcome")
ax.legend()
save(fig, "25_o2_vs_pressure_by_ignition.png")

# 26. colored by gravity regime
fig, ax = plt.subplots(figsize=(10, 7))
cols = {"Microgravity (<=0.01g)": "#1f77b4", "Partial g (0.01-0.9g)": "#ff7f0e",
        "Normal g (~1g)": "#2ca02c", "Hypergravity (>1.1g)": "#d62728"}
reg_sub = regime[m]
for lab, col in cols.items():
    s = sub[reg_sub == lab]
    ax.scatter(s[COL["o2"]], s[COL["pressure"]], s=12, alpha=0.45, c=col,
               label=f"{lab} (n={len(s):,})", edgecolors="none")
ax.set_yscale("log")
ax.set_xlabel("Oxygen Concentration [mole fraction]")
ax.set_ylabel("Pressure [kPa] (log scale)")
ax.set_title("O2 vs Pressure per Data Point, Colored by Gravity Regime")
ax.legend()
save(fig, "26_o2_vs_pressure_by_gravity.png")

# 27. colored by diluent
fig, ax = plt.subplots(figsize=(10, 7))
dil_cols = {"N2": "#1f77b4", "CO2": "#d62728", "Ar": "#2ca02c", "He": "#ff7f0e"}
for lab, col in dil_cols.items():
    s = sub[sub[COL["diluent"]] == lab]
    ax.scatter(s[COL["o2"]], s[COL["pressure"]], s=12, alpha=0.5, c=col,
               label=f"{lab} (n={len(s):,})", edgecolors="none")
ax.set_yscale("log")
ax.set_xlabel("Oxygen Concentration [mole fraction]")
ax.set_ylabel("Pressure [kPa] (log scale)")
ax.set_title("O2 vs Pressure per Data Point, Colored by Diluent Gas")
ax.legend()
save(fig, "27_o2_vs_pressure_by_diluent.png")

# 28. hexbin density of O2 vs P (coverage / gaps in 2D)
fig, ax = plt.subplots(figsize=(10, 7))
hb = ax.hexbin(sub[COL["o2"]], sub[COL["pressure"]], gridsize=40, yscale="log",
               cmap="viridis", norm=LogNorm(), mincnt=1)
fig.colorbar(hb, ax=ax, label="Data points per cell (log)")
ax.set_xlabel("Oxygen Concentration [mole fraction]")
ax.set_ylabel("Pressure [kPa] (log scale)")
ax.set_title("2D Coverage Density: O2 vs Pressure (empty regions = testing gaps)")
save(fig, "28_o2_vs_pressure_density_hexbin.png")

# 29. O2 vs gravity coverage
m2 = df[COL["o2"]].notna() & df[COL["gravity"]].notna()
fig, ax = plt.subplots(figsize=(10, 7))
for lab, col in IGN_COLORS.items():
    s = df[m2 & (df["ign_clean"] == lab)]
    ax.scatter(s[COL["gravity"]], s[COL["o2"]], s=12, alpha=0.45, c=col,
               label=f"Ignition={lab}", edgecolors="none")
ax.set_xlabel("Gravity [g/g_earth]")
ax.set_ylabel("Oxygen Concentration [mole fraction]")
ax.set_title("Oxygen vs Gravity per Data Point, Colored by Ignition Outcome")
ax.legend()
save(fig, "29_o2_vs_gravity_by_ignition.png")

# ================================================================
# FSR RELATIONSHIPS
# ================================================================
fs = df[df["fsr_num"].notna() & (df["fsr_num"] > 0)]

# 30. FSR vs O2
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(fs[COL["o2"]], fs["fsr_num"], s=14, alpha=0.5,
                c=fs[COL["pressure"]], cmap="plasma", norm=LogNorm())
fig.colorbar(sc, ax=ax, label="Pressure [kPa] (log)")
ax.set_yscale("log")
ax.set_xlabel("Oxygen Concentration [mole fraction]")
ax.set_ylabel("Flame Spread Rate [mm/s] (log)")
ax.set_title("FSR vs Oxygen Concentration (colored by Pressure)")
save(fig, "30_fsr_vs_oxygen.png")

# 31. FSR vs gravity
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(fs[COL["gravity"]], fs["fsr_num"], s=14, alpha=0.5,
                c=fs[COL["o2"]], cmap="viridis")
fig.colorbar(sc, ax=ax, label="O2 mole fraction")
ax.set_yscale("log")
ax.set_xlabel("Gravity [g/g_earth]")
ax.set_ylabel("Flame Spread Rate [mm/s] (log)")
ax.set_title("FSR vs Gravity Level (colored by O2 Concentration)")
save(fig, "31_fsr_vs_gravity.png")

# 32. FSR vs fuel density
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(fs[COL["fuel_rho"]], fs["fsr_num"], s=14, alpha=0.5, c="#1f77b4",
           edgecolors="none")
ax.set_yscale("log")
ax.set_xlabel("Fuel Density [kg/m3]")
ax.set_ylabel("Flame Spread Rate [mm/s] (log)")
ax.set_title("FSR vs Fuel Density")
save(fig, "32_fsr_vs_fuel_density.png")

# 33. FSR boxplot by geometry
fig, ax = plt.subplots(figsize=(9, 6))
groups, labels = [], []
for gname, gdf in fs.groupby("geom_clean"):
    if len(gdf) >= 10:
        groups.append(gdf["fsr_num"].values)
        labels.append(f"{gname}\n(n={len(gdf)})")
ax.boxplot(groups, tick_labels=labels, showfliers=False)
ax.set_yscale("log")
ax.set_ylabel("Flame Spread Rate [mm/s] (log)")
ax.set_title("FSR Distribution by Sample Geometry (boxplot, outliers hidden)")
save(fig, "33_fsr_boxplot_by_geometry.png")

# 34. FSR boxplot by gravity regime
fig, ax = plt.subplots(figsize=(9, 6))
fs_reg = pd.cut(fs[COL["gravity"]], bins=[-0.001, 0.01, 0.9, 1.1, 100],
                labels=["Micro-g", "Partial g", "Normal g", "Hyper-g"])
groups, labels = [], []
for gname in ["Micro-g", "Partial g", "Normal g", "Hyper-g"]:
    vals = fs.loc[fs_reg == gname, "fsr_num"].values
    if len(vals) >= 5:
        groups.append(vals)
        labels.append(f"{gname}\n(n={len(vals)})")
ax.boxplot(groups, tick_labels=labels, showfliers=False)
ax.set_yscale("log")
ax.set_ylabel("Flame Spread Rate [mm/s] (log)")
ax.set_title("FSR Distribution by Gravity Regime (boxplot, outliers hidden)")
save(fig, "34_fsr_boxplot_by_gravity_regime.png")

# ================================================================
# FUEL PROPERTY SPACE
# ================================================================
# 35. fuel density vs conductivity scatter
m3 = df[COL["fuel_rho"]].notna() & df[COL["fuel_k"]].notna()
fig, ax = plt.subplots(figsize=(10, 7))
s = df[m3]
sc = ax.scatter(s[COL["fuel_rho"]], s[COL["fuel_k"]], s=14, alpha=0.5,
                c=s[COL["fuel_cp"]], cmap="cividis")
fig.colorbar(sc, ax=ax, label="Fuel cp [J/kg-K]")
ax.set_yscale("log")
ax.set_xlabel("Fuel Density [kg/m3]")
ax.set_ylabel("Fuel Conductivity [W/m-K] (log)")
ax.set_title("Fuel Property Space: Density vs Conductivity (colored by cp)")
save(fig, "35_fuel_property_space.png")

# 36. unique fuels tested per paper
fuels_pp = df.groupby(COL["paper"])[COL["fuel_rho"]].nunique().sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(fuels_pp, bins=range(0, fuels_pp.max() + 2), color="#ff7f0e", edgecolor="k")
ax.set_xlabel("Unique fuel densities tested in a paper")
ax.set_ylabel("Number of papers")
ax.set_title("Number of Distinct Fuels (by density) Tested per Paper")
save(fig, "36_unique_fuels_per_paper.png")

# ================================================================
# DATA COMPLETENESS
# ================================================================
# 37. missingness per column
core_cols = [COL[k] for k in ["fuel_rho", "fuel_k", "fuel_cp", "o2", "diluent",
                              "pressure", "flow", "gravity", "facility", "geom",
                              "ign_method", "ignition", "hrr"]] + ["fsr_num", "flame_len_num"]
nice = ["Fuel density", "Fuel k", "Fuel cp", "O2", "Diluent", "Pressure",
        "Flow velocity", "Gravity", "Facility", "Geometry", "Ignition method",
        "Ignition label", "HRR", "FSR (numeric)", "Flame length (numeric)"]
comp = [(100 * df[c].notna().mean()) for c in core_cols]
fig, ax = plt.subplots(figsize=(10, 6))
order = np.argsort(comp)
ax.barh([nice[i] for i in order], [comp[i] for i in order], color="#1f77b4", edgecolor="k")
for i, o in enumerate(order):
    ax.text(comp[o] + 0.5, i, f"{comp[o]:.1f}%", va="center", fontsize=8)
ax.set_xlabel("% of rows with a value")
ax.set_xlim(0, 108)
ax.set_title("Data Completeness by Field (% non-missing)")
save(fig, "37_data_completeness_by_field.png")

# 38. correlation heatmap of numeric variables
num_cols = {COL["fuel_rho"]: "fuel_rho", COL["fuel_k"]: "fuel_k",
            COL["fuel_cp"]: "fuel_cp", COL["o2"]: "O2", COL["pressure"]: "P",
            COL["flow"]: "flow_v", COL["gravity"]: "gravity",
            "fsr_num": "FSR", "flame_len_num": "flame_len"}
cm = df[list(num_cols)].rename(columns=num_cols).corr(method="spearman")
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=45, ha="right")
ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.columns)
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(cm.iloc[i, j]) > 0.5 else "black")
fig.colorbar(im, ax=ax, label="Spearman correlation")
ax.set_title("Spearman Correlation Between Numeric Variables")
save(fig, "38_correlation_heatmap.png")

# ================================================================
# PARAMETER COVERAGE PER PAPER (span of key variables)
# ================================================================
# 39. O2 span per paper (range plot, top 30 papers)
fig, ax = plt.subplots(figsize=(10, 10))
rng = df.groupby("paper_id")[COL["o2"]].agg(["min", "max", "count"])
rng = rng.loc[top30].sort_values("count")
for i, (name, row) in enumerate(rng.iterrows()):
    ax.plot([row["min"], row["max"]], [i, i], lw=3, color="#1f77b4", alpha=0.7)
    ax.plot([row["min"], row["max"]], [i, i], "o", ms=4, color="#1f77b4")
ax.set_yticks(range(len(rng))); ax.set_yticklabels(rng.index, fontsize=7)
ax.set_xlabel("Oxygen Concentration [mole fraction]")
ax.set_title("O2 Range Explored by Each Paper (top 30 papers)")
save(fig, "39_o2_span_per_paper.png")

# 40. pressure span per paper
fig, ax = plt.subplots(figsize=(10, 10))
rng = df.groupby("paper_id")[COL["pressure"]].agg(["min", "max", "count"])
rng = rng.loc[top30].dropna().sort_values("count")
for i, (name, row) in enumerate(rng.iterrows()):
    ax.plot([row["min"], row["max"]], [i, i], lw=3, color="#d62728", alpha=0.7)
    ax.plot([row["min"], row["max"]], [i, i], "o", ms=4, color="#d62728")
ax.set_xscale("log")
ax.set_yticks(range(len(rng))); ax.set_yticklabels(rng.index, fontsize=7)
ax.set_xlabel("Pressure [kPa] (log)")
ax.set_title("Pressure Range Explored by Each Paper (top 30 papers)")
save(fig, "40_pressure_span_per_paper.png")

# ================================================================
# GAP TABLE FIGURE - top gaps of every key variable
# ================================================================
def top_gaps_text(vals, label, unit, n=3, log=False):
    v = pd.to_numeric(vals, errors="coerce").dropna()
    if log:
        v = v[v > 0]
    u = np.sort(v.unique())
    if len(u) < 2:
        return []
    gaps = np.diff(np.log10(u)) if log else np.diff(u)
    idx = np.argsort(gaps)[::-1][:n]
    return [f"{label}: {u[i]:.4g} -> {u[i+1]:.4g} {unit}" for i in sorted(idx)]

gap_lines = []
gap_lines += top_gaps_text(df[COL["fuel_rho"]], "Fuel density", "kg/m3")
gap_lines += top_gaps_text(df[COL["fuel_k"]], "Fuel k", "W/mK", log=True)
gap_lines += top_gaps_text(df[COL["o2"]], "O2", "mol frac")
gap_lines += top_gaps_text(df[COL["pressure"]], "Pressure", "kPa", log=True)
gap_lines += top_gaps_text(df[COL["gravity"]], "Gravity", "g")
gap_lines += top_gaps_text(df["fsr_num"], "FSR", "mm/s", log=True)
fig, ax = plt.subplots(figsize=(9, 7))
ax.axis("off")
ax.set_title("Largest Untested Gaps in Key Variables (top 3 each)", fontsize=13)
for i, line in enumerate(gap_lines):
    ax.text(0.02, 0.95 - i * 0.052, "- " + line, fontsize=10, va="top",
            family="monospace")
save(fig, "41_largest_gaps_summary_table.png")

# ================================================================
# 42. points per paper vs ignition fraction bubble
# ================================================================
agg = df.groupby(COL["paper"]).agg(
    n=("ign_clean", "size"),
    ign_frac=("ign_clean", lambda s: (s == "Yes").mean()),
    o2_span=(COL["o2"], lambda s: s.max() - s.min()),
)
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(agg["n"], agg["ign_frac"] * 100, s=30 + 600 * agg["o2_span"].fillna(0),
                alpha=0.55, c=agg["o2_span"], cmap="viridis", edgecolors="k", linewidths=0.4)
fig.colorbar(sc, ax=ax, label="O2 span explored (mole fraction)")
ax.set_xscale("log")
ax.set_xlabel("Points in paper (log)")
ax.set_ylabel("% Ignition = Yes")
ax.set_title("Paper Size vs Ignition Fraction (bubble size = O2 span explored)")
save(fig, "42_paper_size_vs_ignition_fraction.png")

print(f"\nDone. {len(saved)} figures saved to {OUT}")
