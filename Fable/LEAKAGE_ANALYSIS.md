# Leakage Analysis

**Question answered here: does any variable directly or indirectly identify the
paper / facility / rig / experimental campaign, and does the model exploit it?**

**Answer: yes — the feature vector is a near-perfect fingerprint of the source
paper (98.0% identifiable), the strongest fingerprint carriers are the sample/
chamber dimension features and the free-text material grouping, and the SHAP
vs grouped-permutation comparison shows the baseline model leans on exactly
those features for in-distribution gains that evaporate on unseen papers.**

All numbers in this document are produced by `fable_audit.py` and stored in
`audit_outputs/`. Definitions: "paper" = canonical physical paper (normalised
DOI, falling back to normalised citation string); n = 4,136 deduplicated
labelled rows, 85 papers.

---

## 1. Direct identifier leakage (resolved before modelling)

Two *direct* identity defects were found in the data layer itself — these are
not "features", but they corrupted the previous grouped validation:

1. **Excel-autofilled DOIs** (`database_xgb.csv`): the 416-row Konno
   ISO-4589-4 paper carries 243 distinct DOI values
   (`…103989, …103990, …103991, …`) because the DOI cell was dragged in
   Excel. The old group key (`DOI.fillna(Article)`) therefore had **551
   groups instead of 88**, and GroupKFold scattered single-paper rows across
   folds. **15.4% of all rows belonged to physical papers split across test
   folds** under the old key.
2. **Citation aliases**: six physical papers appear under two citation
   strings each (curly vs straight quotes, "et al." vs full author list,
   different page formats). Grouping by `Article + Authors + DOI` splits each
   of these papers into two groups — half of the paper can sit in train while
   the other half is "held out".

Both defects are eliminated in `fable_common.py` (`canonical_doi`,
`canonical_article`); all Fable results group by the canonical key.

## 2. Post-outcome columns

`Flame Length`, `FSR`, `HRR`, `Smoke/Aerosols` are observed *after* the
ignition event and trivially encode the label. The v2 script already dropped
these — verified still dropped. `ignition_time_s` and `ignition_power_w` are
*protocol settings* (igniter actuation), not outcomes, so they are legal
predictors in principle — but see §4: they behave as campaign fingerprints.

## 3. Indirect identification: the paper-fingerprint probe

A RandomForest trained to predict **which paper a row comes from**, using only
the model's input features, reaches

* **accuracy = 0.980** (5-fold CV, 85 classes, chance level ≈ 0.030).

This is the central leakage result. Any learner flexible enough to fit this
dataset can first infer "which campaign is this?" and then output that
campaign's base ignition rate — which is profitable in-distribution because
papers differ wildly in base rate (43 papers are 100% ignition). On a truly
unseen paper this strategy collapses to noise, which is what the grouped CV
measures.

## 4. Which features carry the fingerprint?

### 4.1 Mutual information (audit_outputs/mutual_information.csv)

MI of each feature with the target vs with the paper identity
(H(paper) = 3.91 nats). Top fingerprint carriers:

| Feature | MI(target) | MI(paper) | fraction of H(paper) | MI ratio paper:target |
|---|---|---|---|---|
| sample_dim_mean_mm | 0.104 | 3.33 | 0.85 | 32× |
| internal_dim_mean_mm | 0.079 | 2.99 | 0.76 | 38× |
| sample_dim_max_mm | 0.061 | 2.97 | 0.76 | 49× |
| sample_dim_1_mm | 0.056 | 2.75 | 0.70 | 49× |
| internal_dim_1_mm | 0.062 | 2.71 | 0.69 | 44× |
| material_grouped | 0.075 | 2.47 | 0.63 | 33× |
| flow_velocity_mm_s | 0.037 | 1.98 | 0.51 | 54× |
| pressure_kpa | 0.040 | 1.69 | 0.43 | 42× |
| oxygen_fraction | 0.051 | 1.53 | 0.39 | 30× |

Interpretation:

* **Every** feature identifies the paper far better than it predicts
  ignition — this is unavoidable in a literature aggregation (each lab runs
  one rig at a few settings). Leakage is therefore a matter of degree and of
  *how the model uses* the features, not of a single rogue column.
* The **dimension features are near-perfect campaign barcodes**: a single
  number like `sample_dim_mean_mm` recovers 85% of the entropy of the paper
  label. A wire of exactly 0.8 mm outer diameter *is* "the Konno campaign".
* `internal_dim_*` (combustion-chamber dimensions) and `facility_cat`
  describe the *apparatus*, not the specimen — they have near-zero target MI
  (0.012–0.079) and exist almost solely as rig identifiers.
* `material_grouped` keeps raw free-text strings with ≥10 occurrences; the
  raw strings are lab-specific spellings ("Cu core; LDPE outer" etc.), i.e. a
  categorical paper alias. Its coarse counterpart `material_family` carries
  the actual chemistry signal.
* `oxygen_fraction` also identifies papers (each study uses a few O₂ levels)
  but it is simultaneously the only feature with strong *causal* target
  signal — see §4.2. It must stay.

### 4.2 Permutation importance on held-out papers (the honest importance)

Permutation importance computed **inside GroupKFold** (importance = drop in
*unseen-paper* ROC-AUC when the feature is shuffled in the test fold):

| Feature | Δ grouped ROC-AUC |
|---|---|
| oxygen_fraction | **+0.131** |
| gravity_g | +0.011 |
| material_family | +0.009 |
| everything else | ≤ +0.006, mostly ≈ 0 within noise |

### 4.3 SHAP on the baseline model (the in-distribution importance)

Mean |SHAP| of the baseline XGBoost fitted on all rows (top parents):
`oxygen_fraction` 1.37, `flow_speed_abs` 0.46, `ignition_time_s` 0.42,
`flow_velocity` 0.41, `material_grouped` 0.35, `sample_dim_3` 0.30,
`internal_dim_mean` 0.26, `pressure` 0.22, `gravity` 0.22.

### 4.4 The leakage signature: SHAP high, grouped permutation zero

Cross-referencing 4.2 and 4.3:

| Feature | SHAP rank | grouped perm. importance | diagnosis |
|---|---|---|---|
| oxygen_fraction | 1 | +0.131 | genuine physics (LOC) — transfers |
| ignition_time_s | 3 | ≈ 0 (+0.005, σ 0.020) | campaign fingerprint — does not transfer |
| material_grouped | 5 | ≈ 0 (+0.006, σ 0.017) | free-text paper alias |
| internal_dim_mean_mm | 7 | ≈ 0 | chamber barcode |
| sample_dim_3_mm | 6 | ≈ 0 (+0.002, σ 0.015) | specimen barcode |
| gravity_g | 10 | +0.011 | genuine but small (dataset is µg-dominated) |

A feature that the model relies on heavily in-distribution (high SHAP) but
whose shuffling does not hurt unseen-paper performance is, operationally, a
memorisation channel. The baseline model spends roughly half of its total
attribution mass on such channels.

## 5. Feature redundancy (clustering)

Spearman hierarchical clustering (audit_outputs/fig_feature_clustering.png,
threshold 0.3) finds three redundancy groups:
`{gravity_g, log10_gravity_g}`,
`{sample_dim_1, sample_dim_2, sample_dim_max, sample_dim_mean}`,
`{geometry_cat, core_material}`.
Redundant copies of fingerprint-heavy features multiply the chances that at
least one survives subsampling (`colsample_bytree=0.8`) in every tree — i.e.
redundancy makes the fingerprint *more* available, not less.

## 6. Actions taken in the Fable pipeline

1. Canonical paper identity for all grouping (fixes §1).
2. A `physics` feature set (`fable_common.feature_lists("physics")`) that
   drops the pure apparatus descriptors identified here:
   `internal_dim_*`, `facility_cat`, `ig_method_cat`, `ignition_power_w`,
   `ignition_time_s`, `ignition_energy_j`, `sample_dim_count`,
   `material_grouped`, `internal_geom_cat` — every dropped feature has
   grouped permutation importance indistinguishable from zero, so the
   removal costs nothing measurable on unseen papers while closing the
   widest fingerprint channels.
3. Paper-balanced weighting (caps the gradient share of any single
   campaign's fingerprint; see comparison table).
4. The remaining, unavoidable leakage (O₂ levels chosen by authors, materials
   studied per lab) is *reported honestly* by evaluating exclusively with
   grouped protocols.

## 7. What is *not* leakage

For completeness: `oxygen_fraction`, `pressure_kpa`, `flow_velocity_mm_s`,
`gravity_g`, `geometry_cat`, `material_family`, `core_material`, and the wire
diameters are physically meaningful, author-chosen *conditions* of the
experiment. They correlate with paper identity (papers choose conditions) but
removing them would remove the physics; the correct treatment is grouped
validation + paper re-weighting, not deletion.
