# Audit Report — Microgravity Ignition Classifier

Scope: `xgb_ignition_model_2.py` + `Microgravity_Database_Latest.csv` (and the
older `database_xgb.csv` it was actually run on). Everything quantitative in
this report is reproducible via `python Fable/fable_audit.py`; artifacts live
in `Fable/audit_outputs/`.

Companion documents:

* `GROUP_COLLAPSE_EXPLANATION.md` — audit question 1 (the 0.90 → 0.65 collapse)
* `LEAKAGE_ANALYSIS.md` — audit question 2 (feature leakage)
* `PAPER_IMBALANCE_ANALYSIS.md` — per-paper imbalance
* `AUGMENTATION_ASSESSMENT.md` — augmentation feasibility
* `eval_outputs/comparison_table.md` — benchmark of all implemented interventions

---

## 1. Executive summary

The pipeline's headline numbers (hold-out ROC-AUC 0.897, stratified CV 0.903)
do not measure the scientific claim the project wants to make. They measure
within-campaign interpolation on a dataset where (i) ~20% of rows are exact
duplicates, (ii) features identify the source paper with 98% accuracy, and
(iii) 47% of papers contain only one class. The honest unseen-paper score of
the existing model is **0.649 ± 0.07** (GroupKFold over canonical papers), and
even the previously reported 0.678 was computed against a **corrupted group
key** that scattered rows of single papers across folds.

The collapse decomposes cleanly: the model retains useful *within-paper*
ranking on unseen papers (median per-paper AUC **0.80**) but has almost no
*between-paper* transferability (between-paper component **0.597**), because
campaign-specific offsets are not encoded in any feature. Reweighting,
feature pruning, focal loss and group-aware tuning (all implemented and
benchmarked, §6) recover a measurable but bounded share of the gap; the
remainder is a data-content limitation, not a modelling defect.

## 2. Data audit (the CSV itself)

### 2.1 Findings, ordered by severity

**F1 — Corrupted DOI column in `database_xgb.csv` (critical, affects the
reported 0.678).** The 416-row Konno ISO-4589-4 paper carries 243 distinct
DOIs (`…103989 → …104231`, an Excel autofill drag). The old group key
(`DOI.fillna(Article)`) therefore produced 551 "papers" instead of 88.
Measured consequence: 15.4% of rows belonged to physical papers split across
GroupKFold test folds. The latest CSV fixes the autofill (1 DOI for that
paper) — but the old reported numbers were generated against the corrupted
file.

**F2 — Citation aliases (critical for A+B+C grouping).** Six physical papers
appear under two citation strings each (e.g. *"Konno, Yusuke, et al."* vs the
full author list; curly vs straight quotes; `Olson et al., Combust. Flame 156
(2009)` vs the full MLA form). The user-specified paper key
`Article + Authors + DOI` yields 97 groups; canonicalising DOIs and citation
strings yields **85 physical papers**. Grouping by the raw A+B+C key leaks
half-papers between folds. Fixed in `fable_common.py::canonical_doi/.._article`.

**F3 — Exact duplicate rows (major).** 1,022 of 5,158 labelled rows (19.8%)
are exact duplicates of another row within the same paper (identical features
*and* label). Measured effect: stratified 5-fold ROC-AUC 0.915 (with
duplicates) vs 0.894 (deduplicated) — pure train/test contamination in any
random split. The Fable loader drops within-paper exact duplicates (keeping
conflicting-outcome replicates, which are physically meaningful stochastic
repeats).

**F4 — Mixed encodings and unit chaos (moderate, handled).** The latest CSV is
cp1252 (the v2 script crashes on it: `UnicodeDecodeError`). O₂ appears as
both `"21%"` and `0.21`; pressure in kPa/atm/psia/MPa and bare numbers;
gravity as g-multiples and the token `1e-05`; dimensions as free text with
µm/mm/cm/m. The v2 parsers handle most of this and were retained, after
verification, in `fable_common.py`.

**F5 — Label-definition heterogeneity (structural, unfixable in code).**
"Ignition (Yes/No)" aggregates different operational definitions across
papers (sustained spread vs flash vs self-extinguishing flame). This is a
label-shift component of the collapse that no amount of modelling removes;
see `GROUP_COLLAPSE_EXPLANATION.md` §3.

**F6 — 56 unlabelled rows** (dropped, same as v2). **40 single-class papers**
(all 100%-ignition; 903 rows) — see `PAPER_IMBALANCE_ANALYSIS.md`.

### 2.2 Consequence

Any result computed on the old file or with the old grouping is unreliable in
the optimistic direction. All Fable numbers use: latest CSV, cp1252-aware
loading, canonical paper identity, within-paper deduplication.

## 3. Preprocessing & feature-engineering audit

What the v2 script does well:

* unit normalisation is broadly correct and was kept;
* post-outcome columns (Flame Length, FSR, HRR, Smoke) are correctly dropped;
* native-NaN handling for XGBoost instead of fabricated imputations;
* rare-category grouping prevents one-hot blowup.

Criticisms:

* **C1 — `material_grouped` is a paper alias.** It one-hot-encodes raw
  free-text material strings (≥10 occurrences). Lab-specific spellings make
  it a categorical fingerprint (MI with paper = 2.47 nats; grouped
  permutation importance ≈ 0). It duplicates the legitimate
  `material_family`.
* **C2 — Apparatus descriptors are fed to the model as if they were physics.**
  `internal_dim_*` (chamber size), `facility_cat`, `ig_method_cat`,
  `ignition_power_w/time_s/energy_j` (igniter protocol), `sample_dim_count`
  (a parsing artefact — literally "how many numbers were in the dimension
  string", target MI 0.000, paper MI 1.01). None of these transfers
  (grouped permutation importance ≤ noise); all identify the rig.
* **C3 — Redundant features amplify fingerprints.** `{sample_dim_1, _2,
  _max, _mean}` and `{gravity_g, log10_gravity_g}` are near-duplicates
  (Spearman clustering, `fig_feature_clustering.png`); with column
  subsampling, redundancy guarantees a fingerprint copy is available to
  every tree.
* **C4 — `parse_core_outer` regex misses** patterns like "Ø0.5/0.8 mm"; minor,
  affects coverage not correctness.
* **C5 — duplicated-row handling**: v2 deduplicates only *after* attaching
  `source_group`, so alias papers' duplicates survive; and it never
  deduplicates within stratified CV at all (F3).

Fable response: a tagged feature registry (`physics` vs `apparatus` roles) and
a benchmarked `--feature-set physics` option that drops C1+C2 features.

## 4. Train/test strategy & hyperparameter-search audit

* **C6 — All numbers the project quotes derive from random splits.** The
  80/20 hold-out and StratifiedKFold both put rows of every large campaign on
  both sides of the split. Given the 98% paper identifiability, these
  protocols measure campaign recall, not generalisation. Evidence of the
  optimism ladder (same model, deduplicated data): random hold-out ≈ 0.904,
  stratified 5-fold 0.894, grouped 5-fold 0.649, LOPO pooled 0.651.
* **C7 — Hyperparameter search optimises the wrong objective.**
  `RandomizedSearchCV(..., cv=StratifiedKFold, scoring="roc_auc")` selects for
  memorisation capacity (deep trees, mild regularisation). The grouped CV in
  the script is computed *after* tuning, as a diagnostic only — the model was
  never selected for it. Fable rebuilds the search with
  StratifiedGroupKFold scoring (requirement E). Measured outcome: the
  group-aware search landscape is *flat* (60 candidates span 0.584–0.646
  grouped AUC, top-10 within one fold-σ) — hyperparameters were never the
  binding constraint; this is itself an important negative result.
* **C8 — Threshold tuned on the test set.** `_best_f1_threshold(y_test,
  proba_test)` selects the threshold on the hold-out and then reports the
  hold-out F1 at that threshold ("Best F1 = 0.891"). This is direct test-set
  leakage; the honest F1 must use a threshold frozen on validation data.
  Fable implements nested threshold selection (inner grouped folds only).
* **C9 — `GroupKFold` un-shuffled and unstratified.** sklearn's GroupKFold
  packs groups deterministically by size; with 85 papers and fold-σ ±0.07 a
  single partition is an unreliable estimate, and folds can end up with
  badly skewed class ratios (some test folds are 87% ignition). Fable uses
  StratifiedGroupKFold repeated over shuffle seeds + LOPO.
* **C10 — `scale_pos_weight` conflates two imbalances.** The 76/24 class skew
  is partly a paper-composition artefact (40 all-ignition papers); a single
  global reweighting treats it as a class property. Requirement B (combined
  paper × class weights) addresses this; benchmarked.

## 5. Answers to the three audit questions

### Q1 — Why does ROC-AUC collapse from 0.903 to 0.678?

Short version (full evidence in `GROUP_COLLAPSE_EXPLANATION.md`):

1. both endpoints were biased upward (duplicates → 0.894 honest stratified;
   corrupted groups → 0.649 honest grouped);
2. the collapse decomposes into an intact within-paper component (median
   per-paper AUC 0.80) and a near-chance between-paper component (0.597):
   the model cannot place unseen campaigns on a common probability scale;
3. it is concept/label shift, not covariate shift: feature-space novelty does
   not predict per-paper failure (ρ = −0.17, p = 0.26), while specific
   campaigns probing different physical limits invert the transferred rules
   (Rojas-Alva 2022: AUC 0.27; Urban lunar-g 2025: 0.48; Olson 2009: 0.46);
4. memorisation is measurable: 98% paper identifiability, SHAP mass on
   non-transferring fingerprint features, 43 single-class papers whose labels
   are free once the campaign is recognised.

### Q2 — Is there feature leakage?

Yes — three layers, with evidence in `LEAKAGE_ANALYSIS.md`:
identity-key corruption (F1/F2), exact duplicates (F3), and indirect
campaign fingerprints (dimension features, free-text material strings,
chamber/igniter descriptors: MI with paper up to 0.85·H(paper), grouped
permutation importance ≈ 0, SHAP high). Only `oxygen_fraction` (+0.131
grouped permutation AUC), `gravity_g`, and `material_family` carry
unambiguous transferable signal.

### Q3 — Is the current split scientifically valid?

No, for the claim being made. Measured comparison (baseline config,
deduplicated data; `eval_outputs/protocol_comparison.csv`):

| Protocol | What it estimates | ROC-AUC |
|---|---|---|
| random stratified 80/20 (×5 seeds) | within-campaign interpolation | 0.893 ± 0.006 |
| stratified 5-fold | same, averaged | 0.894 ± 0.010 |
| GroupKFold(5), canonical papers | unseen-campaign, one partition | 0.647 ± 0.076 |
| StratifiedGroupKFold(5) × 3 seeds | unseen-campaign, partition-averaged | 0.647 ± 0.055 |
| Leave-One-Paper-Out (pooled) | unseen-campaign, maximal training data | 0.655 |

**Recommended protocol for publication:**

1. **Primary:** repeated **StratifiedGroupKFold(5)** over canonical paper IDs
   (≥3 shuffle seeds), reporting mean ± sd of pooled fold ROC-AUC/PR-AUC, with
   all model selection (hyperparameters, thresholds, feature sets) nested
   inside the training folds.
2. **Secondary:** **LOPO** with the *distribution* of within-paper AUCs
   (median + IQR + count below 0.5), which separates within-paper ranking
   skill from between-paper calibration and is the most informative plot for
   a reviewer.

   LOPO alone is not sufficient as a primary metric because 40/85 papers are
   single-class (no within-paper AUC defined) and pooled LOPO mixes
   between-paper offsets; StratifiedGroupKFold keeps fold class ratios sane
   while still testing unseen papers.
3. Random/stratified splits may be reported only as an explicit
   "within-campaign interpolation" upper bound, clearly labelled as such.

## 6. Interventions implemented and benchmarked

All requirements A–F are implemented in `fable_train.py` / `fable_eval.py`;
the measured comparison table is in `eval_outputs/comparison_table.md` and the
ranked interventions in `README.md`. Summary of what was built:

* **A.** Paper weights: `1/N`, `1/√N`, effective-number (Cui et al. 2019,
  β=0.999), `1/(1+ln N)` (cluster-information argument), all mean-normalised.
* **B.** Combined `paper × class` weights vs `scale_pos_weight` vs each alone.
* **C.** Threshold optimisation for MCC / F1 / balanced accuracy / Youden J on
  inner held-out-paper validation folds only (vectorised over ROC operating
  points; `fable_train.optimal_thresholds`).
* **D.** XGBoost-compatible focal loss (closed-form grad/hess, Wang et al.
  2019 formulation; verified to reproduce binary logistic at γ=0 with
  correlation 1.000) at γ ∈ {1, 2}, with and without paper weights.
* **E.** Group-aware random search (StratifiedGroupKFold(5) scoring), 60
  candidates; winner stored as `GROUP_TUNED_PARAMS` with full history CSV.
* **F.** Domain generalisation: `physics` feature set (fingerprint channels
  removed), monotone constraint on O₂ (ignition probability non-decreasing in
  oxygen fraction — textbook flammability physics), paper-level cluster
  bagging. Per-paper feature normalisation was evaluated and **rejected**:
  z-scoring O₂/pressure within a paper destroys the absolute physical scales
  (LOC thresholds) that are the model's only transferable signal — it would
  trade real physics for cosmetic domain invariance.

### 6.1 Headline benchmark result (read before the table)

Across 18 configurations evaluated on identical StratifiedGroupKFold
partitions (5 folds × 3 seeds, nested threshold selection), **no intervention
changes pooled unseen-paper ROC-AUC beyond fold noise**: all methods sit in
0.626–0.650 with fold-σ ≈ 0.05–0.07, and no paired per-fold Wilcoxon test
against the baseline reaches significance (best ΔAUC +0.003, focal γ=2,
p=0.39; best ΔMCC +0.029, paper-bagging, p=0.08; full table:
`eval_outputs/paired_vs_baseline.csv`). The same holds for LOPO (baseline
pooled 0.655 vs 0.649 for the most heavily constrained config).

This is the central empirical finding of the project: **the unseen-paper
ceiling of this dataset is set by between-paper concept/label shift —
information that is absent from the features — not by the loss function,
weighting, hyperparameters, or feature pruning.** The honest performance
claim for publication is grouped ROC-AUC ≈ 0.65 (pooled) with median
within-paper AUC ≈ 0.80, and the path to improving it is data curation
(§5 of `PAPER_IMBALANCE_ANALYSIS.md`, §"What would actually help" of
`AUGMENTATION_ASSESSMENT.md`), not further model engineering.

A second operationally important finding: **decision thresholds do not
transfer across papers.** The MCC-optimal threshold selected on inner
validation folds ranges from 0.39 to 0.997 across the 15 outer folds (and the
F1-optimal threshold collapses to ≈0.01 because predicting "ignition"
everywhere nearly maximises F1 at 75% prevalence). Any deployment of this
model must either recalibrate per campaign or report probabilities, not hard
labels — a single global threshold is indefensible.
