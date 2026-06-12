# Why ROC-AUC Collapses from ~0.90 (stratified) to ~0.65–0.68 (grouped)

This document answers audit question 1 with measurements, not speculation.
All numbers from `fable_audit.py` (`audit_outputs/audit_metrics.json`,
`per_paper_lopo.csv`, `lopo_oof_predictions.csv`).

## 0. First correction: both of the user-reported numbers were biased

| Quantity | reported | measured honest value | bias source |
|---|---|---|---|
| Stratified 5-fold ROC-AUC | 0.903 | **0.894** (0.915 with duplicates) | 19.8% exact duplicate rows shared between train and test folds |
| Group-by-DOI 5-fold ROC-AUC | 0.678 ± 0.10 | **0.649 ± 0.07** (canonical grouping, latest CSV) | old DOI key had 551 groups (Excel-autofilled DOIs + citation aliases); 15.4% of rows belonged to papers split across folds |

So the true gap is **0.894 → 0.649**, slightly larger than reported. The
collapse is real and is *not* an artefact of the grouped protocol; the
artefacts actually ran in the optimistic direction.

## 1. Decomposition of the collapse

Leave-One-Paper-Out (85 fits, baseline model) decomposes the grouped score
into interpretable parts:

| Quantity | value |
|---|---|
| Pooled LOPO ROC-AUC (all out-of-paper predictions ranked together) | **0.651** |
| Between-paper component (replace each prediction by its paper's mean prediction) | **0.597** |
| Mean within-paper AUC (45 scorable two-class papers) | **0.747** |
| Median within-paper AUC | **0.804** |
| Row-weighted mean within-paper AUC | 0.691 |

Interpretation:

* **The model still ranks conditions correctly *inside* a typical unseen
  paper** (median 0.80): more O₂ → flame, less O₂ → no flame is learned and
  transfers.
* **The pooled grouped score is dragged down by the between-paper axis**
  (0.597, barely above chance): the model cannot place *papers* on a common
  probability scale. Each campaign has its own offset (different rigs,
  materials, unreported covariates, label conventions), so probabilities are
  miscalibrated across papers, and pooled AUC mixes these offsets into the
  ranking.
* In other words: stratified CV asks *"can you interpolate inside campaigns
  you have seen?"* (yes, 0.89); grouped CV asks *"can you rank rows from
  campaigns you have never seen, all on one scale?"* (barely, 0.65); the
  scientifically meaningful intermediate question *"on a new campaign, do you
  order its conditions correctly?"* sits at 0.75–0.80.

## 2. Which papers drive the collapse (evidence)

Within-paper LOPO AUC, worst offenders (all large or moderate two-class
papers):

| Paper | n | ign. rate | within-paper AUC |
|---|---|---|---|
| Rojas-Alva 2022 (combustflame.2022.112009) | 103 | 0.53 | **0.270** |
| Takahashi 2013 (proci/jcombustflame .2013.03.029) | 18 | 0.83 | 0.311 |
| Feier 2002 (S0082-0784(02)80211-3) | 39 | 0.41 | 0.386 |
| Bolodyan 2019 (firesaf.2019.01.044) | 32 | 0.75 | 0.451 |
| Olson 2009 (combustflame.2009.01.015) | 108 | 0.82 | 0.458 |
| Urban 2025 (ICES-2025-392, lunar-g) | 221 | 0.48 | 0.478 |
| Konno 2024 (proci.2024.105199, centrifuge) | 38 | 0.68 | 0.526 |
| Ma 2025 (firesaf.2024.104329) | 255 | 0.61 | 0.540 |
| Olson 2012 (ICES 2012, Mars-g) | 58 | 0.43 | 0.543 |
| Konno 2023 (ISO 4589-4) | 314 | 0.69 | 0.563 |

* 6 of 45 scorable papers score **below 0.5** — the model's transferred rules
  are *anti-correlated* with those papers' outcomes.
* The five worst large papers (Rojas-Alva, Urban, Ma, Olson 2009, Konno ISO)
  jointly hold 1,001 rows ≈ 24% of the corpus; their near-random or inverted
  within-paper scores, pooled with everyone else's offsets, fully account
  for the fold-level grouped scores of 0.55–0.65.
* Mechanism for inversion (Rojas-Alva 0.27 is the clearest case): these
  campaigns probe a *different physical boundary* (e.g. concurrent-flow
  spread limits, lunar/partial gravity, regression burning of thick fuels)
  where the corpus-level shortcut "higher O₂/flow ⇒ ignition" reverses or
  saturates. The model transfers the majority-campaign rule with high
  confidence and gets systematically inverted outcomes.

## 3. Are the failing papers out-of-distribution? (mostly no)

Feature-space novelty (median robust z-score of a paper's rows w.r.t. the
rest of the corpus) correlates only weakly and *insignificantly* with
within-paper AUC: Spearman ρ = −0.17 (p = 0.26, n = 45). The most novel
papers by feature range (Olson 2004, Olson 2024) actually score *well*
(AUC 1.00, 0.60). 

**Conclusion: the collapse is not primarily a covariate-shift problem; it is
a concept/label-shift problem.** The same nominal (O₂, flow, pressure,
geometry) coordinates map to different outcomes in different campaigns,
because the campaigns differ in unrecorded covariates: igniter coupling,
sample conditioning/moisture, opposed vs concurrent configuration details,
chamber confinement effects, and the authors' operational definition of
"ignition" (sustained flame vs any flash vs spread). No reweighting or
regularisation can fully remove an effect that is not encoded in the
features; this bounds what any of the implemented interventions can achieve
(see comparison table) and is the honest headline for the paper.

## 4. Is the model memorising experimental campaigns? (yes, measurably)

* Features identify the source paper with **98.0%** accuracy (85-class probe,
  chance 3%) — the input space is partitioned into campaign cells.
* The baseline model spends large SHAP mass on features whose grouped
  permutation importance is zero (`ignition_time_s`, `material_grouped`,
  `internal_dim_*`, `sample_dim_3`) — i.e. on features that only help to
  *recognise the campaign*, not to predict ignition on a new one
  (`LEAKAGE_ANALYSIS.md` §4.4).
* 43 papers are 100%-ignition; recognising such a campaign yields a free
  perfect score on its rows in any random split. In grouped splits this
  channel pays nothing — which is precisely the stratified-vs-grouped gap.
* Duplicates sharpen the effect: before deduplication, 19.8% of rows have an
  exact twin, and stratified CV rises from 0.894 to 0.915 with zero change in
  knowledge.

## 5. What the collapse is *not*

* Not the grouped protocol being "too hard/unfair": grouped CV is the only
  protocol matching the deployment question (predict ignition for an
  experiment from a lab/campaign not in the training set).
* Not small-data noise alone: fold spread is ±0.07, but the gap is 0.25.
* Not a bug in GroupKFold usage — though the *old* grouped number 0.678 was
  computed on a corrupted key and on its own file would have been ≈0.65 with
  honest groups (measured: old key 0.705 vs canonical key 0.646 on
  `database_xgb.csv` with the legacy pipeline).

## 6. Implications used downstream

1. Optimise and select models **only** on grouped protocols
   (`fable_train.py --tune` uses StratifiedGroupKFold).
2. Report between-paper calibration separately from within-paper ranking;
   consider per-paper recalibration (one global threshold cannot be optimal
   for every campaign — measured threshold spread across validation folds is
   wide).
3. Close fingerprint channels (physics feature set) and cap campaign
   gradient share (paper weighting) — implemented and benchmarked.
4. The remaining gap is a *data* problem (unrecorded covariates,
   inconsistent ignition definitions), not a modelling problem. The paper
   should say so.
