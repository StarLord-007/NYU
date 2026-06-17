# Fable — Unseen-Paper Generalisation Pipeline for the Microgravity Ignition Classifier

This folder contains the full audit, the rebuilt training/evaluation pipeline,
and the benchmark answering one question: **how well does an ignition /
no-ignition classifier trained on this literature corpus work on papers it has
never seen, and what actually improves that?**

## TL;DR

1. The previously reported numbers were all biased. Honest values
   (deduplicated data, canonical paper grouping): stratified CV **0.894**
   (was 0.903; 19.8% duplicate rows), grouped CV **0.649 ± 0.07** (was 0.678;
   the old DOI group key was corrupted by Excel autofill and citation
   aliases, splitting 15.4% of rows' papers across folds).
2. The stratified→grouped collapse is real and decomposes into: intact
   within-paper ranking on unseen papers (median per-paper AUC **0.80**) +
   near-chance between-paper calibration (**0.597**). It is concept/label
   shift across campaigns, not covariate shift, and it is enabled by the
   feature vector being a 98%-accurate paper fingerprint.
3. All required interventions (paper weighting, combined weights, focal loss,
   group-aware HPO, threshold optimisation, domain-generalisation variants)
   were implemented and benchmarked under a nested, leak-free grouped
   protocol. **None of them moves pooled unseen-paper ROC-AUC beyond fold
   noise** (18 configs, range 0.626–0.650, all paired Wilcoxon p > 0.05).
   The ceiling is set by what the features don't contain. The deliverable
   improvements are therefore measurement validity, leak-free decision
   thresholds, and statistically correct weighting — plus a concrete data
   road map for raising the ceiling.

## Contents

| File | What it is |
|---|---|
| `AUDIT_REPORT.md` | Deliverable 1: full audit (data, preprocessing, features, splits, HPO, evaluation) and answers to questions 1–3 |
| `LEAKAGE_ANALYSIS.md` | Deliverable 2: SHAP / permutation / mutual-information / clustering leakage evidence |
| `PAPER_IMBALANCE_ANALYSIS.md` | Deliverable 3: per-paper size & class-composition analysis |
| `GROUP_COLLAPSE_EXPLANATION.md` | Deliverable 4: quantified explanation of the 0.90 → 0.65 collapse |
| `fable_train.py` | Deliverable 5: training script (weighting A/B, thresholds C, focal loss D, group-aware HPO E, domain options F) |
| `fable_eval.py` | Deliverable 6: evaluation/benchmark harness (grouped protocols, nested thresholds, paired statistics, figures) |
| `fable_common.py` | Shared data layer: cp1252-safe loading, canonical paper IDs, deduplication, tagged feature registry, weighting functions |
| `fable_audit.py` | Reproduces every number in the audit reports |
| `audit_outputs/`, `eval_outputs/`, `model_outputs/` | Deliverable 7: figures + all numeric artifacts |
| `AUGMENTATION_ASSESSMENT.md` | Augmentation feasibility analysis (verdict: rejected, with per-method risk/benefit) |

## How to run

```bash
pip install xgboost scikit-learn pandas numpy matplotlib shap scipy joblib

python Fable/fable_audit.py  --data Microgravity_Database_Latest.csv   # audit evidence
python Fable/fable_eval.py   --data Microgravity_Database_Latest.csv   # full benchmark (~50 min)
python Fable/fable_eval.py   --quick                                   # smoke test (~20 s)
python Fable/fable_train.py  --data Microgravity_Database_Latest.csv   # final model (recommended defaults)
python Fable/fable_train.py  --tune --n-iter 60                        # group-aware hyperparameter search
```

## Deliverable 8 — Comparison table

Primary protocol: StratifiedGroupKFold(5) over canonical papers × 3 shuffle
seeds (15 unseen-paper test folds). MCC / F1 / balanced accuracy use
thresholds tuned **only on inner validation folds** (held-out papers inside
the training split) — no test leakage anywhere. Mean ± sd over folds; full
per-fold data in `eval_outputs/benchmark_folds.csv`.

| Method | Group ROC-AUC | PR-AUC | MCC | Balanced Accuracy |
|---|---|---|---|---|
| focal g=1 | 0.650 ± 0.055 | 0.842 ± 0.031 | 0.152 ± 0.090 | 0.588 ± 0.044 |
| focal g=2 | 0.650 ± 0.054 | 0.843 ± 0.032 | 0.161 ± 0.080 | 0.587 ± 0.046 |
| combined 1/N × class | 0.648 ± 0.051 | 0.841 ± 0.028 | 0.148 ± 0.088 | 0.588 ± 0.049 |
| baseline (spw, v2 config) | 0.647 ± 0.057 | 0.842 ± 0.032 | 0.140 ± 0.088 | 0.582 ± 0.055 |
| paper 1/√N | 0.645 ± 0.049 | 0.840 ± 0.027 | 0.143 ± 0.085 | 0.588 ± 0.049 |
| combined 1/√N × class | 0.644 ± 0.055 | 0.839 ± 0.031 | 0.148 ± 0.081 | 0.592 ± 0.055 |
| class-only (balanced) | 0.644 ± 0.056 | 0.835 ± 0.036 | 0.151 ± 0.087 | 0.597 ± 0.057 |
| paper 1/N | 0.643 ± 0.050 | 0.836 ± 0.028 | 0.153 ± 0.078 | 0.582 ± 0.047 |
| focal g=2 + paper √N | 0.642 ± 0.059 | 0.836 ± 0.034 | 0.135 ± 0.083 | 0.591 ± 0.046 |
| paper effective-number | 0.640 ± 0.048 | 0.837 ± 0.025 | 0.135 ± 0.081 | 0.581 ± 0.048 |
| paper 1/(1+lnN) | 0.640 ± 0.052 | 0.839 ± 0.030 | 0.123 ± 0.075 | 0.576 ± 0.042 |
| fable + paper-bagging 25 | 0.640 ± 0.061 | 0.839 ± 0.036 | **0.169 ± 0.101** | 0.576 ± 0.042 |
| group-tuned HP + combined w | 0.638 ± 0.054 | 0.835 ± 0.031 | 0.138 ± 0.079 | 0.574 ± 0.046 |
| group-tuned HP | 0.637 ± 0.053 | 0.835 ± 0.033 | 0.140 ± 0.078 | 0.586 ± 0.048 |
| physics + combined w + monotone O₂ + group HP ("fable") | 0.633 ± 0.065 | 0.837 ± 0.037 | 0.130 ± 0.097 | 0.572 ± 0.050 |
| physics features | 0.631 ± 0.064 | 0.833 ± 0.043 | 0.139 ± 0.071 | 0.580 ± 0.036 |
| physics + combined w | 0.629 ± 0.070 | 0.832 ± 0.042 | 0.138 ± 0.095 | 0.572 ± 0.046 |
| physics + combined w + group HP | 0.626 ± 0.063 | 0.831 ± 0.037 | 0.135 ± 0.077 | 0.574 ± 0.039 |

Paired per-fold statistics vs the baseline (identical partitions, Wilcoxon
signed-rank, n = 15 folds; `eval_outputs/paired_vs_baseline.csv`): largest
ROC-AUC delta **+0.003** (focal γ=2, p = 0.39); largest MCC delta **+0.029**
(paper-bagging, p = 0.08). Nothing is significant. The reference protocol
ladder for the same baseline model: random hold-out 0.893, stratified 5-fold
0.894, GroupKFold 0.647, StratifiedGroupKFold×3 0.647, LOPO pooled 0.655.

Reading guide: differences *within this table* are noise; the difference
between this table and the 0.89 random-split numbers is the finding.

## Deliverable 9 — Interventions ranked by expected scientific value

1. **Adopt the grouped validation protocol as the only reported protocol**
   (canonical paper IDs; repeated StratifiedGroupKFold + LOPO per-paper
   distribution; nested model/threshold selection). Cost: the headline drops
   to 0.65/0.80. Value: the publication's claim becomes true. Implemented.
2. **Data hygiene: deduplication + canonical DOI/citation identity.**
   Removes 19.8% duplicate rows and the corrupted group key; both directly
   falsified the previous numbers. Implemented (`fable_common.py`).
3. **Curate the corpus instead of augmenting it** (not a code change):
   ingest no-ignition-rich papers (40/85 papers have zero no-ignition rows),
   record per-paper measurement uncertainties and material thermophysical
   properties, and harmonise the operational definition of "ignition" across
   sources. This is the only intervention with a credible path to raising
   the 0.65 ceiling, because the ceiling is informational, not algorithmic.
4. **Leak-free threshold policy** (requirement C). Thresholds tuned on inner
   held-out-paper folds; exposed the fact that no global threshold transfers
   (MCC-optimal threshold spans 0.39–0.997 across folds). Operational
   recommendation: report probabilities or recalibrate per campaign.
   Implemented (`optimal_thresholds`, nested in `fable_eval.py`).
5. **Combined paper × class weighting, `1/√N` default** (requirements A+B).
   Pooled-metric effect ≈ 0, but it is the statistically correct treatment
   of clustered data (papers are the independent units) and removes the
   objection that 5 campaigns steer a third of the gradient. Implemented;
   default in `fable_train.py`.
6. **Group-aware hyperparameter optimisation** (requirement E). Valuable
   mainly as a negative result: 60 candidates span only 0.584–0.646 grouped
   AUC — capacity/regularisation is not the binding constraint, and
   random-split tuning (the old protocol) was selecting for memorisation.
   Implemented (`--tune`).
7. **Focal loss** (requirement D). Best paired deltas of any single change
   (ΔMCC +0.021, p = 0.07 at γ=2) but below significance; keep as an option,
   do not build claims on it. Implemented (verified custom objective).
8. **Paper-level cluster bagging.** Largest MCC gain (+0.029, p = 0.08),
   statistically principled (cluster bootstrap), costs 25× training time;
   optional (`--paper-bagging`).
9. **Physics-only feature set + monotone O₂ constraint** (requirement F).
   Costs ~0.02 pooled AUC (within noise, p ≈ 0.11) and removes the widest
   fingerprint channels; choose it when reviewer-facing interpretability
   ("the model cannot see the rig") outweighs a marginal metric difference.
   Implemented (`--feature-set physics`, `--monotone-o2`).
10. **Data augmentation: do not.** All eight candidate methods either inject
    label noise at the flammability boundary or amplify campaign
    fingerprints; expected grouped-AUC effects are ≤ 0. Full reasoning in
    `AUGMENTATION_ASSESSMENT.md`.

## Key figures (deliverable 7)

* `eval_outputs/fig_protocol_comparison.png` — the optimism ladder (0.89 → 0.65)
* `eval_outputs/fig_method_comparison.png` — all 18 methods, per-fold scatter
* `eval_outputs/fig_lopo_scatter.png` — per-paper baseline-vs-fable AUC
* `audit_outputs/fig_lopo_within_paper_auc.png` — which papers fail and how badly
* `audit_outputs/fig_paper_imbalance.png` — size/class imbalance structure
* `audit_outputs/fig_shap_summary.png`, `fig_feature_clustering.png` — leakage evidence
