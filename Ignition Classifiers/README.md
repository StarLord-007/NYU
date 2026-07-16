# FABLE ignition-classification pipeline

This directory contains one reproducible, leakage-controlled pipeline for binary ignition
classification with XGBoost, K-nearest neighbors, decision trees, multi-layer perceptrons, and
support-vector machines. It compares two deliberately different scientific questions:

- **Interpolation** uses stratified row-level splits. Test conditions can resemble rows from the
  same papers or campaigns represented during training.
- **Extrapolation** holds out complete canonical papers. It estimates transfer to unseen physical
  papers or experimental campaigns and never permits a paper in both partitions.

Canonical paper IDs use normalized true DOIs and fall back to normalized citation identity. Raw DOI
strings are never used as groups. Post-outcome flame length, spread rate, heat release, and smoke
fields are excluded from model features.

## Data assumptions

The default dataset is `../Microgravity_Database_reduced.csv`, one directory above this project.
The CSV has a category row followed by the real header row. The loader tries UTF-8, UTF-8 with BOM,
CP1252, and Latin-1; validates all required source columns; converts units; engineers the declared
feature manifest; reports missingness and target problems; and preserves stable source-based row
IDs. Training and evaluation require `Ignition (Yes/No)`. Inference does not.

The `all` feature set contains physical variables and apparatus descriptors. The `physics` feature
set removes apparatus/campaign descriptors while retaining transferable thermophysical, flow,
gravity, geometry, and gas-composition variables. Missing numeric values are median-imputed for
scikit-learn models; XGBoost retains native numeric missing-value handling. Categoricals use constant
imputation and unknown-safe one-hot encoding.

## Installation

Python 3.11 or newer is recommended. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Full heavy pipeline

Run these commands in order:

```bash
cd "Ignition Classifiers"

python fable_splits.py \
  --data ../Microgravity_Database_reduced.csv \
  --out results/splits \
  --n-seeds 3 \
  --n-group-folds 5 \
  --n-row-folds 5

python fable_evaluate.py \
  --data ../Microgravity_Database_reduced.csv \
  --splits results/splits \
  --config configs/candidates.yaml \
  --out results/evaluation \
  --search-iterations 40 \
  --inner-group-folds 3

python fable_select.py \
  --evaluation results/evaluation \
  --policy configs/selection_policy.yaml \
  --out results/selection.json

python fable_refit.py \
  --data ../Microgravity_Database_reduced.csv \
  --selection results/selection.json \
  --champion interpolation \
  --out artifacts/interpolation_champion

python fable_refit.py \
  --data ../Microgravity_Database_reduced.csv \
  --selection results/selection.json \
  --champion extrapolation \
  --out artifacts/extrapolation_champion

python fable_report.py \
  --data ../Microgravity_Database_reduced.csv \
  --evaluation results/evaluation \
  --selection results/selection.json \
  --artifacts artifacts \
  --out results/report
```

This is intentionally expensive: every candidate is tuned only inside every outer training fold,
all required split protocols are evaluated, LOPO holds out each paper in turn, and uncertainty uses
bootstrap resampling. Runtime depends strongly on paper count, candidate count, CPU, and XGBoost/MLP
convergence.

## Protocol and leakage controls

`fable_splits.py` creates all partitions once and writes exact row and paper assignments. Repeated
80/20 holdouts and stratified five-fold CV measure interpolation. Repeated
`StratifiedGroupKFold(5)` over three seeds measures extrapolation. Leave-One-Paper-Out (LOPO)
provides mandatory paper-by-paper robustness evidence, but is not the sole selection basis.

`fable_evaluate.py` is the only benchmark runner. For each outer fold, `fable_search.py` performs
nested random search on that fold's training data. Interpolation uses stratified inner folds;
extrapolation and LOPO use grouped inner folds. Imputation, scaling, one-hot vocabularies, weighting,
model fitting, and threshold optimization therefore see no outer test rows. ROC-AUC is the search
objective; PR-AUC, MCC, balanced accuracy, F1, Brier score, sensitivity, specificity, and precision
are retained.

Four decision thresholds remain separate: MCC, F1, balanced accuracy, and Youden J. Each is frozen
from inner out-of-fold predictions before scoring the outer test fold. None is called universally
best.

XGBoost and decision trees consume combined class/paper sample weights at fit time. Calibrated SVM
also receives supported fit-time weights. KNN and MLP cannot consume sample weights, so they use a
deterministic probability-proportional bootstrap of each training fold. No model silently ignores
the requested weighting policy. The XGBoost registry also includes logistic and focal objectives,
physics-only monotonic oxygen constraints, and a paper-cluster-bagged candidate.

`fable_select.py` trains nothing. It applies `configs/selection_policy.yaml`, rejects incomplete,
invalid, or unstable candidates, and independently selects interpolation and extrapolation
champions. `fable_refit.py` rebuilds the chosen configuration on all labeled data and derives final
thresholds from full-data out-of-fold predictions under the matching protocol before fitting the
deployable model.

## Reproducibility

The default seed is 42; repeated splits use consecutive seeds. Split files store the dataset SHA-256,
data version, prevalence, paper counts, and exact memberships. Candidate spaces are fixed in
`fable_models.py` and sampled configurations and complete search histories are persisted. Models,
weighted resampling, XGBoost, MLP initialization, calibration, and report resampling use recorded
deterministic seeds.

## Output map

```text
results/
├── splits/                 # CSV, Parquet, compact split assignments and fingerprints
├── evaluation/             # raw predictions, metrics, search history, CIs, integrity, comparisons
├── selection.json          # machine-readable champion decisions
├── selection.md            # ranked human-readable decisions
└── report/                 # publication CSV/Markdown tables and PNG figures
artifacts/
├── interpolation_champion/ # deployable model, card, thresholds, manifest, OOF predictions
└── extrapolation_champion/ # deployable model, card, thresholds, manifest, OOF predictions
```

Generated results and models are intentionally ignored by Git; only `.gitkeep` placeholders are
versioned.

## Inference

```bash
python fable_predict.py \
  --input ../new_conditions.csv \
  --output predictions.csv \
  --champion extrapolation
```

Use `--artifact /path/to/artifact` to override the standard artifact directory. The output contains
one row per input row, class-1 probability, all four threshold decisions, paper identity when
derivable, and model/fingerprint metadata.

Evaluation artifacts are unbiased comparison evidence, not trained production models. Refit
artifacts are trained production models, not unbiased evaluation evidence. Interpolation results
must never be presented as unseen-paper generalization.
