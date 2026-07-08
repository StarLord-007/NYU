# Fable Classifiers BT

This folder contains the five-model Fable ignition classifier benchmark with bootstrap audit outputs. It compares:

- XGBoost
- KNN
- Decision Tree
- MLP
- SVM with RBF kernel

The current full audit was run on `../Microgravity_Database.csv` with `1000` bootstrap resamples.

```powershell
..\.venv\Scripts\python.exe fable_compare.py --data ..\Microgravity_Database.csv --bootstrap-iters 1000
```

## Goal

The benchmark separates two different questions:

1. Interpolation: how well a model performs when train and test rows come from the same overall data distribution.
2. Extrapolation: how well a model transfers to unseen papers.

Each objective has two strategies:

- Interpolation:
  - repeated random stratified 80/20 hold-out
  - stratified 5-fold cross-validation
- Extrapolation:
  - repeated grouped cross-validation, holding out whole papers
  - Leave-One-Paper-Out, holding out each paper once

For each strategy, the audit reports the original non-bootstrap result and a bootstrap confidence interval.

## Code Files

### `fable_common.py`

Shared data loading and feature engineering.

Main responsibilities:

- Reads and cleans the raw microgravity database.
- Canonicalizes paper IDs so rows from the same physical paper stay grouped.
- Parses numerical physics fields such as oxygen fraction, pressure, flow velocity, gravity, ignition power, and dimensions.
- Normalizes categorical fields such as geometry, facility, ignition method, material family, flow direction, and gravity regime.
- Defines feature sets:
  - `all`: all engineered features
  - `physics`: only features expected to transfer across papers
- Defines sample weighting utilities:
  - paper weights
  - class weights
  - combined paper x class weights

Design choice: paper identity is treated as a first-class audit object. Any extrapolation protocol must hold out entire canonical papers.

### `fable_train.py`

XGBoost training pipeline.

Main responsibilities:

- Defines `FableConfig`.
- Builds the XGBoost-based `FableModel`.
- Implements validation-fold threshold optimization.
- Supports paper weighting, class weighting, group-aware hyperparameter presets, focal loss, monotone oxygen constraints, and paper-bagging.

Design choice: thresholds are optimized only on validation folds, never on test folds. This avoids leaking test labels into MCC, F1, or balanced-accuracy thresholds.

### `fable_models.py`

Model-agnostic benchmark core.

Main responsibilities:

- Wraps sklearn models behind the same `.fit(X, y, papers)` and `.predict_proba(X)` interface used by XGBoost.
- Defines `ModelSpec`.
- Provides shared group-aware hyperparameter search.
- Provides benchmark loops for grouped CV and LOPO.
- Provides shared summary and plotting helpers.

Design choice: all models use the same cleaned data, feature set, split definitions, metrics, and threshold logic.

Weighting note:

- XGBoost, Decision Tree, and SVM can consume sample weights during fit.
- KNN and MLP cannot consume fit-time sample weights in sklearn, so equivalent weights are applied during threshold optimization instead.

### `fable_knn.py`

KNN model entry point.

Champion parameters:

```python
{"n_neighbors": 25, "weights": "distance", "p": 1, "leaf_size": 30}
```

Run alone:

```powershell
..\.venv\Scripts\python.exe fable_knn.py --data ..\Microgravity_Database.csv
```

### `fable_mlp.py`

MLP model entry point.

Champion parameters:

```python
{
    "hidden_layer_sizes": (64, 32),
    "alpha": 1e-3,
    "learning_rate_init": 1e-3,
    "activation": "relu",
    "batch_size": 64,
}
```

Run alone:

```powershell
..\.venv\Scripts\python.exe fable_mlp.py --data ..\Microgravity_Database.csv
```

### `fable_svm.py`

SVM model entry point.

Champion parameters:

```python
{"C": 3.0, "gamma": "scale", "shrinking": True}
```

The implementation uses `CalibratedClassifierCV` around an RBF `SVC` so it can expose probabilities through `.predict_proba`.

Run alone:

```powershell
..\.venv\Scripts\python.exe fable_svm.py --data ..\Microgravity_Database.csv
```

### `fable_eval.py`

Older/general evaluation harness for XGBoost configuration studies.

It benchmarks many XGBoost design choices such as paper weighting, class weighting, focal loss, group-tuned hyperparameters, physics-only features, monotone oxygen constraints, and paper-bagging. It is useful for method development, while `fable_compare.py` is the current five-classifier audit entry point.

### `fable_compare.py`

Current main audit script for the five-classifier benchmark.

Main responsibilities:

- Registers all five champion models.
- Runs both interpolation strategies.
- Runs both extrapolation strategies.
- Computes paired Wilcoxon tests on identical grouped-CV folds.
- Computes bootstrap confidence intervals.
- Writes CSV, JSON, Markdown, and figure outputs into `compare_outputs/`.

Useful commands:

```powershell
# Full audit with 1000 bootstrap resamples
..\.venv\Scripts\python.exe fable_compare.py --data ..\Microgravity_Database.csv --bootstrap-iters 1000

# Smoke test: 1 seed, no LOPO
..\.venv\Scripts\python.exe fable_compare.py --data ..\Microgravity_Database.csv --quick

# Full non-bootstrap audit
..\.venv\Scripts\python.exe fable_compare.py --data ..\Microgravity_Database.csv --no-bootstrap
```

## Metrics

The audit reports:

- ROC-AUC
- PR-AUC
- MCC
- F1
- balanced accuracy

For thresholded metrics, thresholds are tuned on inner validation folds and then frozen before scoring the outer test fold.

## Bootstrapping

Bootstrap outputs use 95% confidence intervals.

For fold-based strategies:

- random stratified 80/20
- stratified 5-fold
- repeated grouped CV

the bootstrap resamples folds within each model.

For LOPO:

- the bootstrap resamples whole papers, not rows.

This is intentional because LOPO estimates transfer to new papers; row-level resampling would understate uncertainty.

## Result Files

All current outputs live in `compare_outputs/`.

### Main summaries

#### `comparison_summary.md`

Human-readable audit report. It contains:

- non-bootstrap interpolation tables
- bootstrap interpolation tables
- non-bootstrap extrapolation tables
- bootstrap extrapolation tables
- LOPO pooled results
- pairwise grouped-CV Wilcoxon tests

Start here when reading the results.

#### `comparison_summary.json`

Machine-readable version of the audit summary. It includes:

- row and paper counts
- all non-bootstrap summaries
- all bootstrap summaries
- LOPO pooled metrics
- pairwise tests

### Interpolation outputs

#### `interpolation_random_folds.csv`

Per-model fold results for random stratified 80/20 hold-out.

Important columns:

- `method`
- `seed`
- `fold`
- `roc_auc`
- `pr_auc`
- `mcc`
- `f1`
- `balanced_accuracy`
- threshold columns such as `thr_mcc`

#### `interpolation_random_summary.csv`

Mean and standard deviation summary for random stratified 80/20 hold-out.

#### `interpolation_random_bootstrap_summary.csv`

Bootstrap mean and 95% CI for random stratified 80/20 hold-out.

Columns:

- `strategy`
- `method`
- `metric`
- `mean`
- `ci_low`
- `ci_high`
- `n`
- `bootstrap_iterations`

#### `interpolation_stratified_folds.csv`

Per-model fold results for stratified 5-fold CV.

#### `interpolation_stratified_summary.csv`

Mean and standard deviation summary for stratified 5-fold CV.

#### `interpolation_stratified_bootstrap_summary.csv`

Bootstrap mean and 95% CI for stratified 5-fold CV.

### Extrapolation outputs

#### `extrapolation_grouped_folds.csv`

Per-model results for repeated `StratifiedGroupKFold`, where each outer test fold holds out whole papers.

This is the primary grouped extrapolation table used for pairwise tests.

#### `extrapolation_grouped_summary.csv`

Mean and standard deviation summary for repeated grouped CV.

#### `extrapolation_grouped_bootstrap_summary.csv`

Bootstrap mean and 95% CI for repeated grouped CV.

#### `lopo_per_paper.csv`

Per-paper LOPO results.

Important columns:

- `method`
- `paper_id`
- `paper_label`
- `n_rows`
- `ignition_rate`
- `auc`

Some papers may have undefined within-paper AUC if they contain only one class.

#### `lopo_pooled.json`

Pooled LOPO ROC-AUC and PR-AUC for each model.

#### `lopo_pooled_bootstrap_summary.csv`

Paper-cluster bootstrap mean and 95% CI for pooled LOPO ROC-AUC and PR-AUC.

### Statistical comparison output

#### `pairwise_tests.csv`

Paired Wilcoxon signed-rank tests over identical grouped-CV folds.

Important columns:

- `model_a`
- `model_b`
- `n_folds`
- `delta_roc_auc`
- `wilcoxon_p_roc_auc`
- `delta_pr_auc`
- `wilcoxon_p_pr_auc`
- `delta_mcc`
- `wilcoxon_p_mcc`
- `delta_balanced_accuracy`
- `wilcoxon_p_balanced_accuracy`

Design choice: models are compared on identical `(seed, fold)` partitions, so fold difficulty is paired out of the comparison.

### Figures

#### `fig_compare_grouped_cv.png`

Grouped-CV fold distribution plot for the five models.

#### `fig_compare_lopo_pooled.png`

Pooled LOPO ROC-AUC bar chart.

#### `fig_compare_lopo_per_paper.png`

Per-paper LOPO AUC distribution by model.

#### `fig_compare_interpolation.png`

Legacy interpolation comparison figure retained from previous runs.

## Current Results Summary

The current full run used:

- data: `../Microgravity_Database.csv`
- rows: `4366`
- canonical papers: `86`
- bootstrap resamples: `1000`

### Interpolation: random stratified 80/20

Top ROC-AUC:

- XGBoost: `0.896 [0.889, 0.904]`
- MLP: `0.849 [0.816, 0.872]`
- SVM (RBF): `0.834 [0.828, 0.840]`

### Interpolation: stratified 5-fold

Top ROC-AUC:

- XGBoost: `0.898 [0.893, 0.904]`
- MLP: `0.860 [0.840, 0.880]`
- Decision Tree: `0.816 [0.807, 0.826]`

### Extrapolation: repeated grouped CV

Top ROC-AUC:

- XGBoost: `0.660 [0.625, 0.695]`
- MLP: `0.610 [0.576, 0.644]`
- SVM (RBF): `0.592 [0.533, 0.642]`

### Extrapolation: Leave-One-Paper-Out pooled

Pooled ROC-AUC:

- XGBoost: `0.677 [0.612, 0.752]`
- MLP: `0.637 [0.574, 0.719]`
- SVM (RBF): `0.604 [0.518, 0.702]`
- Decision Tree: `0.593 [0.513, 0.683]`
- KNN: `0.583 [0.493, 0.682]`

## Interpretation

Interpolation scores are high, especially for XGBoost. Extrapolation scores are much lower, which indicates meaningful paper-to-paper domain shift.

XGBoost is the strongest model across all four strategies. The advantage is clearest in grouped CV, where XGBoost beats Decision Tree and KNN on ROC-AUC with paired Wilcoxon p-values below `0.01`. Against MLP and SVM, XGBoost still has higher mean ROC-AUC, but ROC-AUC p-values are not below `0.05` in the grouped-CV pairwise table.

The LOPO bootstrap intervals are wide because the uncertainty unit is the paper. That is expected and is more honest for the extrapolation objective.

## Recommended Reporting

For papers, presentations, or audit notes, report both:

- interpolation results, to show same-distribution discrimination
- extrapolation results, to show unseen-paper transfer

Do not present random or stratified row-level interpolation as evidence of new-paper generalization. Use grouped CV and LOPO for that claim.

