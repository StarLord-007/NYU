# Predicting Flame Spread Rate (FSR) in Microgravity Combustion
### A paper-aware (extrapolation-first) machine-learning study, with bootstrap data augmentation

**Models compared:** Decision Tree · **XGBoost** · K-Nearest Neighbors · **MLP** (neural network)
**Target:** Flame Spread Rate (FSR)
**Data:** `Microgravity_Database.xlsm` (sheet `Sheet2`)
**Code:** [`fsr_extrapolation_regression.py`](fsr_extrapolation_regression.py)
**Reproducibility:** `random_state = 42` everywhere; full console log in [`run_log.txt`](run_log.txt)

> **What changed in this revision**
> 1. The Gradient Boosting model has been **replaced by XGBoost**.
> 2. A fourth model, a **Multi-Layer Perceptron (MLP)** neural network, has been added.
> 3. **Bootstrap resampling (BT)** data augmentation has been added, following the founding paper of this work (see §0 and §12).

---

## 0. Founding paper

This study operationalises the methodology of:

> Jose Rivera, Daniel San Martin, Carlos Fernandez-Pello, Michael J. Gollner,
> Augustin Guibaud, Sandra Olson, Dennis Stocker — *"Using Data Categorization and
> Augmentation Strategies to improve Machine Learning Frameworks for Flame Spread
> over Electrical Wires."*

That paper compiles a literature FSR dataset, trains non-linear ML models
(including **Decision Trees, Gradient Boosting, KNN and MLP**) and shows that
**bootstrap resampling** is the most consistent data-augmentation strategy for
tree- and neighbourhood-based models, pushing R² above 0.9 under random
cross-validation. We adopt its model families (with **XGBoost** standing in for
the paper's Gradient Boosting) and its **bootstrap augmentation** recipe, but we
add a stricter, **paper-aware extrapolation** evaluation that the paper itself
flags as the harder, more honest test of transferability to unseen sources.

---

## 1. Executive summary

We train, tune, evaluate and compare four regressors to predict **Flame Spread
Rate (FSR)** under a paper-aware protocol: because the database aggregates many
correlated rows per paper, all rows of a paper are kept entirely in train **or**
test (never split), so we measure prediction on **completely unseen papers**.

**Headline result (primary, group-aware / unseen papers, raw data "RD"):**

| Model | R² | RMSE | MAE | NRMSE |
|---|---|---|---|---|
| MLP | 0.568 | **10.47** | 5.21 | 1.82 |
| KNN | 0.519 | 11.05 | 5.62 | 1.92 |
| Decision Tree | 0.346 | 12.88 | 6.00 | 2.23 |
| XGBoost | 0.294 | 13.39 | 6.12 | 2.32 |

**Best model by the tuning/selection criterion (GroupKFold CV RMSE): XGBoost**
(CV RMSE = 12.54, the lowest of the four). Note the tension between "lowest CV
RMSE" (XGBoost) and "lowest single-holdout RMSE" (MLP) — with only 14 test
papers the single holdout is noisy, so selection is anchored to the more stable
5-fold grouped CV (see §8 and §11).

**Bootstrap augmentation (RD+BT):** consistent with the founding paper, BT helps
the tree- and neighbourhood-based models — **XGBoost** (group RMSE 13.39 → 12.75,
Δ = −0.64) and **KNN** (11.05 → 10.99) — while it does **not** help the Decision
Tree or the MLP under extrapolation (see §12).

The **generalization gap** (group − random RMSE) remains large for the tree and
neighbour models, confirming that random-split numbers are over-optimistic.

---

## 2. The dataset

| Property | Value |
|---|---|
| Rows loaded from `Sheet2` | 5,118 |
| Rows with a valid numeric FSR (kept) | **2,605** |
| Rows removed (missing / non-numeric FSR) | 2,513 |
| Detected target column | `FSR (Flame Spread Rate)` |
| Detected paper-grouping column | `Article (MLA)` |
| Unique papers | **69** |
| Numeric features | 8 |
| Categorical features | 6 |

**Samples per paper:** mean 37.8, median 28, min 4, max 123 — large, correlated
per-paper blocks, which is exactly why paper-aware evaluation is essential.

---

## 3. Models and tuned hyper-parameters

All four models share one preprocessing pipeline (median impute + `StandardScaler`
for numeric; most-frequent impute + `OneHotEncoder(handle_unknown="ignore")` for
categorical) and are tuned by **`RandomizedSearchCV` with `GroupKFold`** (paper
groups) and `scoring="neg_root_mean_squared_error"` — never random CV.

| Model | Search space (tuned) | Selected best hyper-parameters |
|---|---|---|
| Decision Tree | `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` | `max_depth=12, min_samples_split=20, min_samples_leaf=1, max_features='log2'` |
| **XGBoost** | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`, `reg_alpha`, `gamma` | `n_estimators=300, learning_rate=0.02, max_depth=4, subsample=0.9, colsample_bytree=0.7, min_child_weight=10, reg_lambda=5, reg_alpha=1, gamma=1` |
| KNN | `n_neighbors`, `weights`, `p` | `n_neighbors=31, weights='distance', p=1` |
| **MLP** | `hidden_layer_sizes`, `activation`, `alpha`, `learning_rate_init` | `hidden_layer_sizes=(64,64,32), activation='tanh', alpha=0.1, learning_rate_init=0.005` |

The XGBoost optimum is heavily regularised (strong L1/L2/gamma, large
`min_child_weight`, shallow depth, column/row subsampling) — exactly what we
expect when the objective rewards smooth, transferable functions for unseen
papers.

---

## 4. Benchmark results

### 4.1 Full comparison (sorted by primary Group-Aware RD RMSE)

| Model | Strategy | R² | RMSE | MAE | NRMSE | MBE |
|---|---|---|---|---|---|---|
| MLP | Group-Aware (RD) | 0.568 | 10.47 | 5.21 | 1.82 | 2.33 |
| MLP | Random Split (RD) | 0.705 | 11.63 | 3.54 | 1.25 | −1.32 |
| KNN | Group-Aware (RD) | 0.519 | 11.05 | 5.62 | 1.92 | 2.17 |
| KNN | Random Split (RD) | 0.848 | 8.34 | 3.00 | 0.90 | −0.34 |
| Decision Tree | Group-Aware (RD) | 0.346 | 12.88 | 6.00 | 2.23 | 2.17 |
| Decision Tree | Random Split (RD) | 0.802 | 9.52 | 3.89 | 1.02 | −0.40 |
| XGBoost | Group-Aware (RD) | 0.294 | 13.39 | 6.12 | 2.32 | 1.84 |
| XGBoost | Random Split (RD) | 0.729 | 11.14 | 4.19 | 1.20 | −0.51 |

*(Full table incl. RD+BT rows and MAPE: [`results/model_comparison.csv`](results/model_comparison.csv))*

### 4.2 Generalization gap

| Model | Random RMSE | Group RMSE | Generalization Gap |
|---|---|---|---|
| MLP | 11.63 | 10.47 | **−1.15** |
| XGBoost | 11.14 | 13.39 | +2.25 |
| KNN | 8.34 | 11.05 | +2.71 |
| Decision Tree | 9.52 | 12.88 | +3.36 |

*(machine-readable: [`results/generalization_gap.csv`](results/generalization_gap.csv))*

The tree/neighbour models (KNN, DT, XGBoost) look far better under random
splitting — the classic paper-overfitting signature. The MLP is the exception
(negative gap) but this is partly an artefact of the noisy 14-paper holdout
(see §8/§11); its CV RMSE is in fact the worst of the four.

---

## 5. Model selection

Selection uses the tuning criterion — **GroupKFold CV RMSE** on the training
papers — not the single holdout:

| Model | GroupKFold CV RMSE | |
|---|---|---|
| **XGBoost** | **12.54** | ← selected best |
| Decision Tree | 13.17 | |
| KNN | 13.62 | |
| MLP | 15.68 | |

The selected best model carried into permutation-importance and SHAP is
therefore **XGBoost**.

---

## 6. Per-model diagnostic figures (primary group/unseen-paper split, RD)

### 6.1 XGBoost (selected best)
| Predicted vs Experimental | Residuals | Error histogram |
|---|---|---|
| ![](results/pred_vs_true_xgboost.png) | ![](results/residuals_xgboost.png) | ![](results/error_hist_xgboost.png) |

### 6.2 MLP
| Predicted vs Experimental | Residuals | Error histogram |
|---|---|---|
| ![](results/pred_vs_true_mlp.png) | ![](results/residuals_mlp.png) | ![](results/error_hist_mlp.png) |

### 6.3 KNN
| Predicted vs Experimental | Residuals | Error histogram |
|---|---|---|
| ![](results/pred_vs_true_knn.png) | ![](results/residuals_knn.png) | ![](results/error_hist_knn.png) |

### 6.4 Decision Tree
| Predicted vs Experimental | Residuals | Error histogram |
|---|---|---|
| ![](results/pred_vs_true_decision_tree.png) | ![](results/residuals_decision_tree.png) | ![](results/error_hist_decision_tree.png) |

---

## 7. Per-paper generalization analysis

Average metrics can hide catastrophic failure on individual papers, so each
held-out paper is scored separately.

### 7.1 XGBoost (best model)
| Per-paper RMSE & MAE | Per-paper R² | Per-paper RMSE boxplot |
|---|---|---|
| ![](results/per_paper_rmse_mae_hist_xgboost.png) | ![](results/per_paper_r2_hist_xgboost.png) | ![](results/per_paper_rmse_boxplot_xgboost.png) |

The per-paper RMSE distribution is strongly right-skewed: most unseen papers are
predicted well, but a few campaigns fail badly (long upper tail), and some have
negative per-paper R². This heterogeneity is invisible in the aggregate RMSE.

*(per-paper tables for all four models: `results/per_paper_metrics_*.csv`)*

### 7.2 MLP and KNN
| MLP per-paper RMSE/MAE | KNN per-paper RMSE/MAE |
|---|---|
| ![](results/per_paper_rmse_mae_hist_mlp.png) | ![](results/per_paper_rmse_mae_hist_knn.png) |

---

## 8. Why CV-best ≠ holdout-best (interpretation)

XGBoost wins the 5-fold grouped CV (12.54) but MLP wins the single 14-paper
holdout RMSE (10.47). With so few test papers, the holdout estimate is dominated
by *which* papers happen to land in the test fold — high variance. The
cross-validated score averages over five paper partitions and is the more
trustworthy basis for selection, which is why the script selects **XGBoost**. The
practical message is not "MLP vs XGBoost" hair-splitting but that **all four
models occupy a similar, modest extrapolation regime (group R² ≈ 0.29–0.57)** and
that **random-split R² ≈ 0.7–0.85 overstates real-world capability**.

---

## 9. Feature importance (Decision Tree & XGBoost)

Top-20 importances, names recovered after one-hot encoding.

| Decision Tree | XGBoost |
|---|---|
| ![](results/feature_importance_decision_tree.png) | ![](results/feature_importance_xgboost.png) |

*(CSVs: [`results/feature_importance_decision_tree.csv`](results/feature_importance_decision_tree.csv),
[`results/feature_importance_xgboost.csv`](results/feature_importance_xgboost.csv))*

---

## 10. Permutation importance (best model: XGBoost)

Computed on the **held-out unseen papers**, so it reflects what actually drives
*extrapolation* accuracy (RMSE lost when a feature is shuffled).

![](results/permutation_importance.png)

*(ranked table: [`results/permutation_importance.csv`](results/permutation_importance.csv))*

**Oxygen Concentration dominates** (≈4.2 RMSE units when shuffled), an order of
magnitude above the next features (Ignition time, sample Dimensions, Flow
velocity, Material).

---

## 11. SHAP analysis (best model: XGBoost)

| SHAP summary (beeswarm) | SHAP mean(\|value\|) bar |
|---|---|
| ![](results/shap_summary_xgboost.png) | ![](results/shap_bar_xgboost.png) |

*(ranked table: [`results/shap_ranking_xgboost.csv`](results/shap_ranking_xgboost.csv))*

**Top SHAP drivers of FSR:** Oxygen Concentration (mean|SHAP| ≈ 4.4) ≫ Internal
Dimensions, facility/rig terms (Drop Tower, NASA Glenn), Pressure, sample
Dimensions, Flow velocity. The data-driven ranking matches combustion theory:
FSR is governed first by **oxidiser availability/transport** (O₂, then
pressure/flow), then by **fuel thermal/geometric properties** (dimensions,
material), with facility/rig terms capturing apparatus-specific signal.

---

## 12. Data augmentation: Bootstrap resampling (founding paper)

### 12.1 Method
Following Rivera et al. (Sec. 3.3): **bootstrap resampling adds `n=1000` rows
sampled with replacement (seed 42) to the training set.** Bootstrap rows are
exact copies of real observations, so they "preserve the physical admissibility
of the original observations" and densify the empirical distribution (especially
the high-density low-FSR region).

**Critical extrapolation safeguard:** bootstrap rows are drawn **only from the
training split** (only from training papers); the test set is never augmented or
resampled, so the augmentation cannot leak unseen-paper information.

![](results/bootstrap_fsr_distribution.png)

The figure (cf. the paper's Fig. 4) shows the real training FSR distribution and
the bootstrap-augmented one: BT keeps the same shape but increases density,
particularly in the data-rich low-FSR region.

### 12.2 Effect (RD vs RD+BT)

| Model | Group RMSE (RD) | Group RMSE (RD+BT) | Group ΔRMSE | Random RMSE (RD) | Random RMSE (RD+BT) | Random ΔRMSE |
|---|---|---|---|---|---|---|
| XGBoost | 13.39 | 12.75 | **−0.64** | 11.14 | 10.39 | **−0.75** |
| KNN | 11.05 | 10.99 | **−0.06** | 8.34 | 8.23 | **−0.10** |
| Decision Tree | 12.88 | 15.97 | +3.09 | 9.52 | 10.93 | +1.41 |
| MLP | 10.47 | 13.38 | +2.91 | 11.63 | 13.88 | +2.26 |

*(machine-readable: [`results/augmentation_comparison.csv`](results/augmentation_comparison.csv))*

![](results/augmentation_comparison.png)

### 12.3 Interpretation
The result **reproduces the founding paper's central augmentation finding**:
bootstrap resampling most consistently helps **tree- and neighbourhood-based
models** — here **XGBoost** (the selected best model) and **KNN** both improve in
*both* evaluation settings. By contrast, the single Decision Tree and the MLP do
**not** benefit: duplicating training rows sharpens the Decision Tree's
already-overfit splits and re-weights the MLP's loss toward the dense low-FSR
region, hurting unseen-paper generalisation. This is consistent with the paper's
observation that BT's gains are concentrated in GB/KNN-style models, and it
strengthens the case for **XGBoost + bootstrap** as the recommended FSR pipeline.

---

## 13. Conclusions

1. **Extrapolation to unseen papers is much harder than interpolation** (group
   R² ≈ 0.29–0.57 vs random R² ≈ 0.70–0.85); the generalization gap is large for
   the tree/neighbour models.
2. **XGBoost is the selected model** (lowest grouped-CV RMSE = 12.54) and, with
   bootstrap augmentation, improves further (group RMSE 13.39 → 12.75).
3. **Bootstrap augmentation helps XGBoost and KNN but not the single Decision
   Tree or the MLP**, reproducing the founding paper's finding that BT favours
   tree-/neighbour-based models.
4. **Oxygen concentration is the dominant, physically-consistent driver** across
   feature importance, permutation importance and SHAP.
5. **Per-paper analysis is essential**: a few campaigns are predicted very poorly
   even by the best model.

### Limitations & future work
* First-number parsing of unit-laden columns is transparent but not
  unit-normalised.
* Only 69 papers / 14 test papers → noisy holdout; leave-one-paper-out CV and
  more sources would tighten estimates.
* The paper's other augmentations (Noise Injection, GANs) and a log-transformed,
  non-negative-constrained FSR target are natural next steps.

---

## 14. Reproducibility & how to run

```bash
pip install pandas scikit-learn numpy matplotlib joblib openpyxl shap xgboost
python "FSR Regression/fsr_extrapolation_regression.py"                 # full run (RD + RD+BT)
python "FSR Regression/fsr_extrapolation_regression.py" --no-shap       # skip SHAP
python "FSR Regression/fsr_extrapolation_regression.py" --bootstrap-n 0 # disable bootstrap
```

* `random_state = 42` for every split, search, model and bootstrap draw.
* The script auto-locates `Microgravity_Database.xlsm` and writes all artefacts
  to `FSR Regression/results/` by default.

### Files in this folder
| File | Contents |
|---|---|
| `fsr_extrapolation_regression.py` | the complete, commented, runnable script |
| `REPORT.md` | this report |
| `run_log.txt` | full console output of the benchmarked run |
| `results/model_comparison.csv` | metrics for all 4 models × 4 conditions (RD/RD+BT × random/group) |
| `results/generalization_gap.csv` | random vs group RMSE + gap |
| `results/augmentation_comparison.csv` | bootstrap effect (RD vs RD+BT) |
| `results/metrics.json` | machine-readable summary |
| `results/pred_vs_true_*.png`, `residuals_*.png`, `error_hist_*.png` | per-model diagnostics |
| `results/per_paper_*` | per-paper extrapolation analysis (CSV + plots) |
| `results/feature_importance_*` | Decision Tree & XGBoost importances |
| `results/permutation_importance.*` | best-model permutation importance |
| `results/shap_summary_*`, `shap_bar_*`, `shap_ranking_*` | best-model SHAP analysis |
| `results/bootstrap_fsr_distribution.png`, `augmentation_comparison.png` | bootstrap augmentation figures |
| `results/best_decision_tree.joblib`, `best_xgboost.joblib`, `best_knn.joblib`, `best_mlp.joblib` | saved fitted pipelines |
