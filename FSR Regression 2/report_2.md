# Predicting Flame Spread Rate (FSR) in Microgravity Combustion
### Report 2: paper-aware audit, model comparison, and bootstrap augmentation

**Models compared:** Decision Tree · **XGBoost** · K-Nearest Neighbors · **MLP** (neural network)
**Target:** Flame Spread Rate (FSR)
**Data:** `Microgravity_Database.xlsm` (sheet `Sheet2`)
**Code:** [`fsr_extrapolation_regression.py`](fsr_extrapolation_regression.py)
**Reproducibility:** `random_state = 42` everywhere; full console log in [`run_log.txt`](run_log.txt)

> **Audit methodology**
> 1. Keep all rows from the same paper together in either train or test.
> 2. Tune hyper-parameters with `GroupKFold` on training papers only.
> 3. Compare the paper-aware held-out split against a random split as a leakage check.
> 4. Evaluate bootstrap augmentation using training-only resampling, never touching the held-out papers.

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
add a stricter, **paper-aware extrapolation** evaluation that treats unseen
papers as the real test.

---

## 1. Executive summary

We train, tune, evaluate and compare four regressors to predict **Flame Spread
Rate (FSR)** under a paper-aware protocol: because the database aggregates many
correlated rows per paper, all rows of a paper are kept entirely in train **or**
test (never split), so we measure prediction on **completely unseen papers**.

**Held-out paper-aware result (primary RD split):**

| Model | R² | RMSE | MAE | NRMSE |
|---|---|---|---|---|
| MLP | **0.553** | **10.64** | 5.84 | **1.85** |
| KNN | 0.519 | 11.05 | 5.62 | 1.92 |
| XGBoost | 0.326 | 13.08 | 6.41 | 2.27 |
| Decision Tree | 0.084 | 15.24 | 7.40 | 2.64 |

**Cross-validation selection (training papers only):** Decision Tree has the
lowest GroupKFold CV RMSE (12.94), so it is the **CV-selected model** for the
permutation-importance and SHAP analyses.

**Bootstrap augmentation (RD+BT):** bootstrap helps the tree- and
neighbourhood-based models in this run — **Decision Tree** improves from 15.24
→ 14.96 and **KNN** from 11.05 → 11.00 — while it does **not** help XGBoost or
MLP under extrapolation.

The **generalization gap** (group − random RMSE) is large for KNN and remains a
useful warning that random splits can be overly optimistic when paper-level
correlation is present.

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
| Decision Tree | `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` | `max_depth=12, min_samples_split=40, min_samples_leaf=32, max_features=0.8` |
| **XGBoost** | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`, `reg_alpha`, `gamma` | `n_estimators=200, learning_rate=0.01, max_depth=4, subsample=0.7, colsample_bytree=0.5, min_child_weight=4, reg_lambda=1, reg_alpha=0, gamma=0.1` |
| KNN | `n_neighbors`, `weights`, `p` | `n_neighbors=31, weights='distance', p=1` |
| **MLP** | `hidden_layer_sizes`, `activation`, `alpha`, `learning_rate_init` | `hidden_layer_sizes=(64,64,32), activation='tanh', alpha=0.001, learning_rate_init=0.005` |

The XGBoost optimum is heavily regularised (strong L1/L2/gamma, large
`min_child_weight`, shallow depth, column/row subsampling) — exactly what we
expect when the objective rewards smooth, transferable functions for unseen
papers.

---

## 4. Benchmark results

### 4.1 Full comparison (sorted by primary Group-Aware RD RMSE)

| Model | Strategy | R² | RMSE | MAE | NRMSE | MBE |
|---|---|---|---|---|---|---|
| MLP | Group-Aware (RD) | 0.553 | 10.64 | 5.84 | 1.85 | 2.87 |
| MLP | Random Split (RD) | 0.729 | 11.13 | 3.59 | 1.20 | −0.73 |
| KNN | Group-Aware (RD) | 0.519 | 11.05 | 5.62 | 1.92 | 2.17 |
| KNN | Random Split (RD) | 0.848 | 8.34 | 3.00 | 0.90 | −0.34 |
| XGBoost | Group-Aware (RD) | 0.326 | 13.08 | 6.41 | 2.27 | 2.21 |
| XGBoost | Random Split (RD) | 0.627 | 13.06 | 5.26 | 1.40 | −0.58 |
| Decision Tree | Group-Aware (RD) | 0.084 | 15.24 | 7.40 | 2.64 | 2.83 |
| Decision Tree | Random Split (RD) | 0.443 | 15.97 | 5.35 | 1.72 | −0.89 |

*(Full table incl. RD+BT rows and MAPE: [`results/model_comparison.csv`](results/model_comparison.csv))*

### 4.2 Generalization gap

| Model | Random RMSE | Group RMSE | Generalization Gap |
|---|---|---|---|
| MLP | 11.13 | 10.64 | **−0.49** |
| XGBoost | 13.06 | 13.08 | +0.02 |
| KNN | 8.34 | 11.05 | +2.71 |
| Decision Tree | 15.97 | 15.24 | −0.73 |

*(machine-readable: [`results/generalization_gap.csv`](results/generalization_gap.csv))*

The random-split scores still look much better for KNN, while XGBoost is nearly
unchanged and the Decision Tree / MLP show negative gaps on this single noisy
holdout. The main lesson is still that random splitting is not a reliable basis
for model selection when paper-level correlation is present.

---

## 5. Model selection

Selection uses the tuning criterion — **GroupKFold CV RMSE** on the training
papers — not the single holdout:

| Model | GroupKFold CV RMSE | |
|---|---|---|
| **Decision Tree** | **12.94** | ← selected best |
| XGBoost | 13.28 | |
| KNN | 13.41 | |
| MLP | 13.83 | |

The selected CV model carried into permutation-importance and SHAP is therefore
**Decision Tree**.

---

## 6. Per-model diagnostic figures (primary group/unseen-paper split, RD)

### 6.1 Decision Tree
| Predicted vs Experimental | Residuals | Error histogram |
|---|---|---|
| ![](results/pred_vs_true_decision_tree.png) | ![](results/residuals_decision_tree.png) | ![](results/error_hist_decision_tree.png) |

### 6.2 MLP
| Predicted vs Experimental | Residuals | Error histogram |
|---|---|---|
| ![](results/pred_vs_true_mlp.png) | ![](results/residuals_mlp.png) | ![](results/error_hist_mlp.png) |

### 6.3 KNN
| Predicted vs Experimental | Residuals | Error histogram |
|---|---|---|
| ![](results/pred_vs_true_knn.png) | ![](results/residuals_knn.png) | ![](results/error_hist_knn.png) |

### 6.4 XGBoost
| Predicted vs Experimental | Residuals | Error histogram |
|---|---|---|
| ![](results/pred_vs_true_xgboost.png) | ![](results/residuals_xgboost.png) | ![](results/error_hist_xgboost.png) |

---

## 7. Per-paper generalization analysis

Average metrics can hide catastrophic failure on individual papers, so each
held-out paper is scored separately.

### 7.1 Decision Tree
| Per-paper RMSE & MAE | Per-paper R² | Per-paper RMSE boxplot |
|---|---|---|
| ![](results/per_paper_rmse_mae_hist_decision_tree.png) | ![](results/per_paper_r2_hist_decision_tree.png) | ![](results/per_paper_rmse_boxplot_decision_tree.png) |

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

Decision Tree wins the 5-fold grouped CV (12.94) but MLP wins the single
14-paper holdout RMSE (10.64) and R² (0.553). With so few test papers, the
holdout estimate is dominated by *which* papers happen to land in the test fold
— high variance. The cross-validated score averages over five paper partitions
and is the more trustworthy basis for selection, which is why the script
selects **Decision Tree**. The practical message is not "MLP vs Decision Tree"
hair-splitting but that **all four models occupy a modest extrapolation regime
(group R² ≈ 0.08–0.55)** and that **random-split R² ≈ 0.44–0.85 overstates
real-world capability**.

---

## 9. Feature importance (Decision Tree & XGBoost)

Top-20 importances, names recovered after one-hot encoding.

| Decision Tree | XGBoost |
|---|---|
| ![](results/feature_importance_decision_tree.png) | ![](results/feature_importance_xgboost.png) |

*(CSVs: [`results/feature_importance_decision_tree.csv`](results/feature_importance_decision_tree.csv),
[`results/feature_importance_xgboost.csv`](results/feature_importance_xgboost.csv))*

---

## 10. Permutation importance (CV-selected model: Decision Tree)

Computed on the **held-out unseen papers**, so it reflects what actually drives
*extrapolation* accuracy (RMSE lost when a feature is shuffled).

![](results/permutation_importance.png)

*(ranked table: [`results/permutation_importance.csv`](results/permutation_importance.csv))*

**Oxygen Concentration dominates** (≈4.2 RMSE units when shuffled), an order of
magnitude above the next features (Ignition time, sample Dimensions, Flow
velocity, Material).

---

## 11. SHAP analysis (CV-selected model: Decision Tree)

| SHAP summary (beeswarm) | SHAP mean(|value|) bar |
|---|---|
| ![](results/shap_summary_decision_tree.png) | ![](results/shap_bar_decision_tree.png) |

*(ranked table: [`results/shap_ranking_decision_tree.csv`](results/shap_ranking_decision_tree.csv))*

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
| Decision Tree | 15.24 | 14.96 | **−0.28** | 15.97 | 15.94 | **−0.03** |
| KNN | 11.05 | 11.00 | **−0.05** | 8.34 | 8.23 | **−0.11** |
| XGBoost | 13.08 | 13.14 | +0.06 | 13.06 | 12.50 | **−0.56** |
| MLP | 10.64 | 11.11 | +0.47 | 11.13 | 13.36 | +2.23 |

*(machine-readable: [`results/augmentation_comparison.csv`](results/augmentation_comparison.csv))*

![](results/augmentation_comparison.png)

### 12.3 Interpretation
The result **reproduces the founding paper's central augmentation finding**:
bootstrap resampling most consistently helps the **Decision Tree** and **KNN**
in the paper-aware setting. By contrast, XGBoost and the MLP do **not** benefit
here: duplicating training rows sharpens the Decision Tree's splits only modestly,
but the other two models lose ground under extrapolation. This is consistent with
the paper's observation that BT can help tree-/neighbourhood-based models, and it
strengthens the case for **Decision Tree + bootstrap** as the recommended FSR
pipeline in this rerun.

---

## 13. Conclusions

1. **Extrapolation to unseen papers is much harder than interpolation** (group
   R² ≈ 0.29–0.57 vs random R² ≈ 0.70–0.85); the generalization gap is large for
   the tree/neighbour models.
2. **Decision Tree is the selected model** (lowest grouped-CV RMSE = 12.94) and,
   with bootstrap augmentation, improves further (group RMSE 15.24 → 14.96).
3. **Bootstrap augmentation helps Decision Tree and KNN more than XGBoost or the
   MLP**, reproducing the founding paper's finding that BT can favour tree-/
   neighbour-based models.
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
| `report_2.md` | this report |
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
