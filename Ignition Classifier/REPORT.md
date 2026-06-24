# Predicting Ignition (Yes/No) in Microgravity Combustion
### A paper-aware (extrapolation-first) machine-learning study

**Models compared:** Decision Tree · **XGBoost** · K-Nearest Neighbors · **MLP** (neural network)
**Target:** Ignition (Yes/No) — a **binary classification** problem
**Data:** `Microgravity_Database.xlsm` (sheet `Sheet2`)
**Code:** [`ignition_classification.py`](ignition_classification.py)
**Reproducibility:** `random_state = 42` everywhere; full console log in [`run_log.txt`](run_log.txt)

> **What changed in this revision**
> 1. The Gradient Boosting model has been **replaced by XGBoost**.
> 2. A fourth model, a **Multi-Layer Perceptron (MLP)** neural network, has been added.
> 3. The folder has been renamed to `ignition classifier`.

---

## 0. Regression vs classification — an important note

The companion folder is named `FSR Regression` and this one mirrors it with the
**same four model families** and the **same paper-aware, extrapolation-first
methodology**. However, the ignition target is **binary** (`Yes`/`No`), so the
scientifically correct task is **classification**, not regression. Forcing a
regressor onto a 0/1 label would yield meaningless metrics (e.g. R² on a Bernoulli
target). We therefore use the **classifier** variants
(`DecisionTreeClassifier`, `XGBClassifier`, `KNeighborsClassifier`,
`MLPClassifier`) and report classification metrics (ROC-AUC, PR-AUC, F1, balanced
accuracy, MCC, …). Bootstrap augmentation was requested for the FSR task only and
is not applied here.

---

## 1. Executive summary

We train, tune, evaluate and compare four classifiers to predict whether a
microgravity combustion sample **ignites**, keeping all rows of any paper together
(train **or** test, never split) so we measure prediction on **completely unseen
papers**.

**Headline result (primary, group-aware / unseen papers, sorted by ROC-AUC):**

| Rank | Model | ROC-AUC | PR-AUC | Bal. Acc | F1 | MCC |
|---|---|---|---|---|---|---|
| 1 | **XGBoost** | **0.727** | 0.880 | 0.596 | 0.844 | 0.274 |
| 2 | **MLP** | 0.691 | 0.858 | 0.561 | 0.806 | 0.147 |
| 3 | **KNN** | 0.650 | 0.805 | 0.588 | 0.787 | 0.184 |
| 4 | **Decision Tree** | 0.472 | 0.726 | 0.474 | 0.787 | −0.077 |

**Best model by the tuning/selection criterion (GroupKFold CV ROC-AUC): XGBoost**
(0.681) — and XGBoost is also best on the held-out unseen papers (0.727), so the
selection is unambiguous this time.

Two findings stand out:

1. **Extrapolation is much harder than interpolation.** Group-aware ROC-AUC
   (0.47–0.73) is far below random-split ROC-AUC (0.79–0.91); the generalization
   gap is **+0.17 to +0.32 AUC**.
2. **The Decision Tree fails dramatically out-of-sample** — group ROC-AUC 0.472
   (below chance) and MCC < 0, despite a 0.79 random-split AUC. This is textbook
   paper-specific over-fitting that a random split hides.

---

## 2. The dataset

| Property | Value |
|---|---|
| Rows loaded from `Sheet2` | 5,118 |
| Rows with a valid Yes/No label (kept) | **5,093** |
| Detected target column | `Ignition (Yes/No)` |
| Detected paper-grouping column | `Article (MLA)` |
| Unique papers | **93** |
| Numeric / categorical features | 8 / 6 |

**Class balance (imbalanced):** Ignition (1) = 3,887 (**76.3%**); No-ignition (0)
= 1,206 (23.7%). Because a trivial "always ignite" guess already scores ~76%
accuracy, the honest metrics are **ROC-AUC, PR-AUC, balanced accuracy and MCC**,
which a majority-class guesser cannot fake. These are also what we tune/select on.

**Samples per paper:** mean 54.8, median 33, min 4, max 309.

The realised group split: **train = 4,047 rows (74 papers)**, **test = 1,046 rows
(19 papers)**, test prevalence 72.2% ignition, no shared papers.

---

## 3. Design choices and leakage prevention

* **Automatic, binary-aware target detection** selects `Ignition (Yes/No)` (a
  name match plus ≥80% values mapping to {0,1}), distinguishing it from the
  *input* columns `Ignition method/power/time`.
* **Leakage prevention:** the post-ignition outcomes `FSR`, `Flame Length`,
  `HRR`, `Smoke/Aerosols` are removed (they are observed only *after* ignition),
  along with free-text notes (`Info`) and paper fingerprints (`Authors`, `DOI`,
  `Article`). The ignition **stimulus** (`method`, `power`, `time`) is **kept** as
  a valid predictor.
* **Shared preprocessing**: median impute + `StandardScaler` (numeric);
  most-frequent impute + `OneHotEncoder(handle_unknown="ignore")` (categorical).
* **Two strategies:** stratified random split (baseline/interpolation) and
  `GroupShuffleSplit` by paper (primary/extrapolation, with an assertion of no
  shared papers).
* **Tuning always via `GroupKFold`** with `scoring="roc_auc"`. The Decision Tree
  tunes `class_weight`; XGBoost tunes `scale_pos_weight`; KNN/MLP rely on AUC
  tuning and balanced metrics for the imbalance.

---

## 4. Models and tuned hyper-parameters

| Model | Search space | Selected best hyper-parameters |
|---|---|---|
| Decision Tree | depth/leaf/split/features, `class_weight` | `max_depth=12, min_samples_split=40, min_samples_leaf=4, max_features='sqrt', class_weight=None` |
| **XGBoost** | trees/shrinkage/depth/subsample/colsample/min_child/reg_λ/α/γ, `scale_pos_weight` | `n_estimators=800, learning_rate=0.2, max_depth=3, subsample=0.8, colsample_bytree=0.5, min_child_weight=2, reg_lambda=0.5, gamma=0.1, scale_pos_weight=2` |
| KNN | `n_neighbors`, `weights`, `p` | `n_neighbors=11, weights='uniform', p=1` |
| **MLP** | `hidden_layer_sizes`, `activation`, `alpha`, `learning_rate_init` | `hidden_layer_sizes=(64,64,32), activation='tanh', alpha=0.1, learning_rate_init=0.005` |

---

## 5. Benchmark results

### 5.1 Full comparison (sorted by Group ROC-AUC)

| Model | Strategy | Accuracy | Bal. Acc | Precision | Recall | F1 | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost | Group-Aware | 0.750 | 0.596 | 0.765 | 0.942 | 0.844 | **0.727** | 0.880 | 0.274 |
| XGBoost | Random Split | 0.832 | 0.704 | 0.850 | 0.947 | 0.896 | 0.906 | 0.970 | 0.487 |
| MLP | Group-Aware | 0.698 | 0.561 | 0.751 | 0.870 | 0.806 | 0.691 | 0.858 | 0.147 |
| MLP | Random Split | 0.803 | 0.682 | 0.843 | 0.911 | 0.876 | 0.862 | 0.952 | 0.407 |
| KNN | Group-Aware | 0.685 | 0.588 | 0.769 | 0.805 | 0.787 | 0.650 | 0.805 | 0.184 |
| KNN | Random Split | 0.799 | 0.686 | 0.846 | 0.900 | 0.872 | 0.859 | 0.944 | 0.405 |
| Decision Tree | Group-Aware | 0.655 | 0.474 | 0.710 | 0.882 | 0.787 | 0.472 | 0.726 | −0.077 |
| Decision Tree | Random Split | 0.777 | 0.608 | 0.808 | 0.929 | 0.864 | 0.792 | 0.920 | 0.280 |

*(machine-readable: [`results/model_comparison.csv`](results/model_comparison.csv))*

### 5.2 Generalization gap (Random − Group ROC-AUC)

| Model | Random ROC-AUC | Group ROC-AUC | Generalization Gap |
|---|---|---|---|
| MLP | 0.862 | 0.691 | +0.171 |
| XGBoost | 0.906 | 0.727 | +0.179 |
| KNN | 0.859 | 0.650 | +0.208 |
| Decision Tree | 0.792 | 0.472 | **+0.321** |

*(machine-readable: [`results/generalization_gap.csv`](results/generalization_gap.csv))*

Every model looks far better under random splitting. The **Decision Tree** has
the largest gap and collapses to **below-chance** ranking on unseen papers.
**XGBoost** is the most robust, retaining genuine discriminative skill
(AUC 0.727, PR-AUC 0.880) out-of-sample.

> GB's/XGBoost's group accuracy (0.750) is only modestly above the 0.722 majority
> baseline, but its ROC-AUC 0.727 / PR-AUC 0.880 / MCC 0.274 show it ranks
> ignition vs no-ignition meaningfully better than chance — which is why we tune
> and select on AUC, not accuracy.

---

## 6. Model selection

| Model | GroupKFold CV ROC-AUC | |
|---|---|---|
| **XGBoost** | **0.681** | ← selected best |
| Decision Tree | 0.665 | |
| MLP | 0.663 | |
| KNN | 0.583 | |

The CV ranking and the held-out ranking agree that **XGBoost is best**, so it is
carried into permutation-importance and SHAP analysis.

---

## 7. Per-model diagnostic figures (primary group/unseen-paper split)

### 7.1 XGBoost (selected best)
| Confusion matrix | ROC curve | Precision-Recall |
|---|---|---|
| ![](results/confusion_matrix_xgboost.png) | ![](results/roc_curve_xgboost.png) | ![](results/pr_curve_xgboost.png) |

### 7.2 MLP
| Confusion matrix | ROC curve | Precision-Recall |
|---|---|---|
| ![](results/confusion_matrix_mlp.png) | ![](results/roc_curve_mlp.png) | ![](results/pr_curve_mlp.png) |

### 7.3 KNN
| Confusion matrix | ROC curve | Precision-Recall |
|---|---|---|
| ![](results/confusion_matrix_knn.png) | ![](results/roc_curve_knn.png) | ![](results/pr_curve_knn.png) |

### 7.4 Decision Tree
| Confusion matrix | ROC curve | Precision-Recall |
|---|---|---|
| ![](results/confusion_matrix_decision_tree.png) | ![](results/roc_curve_decision_tree.png) | ![](results/pr_curve_decision_tree.png) |

All models lean toward the majority "Ignition" class (high recall, lower
specificity), visible in the populated top row of each confusion matrix and the
modest balanced accuracy.

---

## 8. Per-paper generalization analysis

### 8.1 XGBoost (best model)
| Per-paper accuracy & F1 | Per-paper ROC-AUC | Per-paper accuracy boxplot |
|---|---|---|
| ![](results/per_paper_acc_f1_hist_xgboost.png) | ![](results/per_paper_auc_hist_xgboost.png) | ![](results/per_paper_accuracy_boxplot_xgboost.png) |

Performance is highly heterogeneous across the 19 unseen test papers: some are
classified almost perfectly while others are predicted worse than chance. Papers
with only one class present (all/none ignited) have undefined per-paper ROC-AUC
and are omitted from that histogram.

*(per-paper tables for all four models: `results/per_paper_metrics_*.csv`)*

### 8.2 MLP and KNN
| MLP per-paper accuracy/F1 | KNN per-paper accuracy/F1 |
|---|---|
| ![](results/per_paper_acc_f1_hist_mlp.png) | ![](results/per_paper_acc_f1_hist_knn.png) |

---

## 9. Feature importance (Decision Tree & XGBoost)

| Decision Tree | XGBoost |
|---|---|
| ![](results/feature_importance_decision_tree.png) | ![](results/feature_importance_xgboost.png) |

*(CSVs: [`results/feature_importance_decision_tree.csv`](results/feature_importance_decision_tree.csv),
[`results/feature_importance_xgboost.csv`](results/feature_importance_xgboost.csv))*

---

## 10. Permutation importance (best model: XGBoost)

Computed on the **held-out unseen papers** with ROC-AUC scoring.

![](results/permutation_importance.png)

*(ranked table: [`results/permutation_importance.csv`](results/permutation_importance.csv))*

**Oxygen Concentration dominates** (≈0.19 AUC drop when shuffled), followed by
Pressure, Material and sample Dimensions.

---

## 11. SHAP analysis (best model: XGBoost)

| SHAP summary (beeswarm) | SHAP mean(\|value\|) bar |
|---|---|
| ![](results/shap_summary_xgboost.png) | ![](results/shap_bar_xgboost.png) |

*(ranked table: [`results/shap_ranking_xgboost.csv`](results/shap_ranking_xgboost.csv))*

**Top SHAP drivers of ignition:** Oxygen Concentration (mean|SHAP| ≈ 1.7) ≫ Flow
Velocity, Pressure, Internal Dimensions, Gravity, Rig (FLARE), sample geometry
(flat/wire) and the ignition stimulus (time). Physically, ignition in
microgravity is governed first by **oxidiser availability and transport** (O₂,
pressure, flow), then by the **ignition stimulus** and the **fuel's
thermal/geometric properties**, with gravity modulating buoyant transport.

---

## 12. Conclusions

1. **Ignition is predictable but extrapolation is hard.** XGBoost reaches
   group-aware ROC-AUC 0.727 / PR-AUC 0.880 on unseen papers — useful, but well
   below its random-split AUC of 0.906.
2. **XGBoost is the selected and most robust model** (best on both CV and
   holdout); MLP and KNN are moderate; the **Decision Tree over-fits
   catastrophically** (group AUC 0.472, MCC < 0).
3. **Generalization gaps of +0.17 to +0.32 AUC** prove random-split metrics
   overstate real-world ignition-prediction capability.
4. **Oxygen concentration is the dominant, physically-consistent driver** across
   feature importance, permutation importance and SHAP.
5. **Accuracy is the wrong headline metric** under the 76/24 imbalance; ROC-AUC,
   PR-AUC, balanced accuracy and MCC are used for tuning/selection.

### Limitations & future work
* First-number parsing of unit-laden columns is transparent but not
  unit-normalised.
* Class imbalance could be tackled more aggressively (SMOTE, calibrated
  thresholds); probability **calibration** would make outputs directly
  actionable.
* Only 93 papers / 19 test papers → noisy holdout; leave-one-paper-out CV and
  more sources would tighten estimates.

---

## 13. Reproducibility & how to run

```bash
pip install pandas scikit-learn numpy matplotlib joblib openpyxl shap xgboost
python "ignition classifier/ignition_classification.py"            # full run
python "ignition classifier/ignition_classification.py" --no-shap  # skip SHAP
```

* `random_state = 42` for every split, search and model.
* The script auto-locates `Microgravity_Database.xlsm` and writes all artefacts to
  `ignition classifier/results/` by default.

### Files in this folder
| File | Contents |
|---|---|
| `ignition_classification.py` | the complete, commented, runnable script |
| `REPORT.md` | this report |
| `run_log.txt` | full console output of the benchmarked run |
| `results/model_comparison.csv` | metrics for all 4 models × both strategies |
| `results/generalization_gap.csv` | random vs group ROC-AUC + gap |
| `results/metrics.json` | machine-readable summary |
| `results/confusion_matrix_*.png`, `roc_curve_*.png`, `pr_curve_*.png` | per-model diagnostics |
| `results/per_paper_*` | per-paper extrapolation analysis (CSV + plots) |
| `results/feature_importance_*` | Decision Tree & XGBoost importances |
| `results/permutation_importance.*` | best-model permutation importance |
| `results/shap_summary_*`, `shap_bar_*`, `shap_ranking_*` | best-model SHAP analysis |
| `results/best_decision_tree.joblib`, `best_xgboost.joblib`, `best_knn.joblib`, `best_mlp.joblib` | saved fitted pipelines |
