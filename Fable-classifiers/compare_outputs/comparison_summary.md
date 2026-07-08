# Fable five-classifier benchmark

All interpolation protocols use row-level splits; all extrapolation protocols hold out whole papers. Thresholds are tuned on inner held-out-paper folds only.

## Interpolation

### Random stratified 80/20 (x5 seeds)

| Model | Group ROC-AUC | PR-AUC | MCC | Balanced Accuracy |
|---|---|---|---|---|
| XGBoost | 0.896 +/- 0.010 | 0.965 +/- 0.003 | 0.489 +/- 0.029 | 0.776 +/- 0.017 |
| MLP | 0.849 +/- 0.037 | 0.948 +/- 0.017 | 0.385 +/- 0.190 | 0.720 +/- 0.121 |
| SVM (RBF) | 0.834 +/- 0.008 | 0.942 +/- 0.003 | 0.439 +/- 0.016 | 0.750 +/- 0.010 |
| Decision Tree | 0.808 +/- 0.014 | 0.910 +/- 0.010 | 0.325 +/- 0.084 | 0.679 +/- 0.089 |
| KNN | 0.773 +/- 0.017 | 0.900 +/- 0.013 | 0.375 +/- 0.026 | 0.718 +/- 0.016 |

### Stratified 5-fold

| Model | Group ROC-AUC | PR-AUC | MCC | Balanced Accuracy |
|---|---|---|---|---|
| XGBoost | 0.898 +/- 0.008 | 0.966 +/- 0.003 | 0.486 +/- 0.057 | 0.769 +/- 0.037 |
| MLP | 0.860 +/- 0.027 | 0.951 +/- 0.007 | 0.401 +/- 0.185 | 0.736 +/- 0.078 |
| Decision Tree | 0.816 +/- 0.013 | 0.915 +/- 0.006 | 0.399 +/- 0.102 | 0.736 +/- 0.064 |
| SVM (RBF) | 0.809 +/- 0.015 | 0.930 +/- 0.008 | 0.349 +/- 0.119 | 0.707 +/- 0.083 |
| KNN | 0.773 +/- 0.013 | 0.898 +/- 0.010 | 0.404 +/- 0.013 | 0.730 +/- 0.009 |

## Extrapolation

### Repeated grouped CV (StratifiedGroupKFold 5x seeds)

| Model | Group ROC-AUC | PR-AUC | MCC | Balanced Accuracy |
|---|---|---|---|---|
| XGBoost | 0.660 +/- 0.070 | 0.845 +/- 0.039 | 0.211 +/- 0.104 | 0.612 +/- 0.058 |
| MLP | 0.610 +/- 0.072 | 0.817 +/- 0.047 | 0.092 +/- 0.117 | 0.561 +/- 0.058 |
| SVM (RBF) | 0.592 +/- 0.109 | 0.811 +/- 0.061 | 0.095 +/- 0.104 | 0.553 +/- 0.053 |
| KNN | 0.581 +/- 0.079 | 0.800 +/- 0.051 | 0.124 +/- 0.101 | 0.553 +/- 0.045 |
| Decision Tree | 0.581 +/- 0.050 | 0.789 +/- 0.030 | 0.103 +/- 0.151 | 0.548 +/- 0.067 |

### Leave-One-Paper-Out (pooled)

| Model | Pooled ROC-AUC | Pooled PR-AUC |
|---|---|---|
| XGBoost | 0.674 | 0.844 |
| MLP | 0.632 | 0.826 |
| SVM (RBF) | 0.596 | 0.797 |
| Decision Tree | 0.587 | 0.777 |
| KNN | 0.579 | 0.792 |

### Pairwise comparisons on grouped CV (paired folds, Wilcoxon)

| A | B | dROC-AUC | p | dMCC | p |
|---|---|---|---|---|---|
| XGBoost | Decision Tree | +0.079 | 0.00262 | +0.108 | 0.121 |
| XGBoost | KNN | +0.078 | 0.00671 | +0.086 | 0.0302 |
| XGBoost | SVM (RBF) | +0.068 | 0.121 | +0.116 | 0.00116 |
| XGBoost | MLP | +0.049 | 0.107 | +0.118 | 0.0181 |
| MLP | SVM (RBF) | +0.019 | 0.762 | -0.002 | 0.804 |
| KNN | Decision Tree | +0.001 | 0.679 | +0.022 | 0.679 |
| KNN | SVM (RBF) | -0.010 | 0.454 | +0.030 | 0.421 |
| Decision Tree | SVM (RBF) | -0.011 | 0.489 | +0.008 | 0.89 |
| KNN | MLP | -0.029 | 0.0103 | +0.032 | 0.599 |
| Decision Tree | MLP | -0.030 | 0.151 | +0.010 | 0.599 |
