# Fable five-classifier benchmark

All interpolation protocols use row-level splits; all extrapolation protocols hold out whole papers. Thresholds are tuned on inner held-out-paper folds only.

For each strategy, the first table is the original non-bootstrap summary (mean +/- fold SD). When enabled, the bootstrap table reports mean [95% CI]. Fold-based strategies bootstrap folds within model; LOPO bootstraps whole papers.

## Interpolation

### Random stratified 80/20 (x5 seeds)

| Model | Group ROC-AUC | PR-AUC | MCC | Balanced Accuracy |
|---|---|---|---|---|
| XGBoost | 0.896 +/- 0.010 | 0.965 +/- 0.003 | 0.489 +/- 0.029 | 0.776 +/- 0.017 |
| MLP | 0.849 +/- 0.037 | 0.948 +/- 0.017 | 0.385 +/- 0.190 | 0.720 +/- 0.121 |
| SVM (RBF) | 0.834 +/- 0.008 | 0.942 +/- 0.003 | 0.439 +/- 0.016 | 0.750 +/- 0.010 |
| Decision Tree | 0.808 +/- 0.014 | 0.910 +/- 0.010 | 0.325 +/- 0.084 | 0.679 +/- 0.089 |
| KNN | 0.773 +/- 0.017 | 0.900 +/- 0.013 | 0.375 +/- 0.026 | 0.718 +/- 0.016 |

#### Bootstrap: random stratified 80/20

| Model | ROC-AUC | PR-AUC | MCC | Balanced Accuracy |
|---|---|---|---|---|
| XGBoost | 0.896 [0.889, 0.904] | 0.965 [0.963, 0.968] | 0.489 [0.468, 0.511] | 0.776 [0.764, 0.790] |
| MLP | 0.849 [0.816, 0.872] | 0.948 [0.932, 0.958] | 0.385 [0.215, 0.494] | 0.720 [0.613, 0.786] |
| SVM (RBF) | 0.834 [0.828, 0.840] | 0.942 [0.940, 0.944] | 0.439 [0.427, 0.451] | 0.750 [0.744, 0.758] |
| Decision Tree | 0.808 [0.796, 0.818] | 0.910 [0.902, 0.917] | 0.325 [0.257, 0.395] | 0.679 [0.606, 0.746] |
| KNN | 0.773 [0.759, 0.785] | 0.900 [0.889, 0.909] | 0.375 [0.358, 0.397] | 0.718 [0.705, 0.731] |

### Stratified 5-fold

| Model | Group ROC-AUC | PR-AUC | MCC | Balanced Accuracy |
|---|---|---|---|---|
| XGBoost | 0.898 +/- 0.008 | 0.966 +/- 0.003 | 0.486 +/- 0.057 | 0.769 +/- 0.037 |
| MLP | 0.860 +/- 0.027 | 0.951 +/- 0.007 | 0.401 +/- 0.185 | 0.736 +/- 0.078 |
| Decision Tree | 0.816 +/- 0.013 | 0.915 +/- 0.006 | 0.399 +/- 0.102 | 0.736 +/- 0.064 |
| SVM (RBF) | 0.809 +/- 0.015 | 0.930 +/- 0.008 | 0.349 +/- 0.119 | 0.707 +/- 0.083 |
| KNN | 0.773 +/- 0.013 | 0.898 +/- 0.010 | 0.404 +/- 0.013 | 0.730 +/- 0.009 |

#### Bootstrap: stratified 5-fold

| Model | ROC-AUC | PR-AUC | MCC | Balanced Accuracy |
|---|---|---|---|---|
| XGBoost | 0.898 [0.893, 0.904] | 0.966 [0.964, 0.968] | 0.486 [0.435, 0.524] | 0.769 [0.737, 0.794] |
| MLP | 0.860 [0.840, 0.880] | 0.951 [0.944, 0.956] | 0.401 [0.252, 0.541] | 0.736 [0.675, 0.797] |
| Decision Tree | 0.816 [0.807, 0.826] | 0.915 [0.911, 0.921] | 0.399 [0.309, 0.475] | 0.736 [0.678, 0.778] |
| SVM (RBF) | 0.809 [0.797, 0.822] | 0.930 [0.924, 0.937] | 0.349 [0.244, 0.430] | 0.707 [0.632, 0.750] |
| KNN | 0.773 [0.761, 0.782] | 0.898 [0.891, 0.905] | 0.404 [0.396, 0.417] | 0.730 [0.723, 0.738] |

## Extrapolation

### Repeated grouped CV (StratifiedGroupKFold 5x seeds)

| Model | Group ROC-AUC | PR-AUC | MCC | Balanced Accuracy |
|---|---|---|---|---|
| XGBoost | 0.660 +/- 0.070 | 0.845 +/- 0.039 | 0.211 +/- 0.104 | 0.612 +/- 0.058 |
| MLP | 0.610 +/- 0.072 | 0.817 +/- 0.047 | 0.092 +/- 0.117 | 0.561 +/- 0.058 |
| SVM (RBF) | 0.592 +/- 0.109 | 0.811 +/- 0.061 | 0.095 +/- 0.104 | 0.553 +/- 0.053 |
| KNN | 0.581 +/- 0.079 | 0.800 +/- 0.051 | 0.124 +/- 0.101 | 0.553 +/- 0.045 |
| Decision Tree | 0.581 +/- 0.050 | 0.789 +/- 0.030 | 0.103 +/- 0.151 | 0.548 +/- 0.067 |

#### Bootstrap: repeated grouped CV

| Model | ROC-AUC | PR-AUC | MCC | Balanced Accuracy |
|---|---|---|---|---|
| XGBoost | 0.660 [0.625, 0.695] | 0.845 [0.826, 0.864] | 0.211 [0.159, 0.257] | 0.612 [0.583, 0.640] |
| MLP | 0.610 [0.576, 0.644] | 0.817 [0.793, 0.840] | 0.092 [0.032, 0.145] | 0.561 [0.533, 0.588] |
| SVM (RBF) | 0.592 [0.533, 0.642] | 0.811 [0.783, 0.841] | 0.095 [0.042, 0.146] | 0.553 [0.525, 0.580] |
| KNN | 0.581 [0.545, 0.620] | 0.800 [0.777, 0.826] | 0.124 [0.075, 0.174] | 0.553 [0.533, 0.576] |
| Decision Tree | 0.581 [0.557, 0.605] | 0.789 [0.775, 0.803] | 0.103 [0.023, 0.175] | 0.548 [0.518, 0.584] |

### Leave-One-Paper-Out (pooled)

| Model | Pooled ROC-AUC | Pooled PR-AUC |
|---|---|---|
| XGBoost | 0.674 | 0.844 |
| MLP | 0.632 | 0.826 |
| SVM (RBF) | 0.596 | 0.797 |
| Decision Tree | 0.587 | 0.777 |
| KNN | 0.579 | 0.792 |

#### Bootstrap: Leave-One-Paper-Out pooled

| Model | Pooled ROC-AUC | Pooled PR-AUC |
|---|---|---|
| XGBoost | 0.677 [0.612, 0.752] | 0.849 [0.788, 0.917] |
| MLP | 0.637 [0.574, 0.719] | 0.830 [0.754, 0.912] |
| SVM (RBF) | 0.604 [0.518, 0.702] | 0.803 [0.703, 0.903] |
| Decision Tree | 0.593 [0.513, 0.683] | 0.785 [0.703, 0.872] |
| KNN | 0.583 [0.493, 0.682] | 0.797 [0.695, 0.889] |

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
