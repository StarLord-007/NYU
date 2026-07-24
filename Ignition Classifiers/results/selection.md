# Reproducible champion selection

> Interpolation and extrapolation champions answer different scientific questions; neither result substitutes for the other.

## Interpolation Champion

Selected `xgb_all_unweighted`. Highest eligible roc_auc within tolerance 0.01; tie-breakers applied in declared order: higher_pr_auc, lower_uncertainty, simpler_model, fewer_features. Refit hyperparameters use: modal outer-fold selected configuration with lexical tie resolution.

- ROC-AUC: 0.9141 ± 0.0064
- PR-AUC: 0.9720 ± 0.0028
- Protocol: `interpolation_stratified`
- Exact hyperparameters: `{"colsample_bytree": 0.6, "gamma": 0.0, "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 2, "n_estimators": 600, "objective_variant": "logistic", "reg_alpha": 0.5, "reg_lambda": 5.0, "subsample": 0.8}`

| Rank | Candidate | ROC-AUC | PR-AUC | SD | Features |
|---:|---|---:|---:|---:|---|
| 1 | xgb_all_unweighted | 0.9141 | 0.9720 | 0.0064 | all |
| 2 | xgb_all_class_paper_weighted | 0.9134 | 0.9715 | 0.0067 | all |
| 3 | xgb_physics_unweighted | 0.8978 | 0.9658 | 0.0077 | physics |
| 4 | xgb_physics_class_paper_weighted_monotone_o2 | 0.8949 | 0.9648 | 0.0093 | physics |
| 5 | xgb_physics_paper_bagging | 0.8747 | 0.9578 | 0.0137 | physics |
| 6 | mlp_all | 0.8651 | 0.9537 | 0.0122 | all |
| 7 | svm_all | 0.8645 | 0.9523 | 0.0115 | all |
| 8 | decision_tree_all | 0.8539 | 0.9381 | 0.0196 | all |
| 9 | svm_physics | 0.8445 | 0.9438 | 0.0132 | physics |
| 10 | mlp_physics | 0.8421 | 0.9442 | 0.0143 | physics |
| 11 | decision_tree_physics | 0.8375 | 0.9351 | 0.0156 | physics |
| 12 | knn_physics | 0.8260 | 0.9279 | 0.0134 | physics |
| 13 | knn_all | 0.8229 | 0.9295 | 0.0134 | all |
| 14 | xgb_all_focal_class_paper_weighted | 0.4915 | 0.7526 | 0.0111 | all |
| 15 | xgb_physics_focal_class_paper_weighted | 0.4892 | 0.7528 | 0.0260 | physics |

## Extrapolation Champion

Selected `xgb_physics_unweighted`. Highest eligible roc_auc within tolerance 0.01; tie-breakers applied in declared order: higher_pr_auc, lower_uncertainty, physics_only, simpler_model. Refit hyperparameters use: modal outer-fold selected configuration with lexical tie resolution.

- ROC-AUC: 0.7210 ± 0.0656
- PR-AUC: 0.8787 ± 0.0447
- Protocol: `extrapolation_grouped`
- Exact hyperparameters: `{"colsample_bytree": 0.6, "gamma": 0.0, "learning_rate": 0.01, "max_depth": 4, "min_child_weight": 4, "n_estimators": 400, "objective_variant": "logistic", "reg_alpha": 0.0, "reg_lambda": 5.0, "subsample": 0.8}`

| Rank | Candidate | ROC-AUC | PR-AUC | SD | Features |
|---:|---|---:|---:|---:|---|
| 1 | xgb_physics_unweighted | 0.7210 | 0.8787 | 0.0656 | physics |
| 2 | xgb_physics_class_paper_weighted_monotone_o2 | 0.7151 | 0.8772 | 0.0763 | physics |
| 3 | xgb_physics_paper_bagging | 0.7133 | 0.8778 | 0.0630 | physics |
| 4 | knn_physics | 0.7040 | 0.8706 | 0.0673 | physics |
| 5 | xgb_all_unweighted | 0.7018 | 0.8713 | 0.0477 | all |
| 6 | svm_physics | 0.6995 | 0.8790 | 0.0631 | physics |
| 7 | xgb_all_class_paper_weighted | 0.6983 | 0.8672 | 0.0482 | all |
| 8 | mlp_physics | 0.6958 | 0.8661 | 0.0833 | physics |
| 9 | mlp_all | 0.6579 | 0.8538 | 0.0846 | all |
| 10 | decision_tree_physics | 0.6560 | 0.8277 | 0.0571 | physics |
| 11 | svm_all | 0.6429 | 0.8590 | 0.0784 | all |
| 12 | knn_all | 0.6284 | 0.8295 | 0.0562 | all |
| 13 | decision_tree_all | 0.5956 | 0.8011 | 0.1106 | all |
| 14 | xgb_all_focal_class_paper_weighted | 0.4923 | 0.7524 | 0.0261 | all |
| 15 | xgb_physics_focal_class_paper_weighted | 0.4911 | 0.7542 | 0.0232 | physics |

