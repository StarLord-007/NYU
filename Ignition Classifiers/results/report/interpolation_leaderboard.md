| model_id                                     | model_family   | feature_set   |   roc_auc_mean |   roc_auc_std |   pr_auc_mean |   pr_auc_std |   brier_mean |   fold_count |
|:---------------------------------------------|:---------------|:--------------|---------------:|--------------:|--------------:|-------------:|-------------:|-------------:|
| xgb_all_unweighted                           | xgboost        | all           |       0.91408  |    0.00640379 |      0.971992 |   0.00280625 |     0.104861 |           15 |
| xgb_all_class_paper_weighted                 | xgboost        | all           |       0.91338  |    0.00668951 |      0.971504 |   0.00318965 |     0.112802 |           15 |
| xgb_physics_unweighted                       | xgboost        | physics       |       0.89776  |    0.00773073 |      0.965804 |   0.00388683 |     0.112774 |           15 |
| xgb_physics_class_paper_weighted_monotone_o2 | xgboost        | physics       |       0.894869 |    0.00926125 |      0.964805 |   0.00440679 |     0.125464 |           15 |
| xgb_physics_paper_bagging                    | xgboost        | physics       |       0.874699 |    0.0137216  |      0.957827 |   0.0062832  |     0.123551 |           15 |
| mlp_all                                      | mlp            | all           |       0.8651   |    0.0121762  |      0.953665 |   0.00521753 |     0.151198 |           15 |
| svm_all                                      | svm            | all           |       0.86453  |    0.011535   |      0.952341 |   0.00580554 |     0.173236 |           15 |
| decision_tree_all                            | decision_tree  | all           |       0.853871 |    0.0196183  |      0.938087 |   0.0119661  |     0.152523 |           15 |
| svm_physics                                  | svm            | physics       |       0.844495 |    0.0132065  |      0.943806 |   0.00702024 |     0.182258 |           15 |
| mlp_physics                                  | mlp            | physics       |       0.842078 |    0.0142802  |      0.94415  |   0.00844566 |     0.170893 |           15 |
| decision_tree_physics                        | decision_tree  | physics       |       0.837517 |    0.015618   |      0.935065 |   0.00710553 |     0.164033 |           15 |
| knn_physics                                  | knn            | physics       |       0.825998 |    0.0134312  |      0.92789  |   0.00803637 |     0.177905 |           15 |
| knn_all                                      | knn            | all           |       0.822944 |    0.0133509  |      0.929528 |   0.00690102 |     0.181257 |           15 |
| xgb_all_focal_class_paper_weighted           | xgboost        | all           |       0.491469 |    0.0111002  |      0.752561 |   0.00324597 |     0.75396  |           15 |
| xgb_physics_focal_class_paper_weighted       | xgboost        | physics       |       0.489189 |    0.0259769  |      0.752751 |   0.00700445 |     0.750942 |           15 |
