| model_id                                     | model_family   | feature_set   |   roc_auc_mean |   roc_auc_std |   pr_auc_mean |   pr_auc_std |   brier_mean |   fold_count |
|:---------------------------------------------|:---------------|:--------------|---------------:|--------------:|--------------:|-------------:|-------------:|-------------:|
| xgb_physics_unweighted                       | xgboost        | physics       |       0.721006 |     0.0656052 |      0.878668 |    0.0446896 |     0.172549 |           15 |
| xgb_physics_class_paper_weighted_monotone_o2 | xgboost        | physics       |       0.715064 |     0.0762542 |      0.877226 |    0.0528109 |     0.189701 |           15 |
| xgb_physics_paper_bagging                    | xgboost        | physics       |       0.713334 |     0.0629542 |      0.877754 |    0.0405151 |     0.175537 |           15 |
| knn_physics                                  | knn            | physics       |       0.703984 |     0.0673397 |      0.870559 |    0.03565   |     0.211304 |           15 |
| xgb_all_unweighted                           | xgboost        | all           |       0.701796 |     0.04774   |      0.871296 |    0.0351237 |     0.185491 |           15 |
| svm_physics                                  | svm            | physics       |       0.699537 |     0.0631079 |      0.879018 |    0.0275254 |     0.199147 |           15 |
| xgb_all_class_paper_weighted                 | xgboost        | all           |       0.698271 |     0.0482146 |      0.867179 |    0.0317869 |     0.189587 |           15 |
| mlp_physics                                  | mlp            | physics       |       0.695804 |     0.0833102 |      0.866147 |    0.0498918 |     0.219266 |           15 |
| mlp_all                                      | mlp            | all           |       0.657936 |     0.0846335 |      0.853818 |    0.0461046 |     0.231849 |           15 |
| decision_tree_physics                        | decision_tree  | physics       |       0.65602  |     0.0571325 |      0.827654 |    0.0337777 |     0.229874 |           15 |
| svm_all                                      | svm            | all           |       0.642944 |     0.078445  |      0.859035 |    0.0321636 |     0.203112 |           15 |
| knn_all                                      | knn            | all           |       0.628443 |     0.056218  |      0.829463 |    0.0318635 |     0.239941 |           15 |
| decision_tree_all                            | decision_tree  | all           |       0.595614 |     0.110641  |      0.801121 |    0.05741   |     0.250374 |           15 |
| xgb_all_focal_class_paper_weighted           | xgboost        | all           |       0.492301 |     0.0260701 |      0.752353 |    0.0192528 |     0.739755 |           15 |
| xgb_physics_focal_class_paper_weighted       | xgboost        | physics       |       0.491078 |     0.0231959 |      0.754184 |    0.0208139 |     0.731516 |           15 |
