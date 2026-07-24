| model_id                                     | model_family   | feature_set   |   roc_auc_mean |   roc_auc_std |   pr_auc_mean |   pr_auc_std |   brier_mean |   fold_count |
|:---------------------------------------------|:---------------|:--------------|---------------:|--------------:|--------------:|-------------:|-------------:|-------------:|
| xgb_all_unweighted                           | xgboost        | all           |       0.912995 |   0.0114064   |      0.97206  |  0.00403488  |     0.104342 |            3 |
| xgb_all_class_paper_weighted                 | xgboost        | all           |       0.911507 |   0.00487618  |      0.971303 |  0.00242859  |     0.118085 |            3 |
| xgb_physics_unweighted                       | xgboost        | physics       |       0.891662 |   0.0102476   |      0.96412  |  0.00422214  |     0.116706 |            3 |
| xgb_physics_class_paper_weighted_monotone_o2 | xgboost        | physics       |       0.886738 |   0.00736197  |      0.962625 |  0.0035807   |     0.133346 |            3 |
| svm_all                                      | svm            | all           |       0.87364  |   0.00603445  |      0.957399 |  0.00137522  |     0.151126 |            3 |
| xgb_physics_paper_bagging                    | xgboost        | physics       |       0.872391 |   0.0163849   |      0.956762 |  0.00777924  |     0.125727 |            3 |
| mlp_all                                      | mlp            | all           |       0.868408 |   0.0018024   |      0.956381 |  0.000202614 |     0.156066 |            3 |
| decision_tree_all                            | decision_tree  | all           |       0.859524 |   0.00743101  |      0.938887 |  0.00252389  |     0.151573 |            3 |
| svm_physics                                  | svm            | physics       |       0.840401 |   0.00728307  |      0.942852 |  0.00690735  |     0.167589 |            3 |
| decision_tree_physics                        | decision_tree  | physics       |       0.833813 |   0.0119141   |      0.929681 |  0.00761683  |     0.169857 |            3 |
| mlp_physics                                  | mlp            | physics       |       0.828339 |   0.01107     |      0.941077 |  0.00506727  |     0.174124 |            3 |
| knn_physics                                  | knn            | physics       |       0.824974 |   0.000957151 |      0.932138 |  0.00362171  |     0.184716 |            3 |
| knn_all                                      | knn            | all           |       0.822384 |   0.0122616   |      0.93115  |  0.00265119  |     0.185136 |            3 |
| xgb_physics_focal_class_paper_weighted       | xgboost        | physics       |       0.501133 |   0.0123678   |      0.755848 |  0.00321666  |     0.754781 |            3 |
| xgb_all_focal_class_paper_weighted           | xgboost        | all           |       0.479309 |   0.0212418   |      0.749181 |  0.00481727  |     0.755896 |            3 |
