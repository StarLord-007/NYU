| model_id                                     | model_family   | feature_set   |   extrapolation_grouped |   interpolation_stratified |   roc_auc_drop |
|:---------------------------------------------|:---------------|:--------------|------------------------:|---------------------------:|---------------:|
| xgb_physics_focal_class_paper_weighted       | xgboost        | physics       |                0.491078 |                   0.489189 |   -0.00188857  |
| xgb_all_focal_class_paper_weighted           | xgboost        | all           |                0.492301 |                   0.491469 |   -0.000832496 |
| knn_physics                                  | knn            | physics       |                0.703984 |                   0.825998 |    0.122014    |
| svm_physics                                  | svm            | physics       |                0.699537 |                   0.844495 |    0.144958    |
| mlp_physics                                  | mlp            | physics       |                0.695804 |                   0.842078 |    0.146274    |
| xgb_physics_paper_bagging                    | xgboost        | physics       |                0.713334 |                   0.874699 |    0.161365    |
| xgb_physics_unweighted                       | xgboost        | physics       |                0.721006 |                   0.89776  |    0.176754    |
| xgb_physics_class_paper_weighted_monotone_o2 | xgboost        | physics       |                0.715064 |                   0.894869 |    0.179805    |
| decision_tree_physics                        | decision_tree  | physics       |                0.65602  |                   0.837517 |    0.181496    |
| knn_all                                      | knn            | all           |                0.628443 |                   0.822944 |    0.194501    |
| mlp_all                                      | mlp            | all           |                0.657936 |                   0.8651   |    0.207164    |
| xgb_all_unweighted                           | xgboost        | all           |                0.701796 |                   0.91408  |    0.212284    |
| xgb_all_class_paper_weighted                 | xgboost        | all           |                0.698271 |                   0.91338  |    0.215109    |
| svm_all                                      | svm            | all           |                0.642944 |                   0.86453  |    0.221587    |
| decision_tree_all                            | decision_tree  | all           |                0.595614 |                   0.853871 |    0.258258    |
