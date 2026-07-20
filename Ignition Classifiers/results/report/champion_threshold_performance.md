| champion               | model_id               | protocol                 | threshold_policy   |   threshold_mean |    tp |   fp |   tn |   fn |   sensitivity |   specificity |   precision |
|:-----------------------|:-----------------------|:-------------------------|:-------------------|-----------------:|------:|-----:|-----:|-----:|--------------:|--------------:|------------:|
| interpolation_champion | xgb_all_unweighted     | interpolation_stratified | mcc                |         0.806025 |  8125 |  383 | 2932 | 2057 |      0.797977 |     0.884465  |    0.954984 |
| interpolation_champion | xgb_all_unweighted     | interpolation_stratified | f1                 |         0.385238 |  9534 | 1546 | 1769 |  648 |      0.936358 |     0.533635  |    0.860469 |
| interpolation_champion | xgb_all_unweighted     | interpolation_stratified | balanced_accuracy  |         0.877451 |  7747 |  218 | 3097 | 2435 |      0.760852 |     0.934238  |    0.97263  |
| interpolation_champion | xgb_all_unweighted     | interpolation_stratified | youden_j           |         0.877451 |  7747 |  218 | 3097 | 2435 |      0.760852 |     0.934238  |    0.97263  |
| extrapolation_champion | xgb_physics_unweighted | extrapolation_grouped    | mcc                |         0.780976 |  7528 | 1401 | 1914 | 2654 |      0.739344 |     0.577376  |    0.843096 |
| extrapolation_champion | xgb_physics_unweighted | extrapolation_grouped    | f1                 |         0.205517 | 10012 | 3158 |  157 |  170 |      0.983304 |     0.0473605 |    0.760213 |
| extrapolation_champion | xgb_physics_unweighted | extrapolation_grouped    | balanced_accuracy  |         0.842782 |  6426 | 1046 | 2269 | 3756 |      0.631114 |     0.684465  |    0.860011 |
| extrapolation_champion | xgb_physics_unweighted | extrapolation_grouped    | youden_j           |         0.842782 |  6426 | 1046 | 2269 | 3756 |      0.631114 |     0.684465  |    0.860011 |
