# Protocol comparison

## Interpolation

| Protocol | ROC-AUC mean | ROC-AUC std |
|---|---|---|
| random stratified 80/20 hold-out | 0.915 | 0.007 |
| stratified 5-fold CV | 0.912 | 0.007 |

## Extrapolation

| Protocol | ROC-AUC mean | ROC-AUC std |
|---|---|---|
| GroupKFold(5) by paper | 0.706 | 0.108 |
| StratifiedGroupKFold(5) x3 seeds | 0.721 | 0.036 |
| Leave-One-Paper-Out (pooled) | 0.695 | nan |
