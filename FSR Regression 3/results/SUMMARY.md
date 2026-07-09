# FSR Extrapolation Regression -- Repeated-Benchmark Summary

- Random state: **42**
- Number of train/test repeats: **10**
- Rows (initial / after target cleaning): **5083 / 2605**
- Unique papers / groups: **67**
- Target column: `FSR (Flame Spread Rate)`
- Bootstrap rows added per train split: **1000**
- Best model (lowest mean GroupKFold CV RMSE): **KNN**

## Mean GroupKFold CV RMSE (lower is better)

```
        Model  Group CV RMSE_mean  Group CV RMSE_std
          KNN             12.1851             0.4373
      XGBoost             13.6018             0.9481
Decision Tree             14.0121             0.7734
          MLP             14.3553             1.7457
```

## Model comparison -- mean +/- std across repeats (4 conditions)

```
        Model  Validation Strategy  R2_mean  R2_std  RMSE_mean  RMSE_std  MAE_mean  MAE_std  MAPE_mean  MAPE_std  NRMSE_mean  NRMSE_std  MBE_mean  MBE_std
          KNN     Group-Aware (RD)   0.5080  0.1066    11.6815    2.1982    6.1312   1.2040   447.3103  400.8837      1.3389     0.2684    0.3117   2.0973
          KNN  Group-Aware (RD+BT)   0.4941  0.1134    11.8266    2.0820    6.2157   1.1600   458.9528  414.2505      1.3532     0.2388    0.4594   2.2206
          KNN    Random Split (RD)   0.7352  0.0454     9.2082    0.9555    3.6332   0.2644   102.1709   21.3910      1.0317     0.1011   -0.0655   0.3356
          KNN Random Split (RD+BT)   0.7383  0.0435     9.1567    0.9298    3.5741   0.2531    94.4720   21.4873      1.0257     0.0964   -0.1482   0.2650
      XGBoost     Group-Aware (RD)   0.3516  0.2261    13.4408    3.5290    7.3015   1.5827   794.6167  435.5773      1.5297     0.3919    1.7440   1.6851
      XGBoost  Group-Aware (RD+BT)   0.3378  0.2377    13.5821    3.7151    7.3975   1.6636   813.8690  442.1243      1.5454     0.4144    1.8700   1.8634
      XGBoost    Random Split (RD)   0.6424  0.1166    10.6814    2.3327    4.8463   0.9054   817.8409  374.6365      1.1916     0.2330   -0.2872   0.4041
      XGBoost Random Split (RD+BT)   0.6511  0.1039    10.5732    2.2141    4.8347   0.8798   822.2171  383.6082      1.1796     0.2169   -0.2245   0.4943
Decision Tree     Group-Aware (RD)   0.1167  0.4264    15.4528    4.7426    8.0439   2.3011   659.9364  497.7109      1.7585     0.5131    2.6904   3.0809
Decision Tree  Group-Aware (RD+BT)   0.2068  0.5265    14.3693    5.7747    7.1636   2.3183   614.4411  476.6806      1.6406     0.6317    1.4782   2.2883
Decision Tree    Random Split (RD)   0.6226  0.1107    10.9921    2.2199    4.4355   0.6981   313.8336  155.3962      1.2273     0.2195   -0.1766   0.4569
Decision Tree Random Split (RD+BT)   0.6265  0.1203    10.9020    2.2261    4.4448   0.5501   308.8850  162.2955      1.2180     0.2279   -0.1787   0.4801
          MLP     Group-Aware (RD)  -0.7613  3.2907    17.0336   10.0298    7.2378   1.5829   440.0217  260.7642      2.0533     1.5146    1.7388   2.5449
          MLP  Group-Aware (RD+BT)  -0.2900  1.7756    16.0674    6.6636    7.5179   1.6447   509.0682  226.9578      1.9066     1.0396    1.8870   2.6723
          MLP    Random Split (RD)   0.7509  0.0571     8.9219    1.2536    3.5921   0.3324   211.4939   70.3616      0.9986     0.1293   -0.0674   0.4904
          MLP Random Split (RD+BT)   0.7563  0.0619     8.8315    1.4542    3.5622   0.4105   216.4013   79.3185      0.9871     0.1431   -0.2723   0.3382
```

## Generalization gap (Group RMSE - Random RMSE)

```
        Model  Random RMSE_mean  Random RMSE_std  Group RMSE_mean  Group RMSE_std  Generalization Gap_mean  Generalization Gap_std
          KNN            9.2082           0.9555          11.6815          2.1982                   2.4732                  2.7604
      XGBoost           10.6814           2.3327          13.4408          3.5290                   2.7594                  4.0143
Decision Tree           10.9921           2.2199          15.4528          4.7426                   4.4607                  6.2033
          MLP            8.9219           1.2536          17.0336         10.0298                   8.1117                 10.3195
```

## Bootstrap augmentation effect (negative dRMSE = BT helps)

```
        Model  Group RMSE (RD)_mean  Group RMSE (RD)_std  Group RMSE (RD+BT)_mean  Group RMSE (RD+BT)_std  Group dRMSE (BT-RD)_mean  Group dRMSE (BT-RD)_std  Random RMSE (RD)_mean  Random RMSE (RD)_std  Random RMSE (RD+BT)_mean  Random RMSE (RD+BT)_std  Random dRMSE (BT-RD)_mean  Random dRMSE (BT-RD)_std
          KNN               11.6815               2.1982                  11.8266                  2.0820                    0.1451                   0.3911                 9.2082                0.9555                    9.1567                   0.9298                    -0.0515                    0.1557
      XGBoost               13.4408               3.5290                  13.5821                  3.7151                    0.1413                   0.4641                10.6814                2.3327                   10.5732                   2.2141                    -0.1082                    0.3167
Decision Tree               15.4528               4.7426                  14.3693                  5.7747                   -1.0835                   2.5267                10.9921                2.2199                   10.9020                   2.2261                    -0.0901                    0.7242
          MLP               17.0336              10.0298                  16.0674                  6.6636                   -0.9663                   3.7882                 8.9219                1.2536                    8.8315                   1.4542                    -0.0904                    0.6822
```

## Output files (all under `results/`)

- `model_comparison.csv` -- headline mean +/- std table (PRIMARY).
- `model_comparison_repeats.csv` -- one row per (repeat, model, strategy).
- `generalization_gap.csv` / `generalization_gap_repeats.csv`
- `augmentation_comparison.csv` / `augmentation_comparison_repeats.csv`
- `best_params_repeats.csv` -- tuned hyper-parameters per repeat.
- `predictions/` -- per-repeat y_true / y_pred CSVs for every model x strategy.
- `model_comparison_bars.png` -- mean +/- std RMSE per model (4 strategies).
- `per_repeat_group_rmse_strip.png` -- run-to-run variability for Group-Aware RD.
- `pred_vs_true_*.png`, `residuals_*.png`, `error_hist_*.png` (representative repeat).
- `per_paper_*` -- per-paper extrapolation analysis (representative repeat).
- `feature_importance_*` -- Decision Tree & XGBoost.
- `permutation_importance.*` -- best repeated-benchmark model.
- `shap_summary_*` / `shap_bar_*` -- best model (unless --no-shap).
- `metrics.json` -- machine-readable bundle.
- `best_*.joblib` -- best representative pipelines per model.