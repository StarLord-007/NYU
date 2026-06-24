# FSR Regression 2 results presentation plan

Source folder: `FSR Regression 2/results`. Report files are intentionally ignored.

## Slide 1 - Title and headline
- Put: title, project goal, one-sentence headline.
- Interpret: this is a paper-aware FSR extrapolation study, not a random-row benchmark.

## Slide 2 - Why paper-aware evaluation matters
- Put: simple schematic/bullets contrasting random split vs group-aware paper split.
- Interpret: random split can leak paper/campaign context; group split tests a new literature campaign.

## Slide 3 - Dataset and pipeline
- Put table: 5118 initial rows, 2605 valid FSR rows, 69 papers, target `FSR (Flame Spread Rate)`, group key `Article (MLA)`, 8 numeric + 6 categorical features.
- Interpret: the code removes target-like outputs, post-experiment measurements, notes, and citation identifiers before modeling.

## Slide 4 - Primary unseen-paper model comparison
- Put figure: `figures/primary_group_holdout_rmse.png`.
- Put table: Group-Aware (RD) rows from `model_comparison.csv`.
- Interpret: MLP is best on the final holdout (RMSE 10.64, R2 0.55); KNN is close; XGBoost and Decision Tree are weaker on this split.

## Slide 5 - CV selection caveat
- Put table: GroupKFold CV RMSE from `metrics.json`.
- Interpret: `metrics.json` selects Decision Tree by CV RMSE, but the final holdout ranks MLP first. This indicates high paper-split sensitivity and argues for repeated group splits before making a final model-selection claim.

## Slide 6 - Random vs group-aware performance
- Put figure: `figures/random_vs_group_rmse.png`.
- Put table: `generalization_gap.csv`.
- Interpret: KNN has the clearest optimistic random-split result; random split should be framed only as interpolation baseline.

## Slide 7 - Bootstrap augmentation effect
- Put figure: `figures/bootstrap_delta_group_rmse.png`.
- Put table: `augmentation_comparison.csv`.
- Interpret: bootstrap resampling is mixed; KNN and Decision Tree improve slightly, MLP and XGBoost worsen slightly in the current result files.

## Slide 8 - Per-paper error heterogeneity
- Put figure: `figures/worst_per_paper_rmse.png`.
- Put table: top 5 worst held-out papers by RMSE for MLP and KNN.
- Interpret: aggregate metrics hide paper-specific failures; high-FSR/older campaign domains dominate error.

## Slide 9 - Permutation importance
- Put figure: `figures/permutation_importance_top10.png`.
- Put table: top rows of `permutation_importance.csv`.
- Interpret: oxygen concentration dominates; sample dimensions, rig/facility, geometry, and material are secondary; negative importances should be treated as instability/noise, not physical irrelevance.

## Slide 10 - SHAP feature interpretation
- Put figure: `figures/shap_top_features.png`.
- Put table: top 10 from `shap_ranking_decision_tree.csv` and/or `shap_ranking_xgboost.csv`.
- Interpret: robust drivers are oxygen, size/geometry, pressure, facility/rig, and material. Facility/rig effects may encode experimental domain, not pure physics.

## Slide 11 - Recommended final figure/table checklist
- Put: concise checklist of all figures/tables above.
- Interpret: use this slide as the practical assembly guide for the oral deck.

## Slide 12 - Conclusions
- Put: key takeaways and next analysis.
- Interpret: claim paper-aware generalization cautiously; add repeated group validation or nested group CV before publication-level model ranking.
