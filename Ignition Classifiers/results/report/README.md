# Ignition classification evaluation report

Interpolation evaluates similar row-level conditions; extrapolation holds out entire canonical
papers and is the relevant evidence for transfer to unseen campaigns. Grouped nested tuning,
frozen validation thresholds, persisted outer predictions, and paper-cluster bootstrap intervals
protect the distinction.

## Selected models

- Interpolation: `xgb_all_unweighted`
- Extrapolation: `xgb_physics_unweighted`

These champions answer different scientific questions. Fold uncertainty, paired deltas, per-paper
variation, and calibration figures must be considered with point estimates. LOPO is a robustness
analysis, not the sole selection basis. Database heterogeneity, sparse features, campaign effects,
and observational sampling limit causal or universal claims.

## Integrity

Evaluation integrity overall: `True`. Failed candidate/protocol combinations are
excluded and documented in `../evaluation/integrity_checks.json`.

## Exact commands

```bash
cd "Ignition Classifiers"
python fable_splits.py --data ../Microgravity_Database_reduced.csv --out results/splits --n-seeds 3 --n-group-folds 5 --n-row-folds 5
python fable_evaluate.py --data ../Microgravity_Database_reduced.csv --splits results/splits --config configs/candidates.yaml --out results/evaluation --search-iterations 40 --inner-group-folds 3
python fable_select.py --evaluation results/evaluation --policy configs/selection_policy.yaml --out results/selection.json
python fable_refit.py --data ../Microgravity_Database_reduced.csv --selection results/selection.json --champion interpolation --out artifacts/interpolation_champion
python fable_refit.py --data ../Microgravity_Database_reduced.csv --selection results/selection.json --champion extrapolation --out artifacts/extrapolation_champion
python fable_report.py --data ../Microgravity_Database_reduced.csv --evaluation results/evaluation --selection results/selection.json --artifacts artifacts --out results/report
```

Evaluation artifacts are unbiased comparison evidence, not deployable models. Refit artifacts are
deployable models, not unbiased evaluation evidence.
