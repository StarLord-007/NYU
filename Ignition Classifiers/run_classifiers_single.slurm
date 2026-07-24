#!/bin/bash
#SBATCH --job-name=ignition-classifiers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00

set -euo pipefail

CONTAINER="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVERLAY="/scratch/me3144/pytorch-example/my_pytorch.ext3"
CLASSIFIER_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$CLASSIFIER_DIR")"

module purge
srun singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  --bind "${REPO_DIR}:${REPO_DIR}" \
  "$CONTAINER" \
  /bin/bash -lc "source /ext3/env.sh && cd \"${CLASSIFIER_DIR}\" && \
python fable_splits.py \
  --data ../Microgravity_Database_reduced.csv \
  --out results/splits \
  --n-seeds 3 \
  --n-group-folds 5 \
  --n-row-folds 5 && \
python fable_evaluate.py \
  --data ../Microgravity_Database_reduced.csv \
  --splits results/splits \
  --config configs/candidates.yaml \
  --out results/evaluation \
  --search-iterations 40 \
  --inner-group-folds 3 && \
python fable_select.py \
  --evaluation results/evaluation \
  --policy configs/selection_policy.yaml \
  --out results/selection.json && \
python fable_refit.py \
  --data ../Microgravity_Database_reduced.csv \
  --selection results/selection.json \
  --champion interpolation \
  --out artifacts/interpolation_champion && \
python fable_refit.py \
  --data ../Microgravity_Database_reduced.csv \
  --selection results/selection.json \
  --champion extrapolation \
  --out artifacts/extrapolation_champion && \
python fable_report.py \
  --data ../Microgravity_Database_reduced.csv \
  --evaluation results/evaluation \
  --selection results/selection.json \
  --artifacts artifacts \
  --out results/report"
