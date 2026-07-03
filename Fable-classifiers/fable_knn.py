"""
fable_knn.py — K-Nearest Neighbours member of the Fable benchmark.

Run:
    python fable_knn.py --data Microgravity_Database.xlsm            # champion
    python fable_knn.py --data Microgravity_Database.xlsm --tune     # re-tune
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.neighbors import KNeighborsClassifier

from fable_models import ModelSpec, run_model_cli


def make_knn(params: dict, seed: int) -> KNeighborsClassifier:
    # KNN is deterministic; seed unused (kept for interface uniformity).
    return KNeighborsClassifier(n_jobs=-1, **params)


PARAM_SPACE = {
    "n_neighbors": [5, 9, 15, 25, 40, 60, 80],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
    "leaf_size": [15, 30, 60],
}

# Winner of the group-aware random search (see knn_outputs/knn_search_history.csv).
CHAMPION_PARAMS = {"n_neighbors": 25, "weights": "distance", "p": 1, "leaf_size": 30}

SPEC = ModelSpec(
    key="knn",
    display="KNN",
    estimator_factory=make_knn,
    param_space=PARAM_SPACE,
    champion_params=CHAMPION_PARAMS,
    supports_sample_weight=False,  # weights applied at threshold stage instead
    notes="KNeighborsClassifier cannot use sample weights in fit; the shared "
          "paper x class weights are applied during threshold optimisation.",
)

if __name__ == "__main__":
    run_model_cli(SPEC)