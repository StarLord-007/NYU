"""
fable_mlp.py — Multi-Layer Perceptron member of the Fable benchmark.

Run:
    python fable_mlp.py --data ../Microgravity_Database_reduced.csv [--tune]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.neural_network import MLPClassifier

from fable_models import ModelSpec, run_model_cli


def make_mlp(params: dict, seed: int) -> MLPClassifier:
    return MLPClassifier(max_iter=600, early_stopping=True, n_iter_no_change=25,
                         random_state=seed, **params)


PARAM_SPACE = {
    "hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (64, 64, 32)],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3],
    "activation": ["relu", "tanh"],
    "batch_size": [32, 64, 128],
}

CHAMPION_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "alpha": 1e-3,
    "learning_rate_init": 1e-3,
    "activation": "relu",
    "batch_size": 64,
}

SPEC = ModelSpec(
    key="mlp",
    display="MLP",
    estimator_factory=make_mlp,
    param_space=PARAM_SPACE,
    champion_params=CHAMPION_PARAMS,
    supports_sample_weight=False,  # sklearn MLP has no fit-time sample_weight
    notes="MLPClassifier cannot use sample weights in fit; the shared "
          "paper x class weights are applied during threshold optimisation. "
          "Early stopping uses a random row-level split of the training data "
          "only — the test papers are never seen.",
)

if __name__ == "__main__":
    run_model_cli(SPEC)