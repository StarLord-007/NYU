"""
fable_svm.py — Support Vector Machine member of the Fable benchmark.

Run:
    python fable_svm.py --data ../Microgravity_Database_reduced.csv [--tune]

Note: SVC(probability=True) uses internal Platt scaling (row-level CV on the
training fold only), which is required so that .predict_proba matches the
common interface. This makes LOPO runs slower but leaks nothing. The current
implementation uses CalibratedClassifierCV around SVC to avoid the sklearn
deprecation warning for probability=True.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from fable_models import ModelSpec, run_model_cli


def make_svm(params: dict, seed: int) -> CalibratedClassifierCV:
    base = SVC(kernel="rbf", cache_size=500, random_state=seed, **params)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


PARAM_SPACE = {
    "C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
    "gamma": ["scale", 0.01, 0.03, 0.1, 0.3],
    "shrinking": [True, False],
}

CHAMPION_PARAMS = {"C": 3.0, "gamma": "scale", "shrinking": True}

SPEC = ModelSpec(
    key="svm",
    display="SVM (RBF)",
    estimator_factory=make_svm,
    param_space=PARAM_SPACE,
    champion_params=CHAMPION_PARAMS,
    supports_sample_weight=True,  # combined weights used in fit, as for XGBoost
)

if __name__ == "__main__":
    run_model_cli(SPEC)