from pathlib import Path
import joblib
import json
from xgb_ignition_model_2 import load_clean, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

# Load cleaned data and reproduce the same split
df = load_clean(Path('database_xgb.csv'))
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df['ignition_binary'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Load fitted pipeline from artifacts_v3
pipe = joblib.load(Path('artifacts_v3') / 'xgb_ignition_model_v2.joblib')
proba_test = pipe.predict_proba(X_test)[:, 1]

# Get precision-recall thresholds
prec, rec, thr = precision_recall_curve(y_test, proba_test)
# thr is a numpy array of thresholds tested (length = len(prec)-1)
print(json.dumps({'n_thresholds': int(len(thr)), 'thresholds': [float(v) for v in thr]}))
