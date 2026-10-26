"""Model loading strategy (used by the app, CLI and tests).

1. Load models/career_classifier.joblib (committed to the repo, ~0.3 MB).
2. If it is missing, corrupt, or was built with a different scikit-learn major.minor version or
   preprocessing version, quickly retrain a logistic-regression model from the synthetic CSV
   (a few seconds) and try to save it. The app therefore never crashes on a stale pickle.
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

import config
from src.data_generator import generate_dataset
from src.preprocessing import PREP_VERSION, prepare_many
from src.train import make_pipeline
from sklearn.linear_model import LogisticRegression


def _minor(v: str) -> tuple:
    return tuple(v.split(".")[:2])


def is_compatible(bundle: dict) -> bool:
    meta = bundle.get("meta", {})
    return (_minor(meta.get("sklearn_version", "0.0")) == _minor(sklearn.__version__)
            and meta.get("prep_version") == PREP_VERSION and "pipeline" in bundle)


def quick_bundle() -> dict:
    """Fast fallback: logistic regression on the synthetic data (no CV, no evaluation)."""
    if config.SYNTHETIC_RESUMES_CSV.exists():
        df = pd.read_csv(config.SYNTHETIC_RESUMES_CSV)
    else:
        df = generate_dataset(1500)
    X = np.array(prepare_many(df["resume_text"].tolist()), dtype=object)
    pipe = make_pipeline(LogisticRegression(C=10, max_iter=3000)).fit(X, df["target_role"].to_numpy())
    return {"pipeline": pipe, "classes": list(pipe.classes_),
            "meta": {"sklearn_version": sklearn.__version__, "prep_version": PREP_VERSION,
                     "model_name": "Logistic Regression (auto-retrained)", "n_resumes": int(len(df)),
                     "holdout_f1_macro": None, "holdout_accuracy": None, "trained_at": None}}


def load_bundle(path=None) -> tuple:
    """Return (bundle, status). status is 'loaded' or a message explaining why we retrained."""
    path = path or config.MODEL_PATH
    status = "loaded"
    try:
        if not path.exists():
            raise FileNotFoundError("model file missing")
        bundle = joblib.load(path)
        if not is_compatible(bundle):
            raise ValueError("saved model was built with a different scikit-learn/preprocessing version")
        return bundle, status
    except Exception as exc:  # corrupt pickle, missing file, version mismatch ...
        status = f"retrained automatically ({exc})"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = quick_bundle()
    try:
        joblib.dump(bundle, path, compress=3)
    except OSError:
        pass  # read-only file system (e.g. some hosting): keep the model in memory only
    return bundle, status
