"""M4 - Train, compare and evaluate the career-role classifier.

Pipeline:  preprocessed text -> TF-IDF (1-2 grams) -> classifier
Protocol:  80/20 stratified hold-out split; 5-fold stratified CV on the 80% to compare
           candidate models (selection by macro-F1); final metrics on the untouched 20%;
           a second independent synthetic set (different seed) and a rule-based baseline
           are also reported. The deployed model is refit on all data.

Run:  python -m src.train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import config
from src.data_generator import generate_dataset
from src.knowledge_base import ROLES
from src.preprocessing import PREP_VERSION, prepare_many
from src.skill_extractor import extract_skills

LIMITATIONS = [
    "All training resumes are synthetic and rule-generated from the same skill lists the model is later evaluated on; accuracy on real resumes will be lower.",
    "Test data comes from the same generator as training data, so the test score measures in-distribution generalisation, not real-world performance.",
    "Only 10 roles are modelled; resumes outside these roles are still forced into one of them (the probabilities are relative, not an 'out-of-scope' detector).",
    "Roles overlap heavily in the data (e.g. Frontend / Full Stack, Data Scientist / ML Engineer), so the top-3 view is more meaningful than top-1.",
    "No demographic or personal attributes are used, but real-world hiring models need fairness audits that synthetic data cannot provide.",
]


def candidates() -> dict:
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(C=10, max_iter=3000),
        "Linear SVM (calibrated)": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=rs), cv=3),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=rs, n_jobs=1),
    }


def make_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)),
        ("clf", clf),
    ])


def top_k_accuracy(proba: np.ndarray, classes, y_true, k: int = 3) -> float:
    classes = np.asarray(classes)
    top = classes[np.argsort(-proba, axis=1)[:, :k]]
    return float(np.mean([t in row for t, row in zip(y_true, top)]))


def evaluate(pipe, X, y) -> dict:
    classes = pipe.classes_
    proba = pipe.predict_proba(X)
    pred = classes[proba.argmax(axis=1)]
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision_macro": round(precision_score(y, pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y, pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y, pred, average="macro", zero_division=0), 4),
        "top3_accuracy": round(top_k_accuracy(proba, classes, y, 3), 4),
        "log_loss": round(float(log_loss(y, proba, labels=classes)), 4),
    }


def rule_baseline(texts, y) -> dict:
    """Non-ML baseline: pick the role whose CORE skills are most covered by the extracted skills."""
    roles = list(ROLES)
    preds = []
    for t in texts:
        have = set(extract_skills(t).technical)
        scores = [len(have & set(ROLES[r]["core"])) / len(ROLES[r]["core"]) for r in roles]
        preds.append(roles[int(np.argmax(scores))])
    return {"accuracy": round(accuracy_score(y, preds), 4),
            "f1_macro": round(f1_score(y, preds, average="macro", zero_division=0), 4)}


def top_terms(pipe, n: int = 10) -> dict:
    """Most positive TF-IDF coefficients per role from a logistic regression (explainability)."""
    lr = pipe.named_steps["clf"] if isinstance(pipe.named_steps["clf"], LogisticRegression) else None
    vec = pipe.named_steps["tfidf"]
    if lr is None:
        return {}
    names = np.array(vec.get_feature_names_out())
    return {cls: [t.replace("sk_", "") for t in names[np.argsort(-lr.coef_[i])[:n]]] for i, cls in enumerate(lr.classes_)}


def train(df: pd.DataFrame | None = None, save: bool = True, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    if df is None:
        if not config.SYNTHETIC_RESUMES_CSV.exists():
            config.DATA_DIR.mkdir(exist_ok=True)
            generate_dataset(1500).to_csv(config.SYNTHETIC_RESUMES_CSV, index=False)
        df = pd.read_csv(config.SYNTHETIC_RESUMES_CSV)
    log(f"Preprocessing {len(df)} resumes with spaCy ...")
    X = np.array(prepare_many(df["resume_text"].tolist()), dtype=object)
    y = df["target_role"].to_numpy()

    X_tr, X_te, y_tr, y_te, txt_tr, txt_te = train_test_split(
        X, y, df["resume_text"].to_numpy(), test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)

    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_results = {}
    for name, clf in candidates().items():
        r = cross_validate(make_pipeline(clf), X_tr, y_tr, cv=cv, scoring=["accuracy", "f1_macro"])
        cv_results[name] = {
            "accuracy_mean": round(float(r["test_accuracy"].mean()), 4), "accuracy_std": round(float(r["test_accuracy"].std()), 4),
            "f1_macro_mean": round(float(r["test_f1_macro"].mean()), 4), "f1_macro_std": round(float(r["test_f1_macro"].std()), 4)}
        log(f"  CV {name:26s} acc={cv_results[name]['accuracy_mean']:.3f}  f1={cv_results[name]['f1_macro_mean']:.3f}")
    best = max(cv_results, key=lambda n: cv_results[n]["f1_macro_mean"])
    log(f"Best by CV macro-F1: {best}")

    pipe = make_pipeline(candidates()[best]).fit(X_tr, y_tr)
    test_metrics = evaluate(pipe, X_te, y_te)
    classes = list(pipe.classes_)
    pred = pipe.predict(X_te)
    report = classification_report(y_te, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=classes)
    baseline_dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    dummy_acc = round(accuracy_score(y_te, baseline_dummy.predict(X_te)), 4)
    rule = rule_baseline(txt_te, y_te)

    indep = generate_dataset(500, seed=777)
    ind_metrics = evaluate(pipe, np.array(prepare_many(indep["resume_text"].tolist()), dtype=object), indep["target_role"].to_numpy())
    log(f"Hold-out test: {test_metrics}")
    log(f"Independent synthetic set: {ind_metrics}")
    log(f"Baselines  majority={dummy_acc}  rule-based={rule}")

    terms = top_terms(pipe)
    final = make_pipeline(candidates()[best]).fit(X, y)  # deployed model uses all data
    if not terms:
        terms = top_terms(make_pipeline(candidates()["Logistic Regression"]).fit(X, y))

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_model": best, "cv_5fold_on_train": cv_results, "holdout_test": test_metrics,
        "independent_synthetic_test": ind_metrics,
        "baselines": {"majority_class_accuracy": dummy_acc, "rule_based_core_skill_overlap": rule},
        "per_class": {c: {k: round(v, 4) for k, v in report[c].items() if k != "support"} | {"support": int(report[c]["support"])} for c in classes},
        "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
        "top_terms_per_role": terms,
        "dataset": {"n_resumes": int(len(df)), "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "n_classes": len(classes),
                    "sha256": hashlib.sha256(pd.util.hash_pandas_object(df["resume_text"]).values.tobytes()).hexdigest()[:16]},
        "limitations": LIMITATIONS,
    }
    bundle = {"pipeline": final, "classes": classes,
              "meta": {"sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
                       "prep_version": PREP_VERSION, "model_name": best, "trained_at": metrics["trained_at"],
                       "n_resumes": int(len(df)), "holdout_f1_macro": test_metrics["f1_macro"],
                       "holdout_accuracy": test_metrics["accuracy"]}}
    if save:
        config.MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(bundle, config.MODEL_PATH, compress=3)
        config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        log(f"Saved model -> {config.MODEL_PATH.name} ({config.MODEL_PATH.stat().st_size / 1e3:.0f} KB) and metrics.json")
    return {"bundle": bundle, "metrics": metrics}


if __name__ == "__main__":
    argparse.ArgumentParser(description="Train the career classifier").parse_args()
    train()
    sys.exit(0)
