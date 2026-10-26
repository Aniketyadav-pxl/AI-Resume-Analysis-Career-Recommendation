import json

import joblib
import numpy as np
import pytest

import config
from src.data_generator import generate_dataset
from src.knowledge_base import ROLES
from src.model_store import is_compatible, load_bundle
from src.recommender import confidence_label, recommend
from src.text_extractor import extract_from_path
from src.train import top_k_accuracy, train

EXPECTED = {  # sample file stem -> role
    "sample_data_scientist": "Data Scientist", "sample_frontend_developer": "Frontend Developer",
    "sample_devops_engineer": "DevOps Engineer", "sample_cybersecurity_analyst": "Cybersecurity Analyst",
    "sample_mobile_app_developer": "Mobile App Developer", "sample_data_analyst": "Data Analyst",
    "sample_qa_test_engineer": "QA / Test Engineer", "sample_backend_developer": "Backend Developer",
    "sample_machine_learning_engineer": "Machine Learning Engineer", "sample_full_stack_developer": "Full Stack Developer",
    "handwritten_ml_student": {"Data Scientist", "Machine Learning Engineer"},
    "handwritten_frontend": {"Frontend Developer", "Full Stack Developer"},
    "handwritten_devops": {"DevOps Engineer"},
}


def test_top_k_accuracy_helper():
    proba = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    assert top_k_accuracy(proba, ["a", "b", "c"], ["b", "a"], k=1) == 0.0
    assert top_k_accuracy(proba, ["a", "b", "c"], ["b", "a"], k=2) == 0.5
    assert top_k_accuracy(proba, ["a", "b", "c"], ["b", "a"], k=3) == 1.0


@pytest.fixture(scope="module")
def small_run():
    return train(generate_dataset(300, seed=5), save=False, verbose=False)


def test_training_pipeline_produces_valid_metrics(small_run):
    m = small_run["metrics"]
    assert m["best_model"] in m["cv_5fold_on_train"]
    assert m["holdout_test"]["accuracy"] > 0.7 and m["holdout_test"]["accuracy"] > 5 * m["baselines"]["majority_class_accuracy"]
    assert m["holdout_test"]["top3_accuracy"] >= m["holdout_test"]["accuracy"]
    assert m["dataset"]["n_train"] + m["dataset"]["n_test"] == 300
    cm = np.array(m["confusion_matrix"]["matrix"])
    assert cm.sum() == m["dataset"]["n_test"] and len(cm) == 10
    assert set(m["top_terms_per_role"]) == set(ROLES)


def test_saved_artifacts_consistent():
    metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    bundle = joblib.load(config.MODEL_PATH)
    assert is_compatible(bundle)
    assert bundle["meta"]["holdout_f1_macro"] == metrics["holdout_test"]["f1_macro"]
    assert metrics["holdout_test"]["f1_macro"] > metrics["baselines"]["rule_based_core_skill_overlap"]["f1_macro"]
    assert metrics["holdout_test"]["f1_macro"] < 0.995, "suspiciously perfect - check for data leakage"


@pytest.mark.parametrize("stem,expected", EXPECTED.items())
def test_unseen_resumes_get_right_role_in_top3(bundle, stem, expected):
    path = next(config.SAMPLE_DIR.glob(stem + ".*"))
    recs, probs = recommend(extract_from_path(path).text, bundle)
    roles = {r.role for r in recs}
    expected = {expected} if isinstance(expected, str) else expected
    assert roles & expected, (stem, [(r.role, round(r.probability, 2)) for r in recs])
    assert len(recs) == 3 and recs[0].probability >= recs[1].probability >= recs[2].probability
    assert abs(sum(probs.values()) - 1) < 1e-6


def test_recommendation_details(bundle):
    text = extract_from_path(config.SAMPLE_DIR / "handwritten_devops.txt").text
    top = recommend(text, bundle)[0][0]
    assert top.role == "DevOps Engineer" and "docker" in top.matched_core and 0 < top.core_coverage <= 1
    assert set(top.matched_core) | set(top.missing_core) == set(ROLES[top.role]["core"])


def test_committed_model_loads_without_retraining():
    _, status = load_bundle()
    assert status == "loaded", status  # requirements.txt pins the scikit-learn version the model was trained with


def test_confidence_labels():
    assert confidence_label(0.8, 0.1) == "High" and confidence_label(0.5, 0.4) == "Medium" and confidence_label(0.3, 0.25) == "Low"


def test_model_store_recovers_from_bad_files(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a pickle")
    bundle, status = load_bundle(bad)
    assert status.startswith("retrained") and is_compatible(bundle)
    missing, status2 = load_bundle(tmp_path / "none.joblib")
    assert "missing" in status2 and (tmp_path / "none.joblib").exists()
