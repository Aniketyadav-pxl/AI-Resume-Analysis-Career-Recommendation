import re
from src.data_generator import generate_dataset
from src.knowledge_base import ROLES, SKILLS

COLS = ["resume_id", "target_role", "experience_years", "education", "skills_truth", "resume_text"]


def test_schema_and_balance():
    df = generate_dataset(200, seed=1)
    assert list(df.columns) == COLS
    assert len(df) == 200 and df["resume_id"].is_unique
    assert set(df["target_role"]) == set(ROLES)
    assert df["target_role"].value_counts().nunique() == 1  # perfectly balanced
    assert df.isna().sum().sum() == 0


def test_ground_truth_skills_valid_and_present():
    df = generate_dataset(100, seed=2)
    for s in df["skills_truth"]:
        skills = s.split("|")
        assert len(skills) >= 4 and all(k in SKILLS for k in skills)


def test_no_real_personal_data():
    df = generate_dataset(100, seed=3)
    emails = df["resume_text"].str.findall(r"[\w.]+@[\w.]+").explode()
    assert emails.str.endswith("@example.com").all()
    assert not df["resume_text"].str.contains(r"\b\d{10}\b").any()  # no phone numbers


def test_reproducible():
    a, b = generate_dataset(50, seed=7), generate_dataset(50, seed=7)
    assert a.equals(b)


def test_role_not_leaked_in_text():
    df = generate_dataset(100, seed=4)
    leaks = [r for r, t in zip(df["target_role"], df["resume_text"]) if r.lower() in t.lower()]
    assert not leaks
