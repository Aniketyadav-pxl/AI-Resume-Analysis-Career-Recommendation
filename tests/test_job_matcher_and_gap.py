import pandas as pd
import pytest

import config
from src.job_generator import generate_jobs
from src.job_matcher import JobMatcher, load_jobs
from src.knowledge_base import ROLES, SKILLS
from src.skill_extractor import SkillResult, extract_skills
from src.skill_gap import analyze_gap
from src.text_extractor import extract_from_path


@pytest.fixture(scope="module")
def matcher():
    return JobMatcher(load_jobs())


def test_jobs_csv_schema_and_validity():
    df = pd.read_csv(config.JOBS_CSV)
    assert list(df.columns) == ["job_id", "title", "company", "location", "role_family", "min_experience",
                                "required_skills", "nice_to_have", "description"]
    assert len(df) == 60 and df["job_id"].is_unique and set(df["role_family"]) == set(ROLES)
    for col in ("required_skills", "nice_to_have"):
        assert all(s in SKILLS for cell in df[col] for s in cell.split("|"))
    assert generate_jobs().equals(generate_jobs())


@pytest.mark.parametrize("stem,family", [("handwritten_devops", "DevOps Engineer"), ("handwritten_frontend", "Frontend Developer"),
                                         ("sample_cybersecurity_analyst", "Cybersecurity Analyst"), ("sample_data_analyst", "Data Analyst")])
def test_relevant_jobs_rank_high(matcher, stem, family):
    text = extract_from_path(next(config.SAMPLE_DIR.glob(stem + ".*"))).text
    top = matcher.match(text, extract_skills(text), top_n=5)
    assert any(m.role_family == family for m in top[:3]), [(m.title, m.match_score) for m in top]


def test_scores_sorted_bounded_and_consistent(matcher):
    text = extract_from_path(config.SAMPLE_DIR / "handwritten_devops.txt").text
    res = matcher.match(text, extract_skills(text), top_n=60)
    scores = [m.match_score for m in res]
    assert scores == sorted(scores, reverse=True) and all(0 <= s <= 100 for s in scores)
    m = res[0]
    assert set(m.matched_required) | set(m.missing_required) == set(
        matcher.jobs.set_index("job_id").loc[m.job_id, "required_skills"].split("|"))


def test_location_filter_and_empty_resume(matcher):
    text = "SKILLS\nPython, SQL"
    res = matcher.match(text, extract_skills(text), top_n=100, location="Pune")
    assert res and all(m.location == "Pune" for m in res)
    empty = matcher.match("hello world", SkillResult(), top_n=3)
    assert len(empty) == 3 and all(m.match_score == 0 for m in empty)


def test_skill_gap_math_and_ordering():
    have = SkillResult(technical=["docker", "git", "linux"])
    gap = analyze_gap("DevOps Engineer", have)
    core = ROLES["DevOps Engineer"]["core"]
    assert set(gap.core_have) == {"docker", "git", "linux"} and set(gap.core_missing) == set(core) - {"docker", "git", "linux"}
    assert gap.readiness == round(100 * 0.7 * 3 / len(core), 1)
    tiers = [p["tier"] for p in gap.priorities]
    assert tiers == sorted(tiers, key=lambda t: t != "core")  # core gaps always listed first
    assert all(p["suggestion"].startswith("Learn ") for p in gap.priorities)


def test_skill_gap_perfect_and_invalid():
    full = SkillResult(technical=ROLES["QA / Test Engineer"]["core"] + ROLES["QA / Test Engineer"]["secondary"])
    g = analyze_gap("QA / Test Engineer", full)
    assert g.readiness == 100 and not g.priorities
    with pytest.raises(ValueError):
        analyze_gap("Astronaut", full)
