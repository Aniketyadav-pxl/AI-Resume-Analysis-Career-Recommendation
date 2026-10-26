from src.analyzer import WEIGHTS, analyze_quality
from src.skill_extractor import extract_skills
from src.text_extractor import extract_from_path
import config

GOOD = """JANE SAMPLE
jane@example.com | +91 90000 00000 | github.com/jane-sample

SUMMARY
Backend engineer with three years of experience building reliable services.

SKILLS
Languages: Python, SQL, Java
Backend: Flask, Django, REST API, PostgreSQL, Redis
DevOps: Docker, Git, AWS, Linux

EXPERIENCE
- Built a REST API serving 10,000 users using Flask and PostgreSQL.
- Reduced response time by 40% by adding Redis caching.
- Deployed services with Docker on AWS.
- Implemented CI/CD pipelines and automated testing.
- Designed database schemas and documented endpoints.
- Collaborated with a team of 5 engineers on code reviews.

PROJECTS
- Developed an inventory service with Django and PostgreSQL.

EDUCATION
B.Tech in Computer Science, Sample University, 2022
"""


def _report(text):
    return analyze_quality(text, extract_skills(text))


def test_weights_sum_to_100():
    assert sum(WEIGHTS.values()) == 100


def test_good_resume_scores_high_with_few_tips():
    r = _report(GOOD)
    assert r.score >= 85 and r.grade == "Excellent", r
    assert len(r.suggestions) <= 2


def test_weak_resume_scores_low_with_actionable_tips():
    r = _report("I am a hard working person. I know computers and want a job.")
    assert r.score < 30 and r.grade == "Needs work"
    joined = " ".join(r.suggestions)
    assert "Skills" in joined and "email" in joined and "bullet" in joined


def test_score_bounds_and_breakdown_consistency():
    for p in config.SAMPLE_DIR.glob("*.*"):
        r = _report(extract_from_path(p).text)
        assert 0 <= r.score <= 100
        assert abs(r.score - sum(v for v, _ in r.breakdown.values())) < 0.2
        assert all(0 <= v <= m for v, m in r.breakdown.values())


def test_missing_contact_and_sections_reduce_score():
    full = _report(GOOD).score
    no_contact = _report(GOOD.replace("jane@example.com | +91 90000 00000 | github.com/jane-sample", "JANE")).score
    no_edu = _report(GOOD.replace("EDUCATION", "OTHER")).score
    assert no_contact < full and no_edu < full
