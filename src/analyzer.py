"""M3 - Rule-based resume quality analysis (0-100 score + actionable suggestions).

This score is a transparent heuristic, NOT a machine-learning output. The ML component of
the project is the career classifier (src/train.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.sections import section_names
from src.skill_extractor import SkillResult

ACTION_VERBS = {
    "developed", "built", "implemented", "designed", "led", "improved", "reduced", "increased",
    "automated", "deployed", "created", "managed", "optimized", "optimised", "analyzed", "analysed",
    "collaborated", "maintained", "delivered", "integrated", "tested", "launched", "migrated",
    "configured", "documented", "trained", "evaluated", "provisioned", "containerised", "containerized",
}
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(\+?\d[\d\s\-()]{8,}\d)")
_PROFILE = re.compile(r"(linkedin\.com|github\.com)", re.I)
_METRIC = re.compile(r"\d+\s?%|\$\s?\d|\b\d[\d,]*\+?\s(users|customers|clients|requests|services|projects|ms|hours|minutes)\b", re.I)
_BULLET = re.compile(r"^\s*[-*\u2022]\s+", re.M)

CORE_SECTIONS = ["summary", "skills", "experience", "projects", "education"]
WEIGHTS = {"Structure": 25, "Skills": 20, "Experience & impact": 20, "Contact": 10,
           "Length": 10, "Skill breadth": 5, "Formatting": 10}


@dataclass
class QualityReport:
    score: float
    grade: str
    breakdown: dict = field(default_factory=dict)    # component -> (points, max)
    suggestions: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _grade(score: float) -> str:
    return "Excellent" if score >= 85 else "Good" if score >= 70 else "Fair" if score >= 50 else "Needs work"


def analyze_quality(text: str, skills: SkillResult) -> QualityReport:
    words = len(text.split())
    secs = section_names(text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    bullets = [ln for ln in lines if _BULLET.match(ln)]
    verb_bullets = [b for b in bullets if b.lstrip("-*\u2022 ").split(" ")[0].lower().strip(",.") in ACTION_VERBS]
    metrics = len(_METRIC.findall(text))
    has_email, has_phone, has_profile = bool(_EMAIL.search(text)), bool(_PHONE.search(text)), bool(_PROFILE.search(text))
    n_tech, n_cat = len(skills.technical), len(skills.by_category)
    longest = max((len(ln) for ln in lines), default=0)

    pts, tips = {}, []
    present = [s for s in CORE_SECTIONS if s in secs]
    pts["Structure"] = 5 * len(present)
    for s in CORE_SECTIONS:
        if s not in secs:
            tips.append(f"Add a clearly labelled '{s.title()}' section.")

    pts["Skills"] = round(20 * min(1, n_tech / 12), 1)
    if n_tech < 12:
        tips.append(f"List more relevant technical skills ({n_tech} found; aim for 12+).")

    bullet_pts = 8 * min(1, len(bullets) / 6)
    verb_pts = 8 * (len(verb_bullets) / len(bullets)) if bullets else 0
    metric_pts = 4 * min(1, metrics / 3)
    pts["Experience & impact"] = round(bullet_pts + verb_pts + metric_pts, 1)
    if len(bullets) < 6:
        tips.append("Describe your experience and projects in bullet points (at least 6).")
    if bullets and len(verb_bullets) / len(bullets) < 0.7:
        tips.append("Start bullets with strong action verbs (Built, Implemented, Reduced...).")
    if metrics < 3:
        tips.append("Quantify impact with numbers (e.g. 'reduced load time by 30%', '5,000 users').")

    pts["Contact"] = 4 * has_email + 3 * has_phone + 3 * has_profile
    if not has_email:
        tips.append("Add a professional email address.")
    if not has_phone:
        tips.append("Add a phone number.")
    if not has_profile:
        tips.append("Link your GitHub and/or LinkedIn profile.")

    if 200 <= words <= 700:
        pts["Length"] = 10
    else:
        pts["Length"] = round(10 * (min(words, 200) / 200 if words < 200 else max(0.0, 1 - (words - 700) / 700)), 1)
        tips.append(f"Resume length is {words} words; target 200-700 for a concise one-to-two page resume.")

    pts["Skill breadth"] = round(5 * min(1, n_cat / 3), 1)
    if n_cat < 3:
        tips.append("Show breadth: skills from at least 3 categories (e.g. languages, frameworks, tools).")

    fmt = (5 if bullets else 0) + (5 if longest <= 300 else 0)
    pts["Formatting"] = fmt
    if longest > 300:
        tips.append("Break very long paragraphs into short lines or bullets.")

    breakdown = {k: (round(v, 1), WEIGHTS[k]) for k, v in pts.items()}
    score = round(min(100.0, sum(v for v, _ in breakdown.values())), 1)
    stats = {"words": words, "bullets": len(bullets), "action_verb_bullets": len(verb_bullets),
             "metrics": metrics, "sections": sorted(secs), "technical_skills": n_tech}
    return QualityReport(score=score, grade=_grade(score), breakdown=breakdown, suggestions=tips, stats=stats)
