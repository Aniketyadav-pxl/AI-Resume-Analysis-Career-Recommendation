"""M2 - Taxonomy-based skill extraction (alias aware, section aware).

Short or common-word terms (e.g. 'excel', 'ml', 'cv', 'r') are AMBIGUOUS: they are only
counted inside a SKILLS section, or (for resumes with no section headers at all) when
written with a capital letter. This avoids false positives such as 'excel in teamwork'.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from src.knowledge_base import SKILLS, SOFT_SKILLS
from src.sections import split_sections

AMBIGUOUS = {"r", "excel", "express", "cv", "ml", "dl", "tf", "ts", "py", "node", "torch"}
SOFT_PATTERNS = {
    "communication": r"communicat\w*",
    "teamwork": r"team ?work|team player|collaborat\w*",
    "problem solving": r"problem[- ]solving",
    "leadership": r"leadership|led a team|team lead",
    "agile": r"agile|scrum",
    "time management": r"time management",
}


@dataclass
class SkillResult:
    technical: list = field(default_factory=list)      # canonical names, sorted
    counts: dict = field(default_factory=dict)         # canonical -> mentions
    by_category: dict = field(default_factory=dict)    # category -> [skills]
    soft: list = field(default_factory=list)


@lru_cache(maxsize=1)
def _matcher():
    surface = {}
    for canon, (_cat, aliases) in SKILLS.items():
        surface[canon] = canon
        for a in aliases:
            surface[a] = canon
    terms = sorted(surface, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![\w+#.])(?:" + "|".join(re.escape(t) for t in terms) + r")(?![\w+#]|\.\w)", re.IGNORECASE)
    return pattern, surface


def slug(skill: str) -> str:
    """Model-friendly token for a canonical skill, e.g. 'c++' -> 'sk_cpp'."""
    s = skill.replace("++", "pp").replace("#", "sharp")
    return "sk_" + re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def find_skill_spans(text: str) -> list:
    """Return [(start, end, canonical)] for every accepted skill mention."""
    pattern, surface = _matcher()
    sections = split_sections(text)
    has_headers = any(s.name != "header" for s in sections)
    spans = []
    for sec in sections:
        in_skills = sec.name == "skills"
        for m in pattern.finditer(sec.text):
            term = m.group(0)
            if term.lower() in AMBIGUOUS:
                if sec.text[m.end():m.end() + 3].lstrip(" ").startswith(":"):
                    continue  # a category label such as 'Data/ML:' is not a skill mention
                if has_headers and not in_skills:
                    continue
                if not has_headers and not term[0].isupper():
                    continue
                if term.lower() == "r" and term != "R":
                    continue
            spans.append((sec.start + m.start(), sec.start + m.end(), surface[term.lower()]))
    return spans


def extract_skills(text: str) -> SkillResult:
    counts = {}
    for _s, _e, canon in find_skill_spans(text):
        counts[canon] = counts.get(canon, 0) + 1
    by_cat = {}
    for skill in sorted(counts):
        by_cat.setdefault(SKILLS[skill][0], []).append(skill)
    low = text.lower()
    soft = [s for s in SOFT_SKILLS if re.search(SOFT_PATTERNS.get(s, re.escape(s)), low)]
    return SkillResult(technical=sorted(counts), counts=counts, by_category=by_cat, soft=soft)


def evaluate_extractor(df) -> dict:
    """Micro precision/recall/F1 of extracted skills vs the synthetic ground truth."""
    tp = fp = fn = 0
    for text, truth in zip(df["resume_text"], df["skills_truth"]):
        pred, gold = set(extract_skills(text).technical), set(truth.split("|"))
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "n_resumes": len(df)}
