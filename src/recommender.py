"""M5 - Top-k career recommendations from the trained classifier."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.knowledge_base import ROLES
from src.preprocessing import prepare_model_text
from src.skill_extractor import SkillResult, extract_skills


@dataclass
class Recommendation:
    role: str
    probability: float
    description: str
    matched_core: list = field(default_factory=list)
    missing_core: list = field(default_factory=list)
    matched_secondary: list = field(default_factory=list)
    core_coverage: float = 0.0


def confidence_label(top_probability: float, runner_up: float) -> str:
    margin = top_probability - runner_up
    if top_probability >= 0.6 and margin >= 0.25:
        return "High"
    if top_probability >= 0.4:
        return "Medium"
    return "Low"


def recommend(text: str, bundle: dict, skills: SkillResult | None = None, top_k: int = 3) -> tuple:
    """Return (recommendations, all_probabilities dict). Probabilities come from predict_proba."""
    skills = skills or extract_skills(text)
    pipe = bundle["pipeline"]
    proba = pipe.predict_proba([prepare_model_text(text)])[0]
    classes = list(pipe.classes_)
    order = np.argsort(-proba)[:top_k]
    have = set(skills.technical)
    recs = []
    for i in order:
        role = classes[i]
        core, sec = ROLES[role]["core"], ROLES[role]["secondary"]
        matched = [s for s in core if s in have]
        recs.append(Recommendation(
            role=role, probability=float(proba[i]), description=ROLES[role]["description"],
            matched_core=matched, missing_core=[s for s in core if s not in have],
            matched_secondary=[s for s in sec if s in have], core_coverage=len(matched) / len(core)))
    return recs, {c: float(p) for c, p in zip(classes, proba)}
