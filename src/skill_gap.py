"""M7 - Skill gap analysis against a target role."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.knowledge_base import ROLES, SKILLS, display_name
from src.skill_extractor import SkillResult

PROJECT_IDEAS = {
    "Programming": "solve 10 small problems and write 3 utility scripts with it",
    "Frontend": "rebuild a landing page or a small component library with it",
    "Backend": "build a CRUD service with authentication using it",
    "Database": "design a schema and write 20 queries against sample data",
    "Data/ML": "reproduce an end-to-end notebook on a public dataset with it",
    "Analytics": "build a dashboard from an open dataset with it",
    "DevOps": "containerise and deploy a sample app using it",
    "Mobile": "ship a small three-screen app with it",
    "Security": "complete a beginner lab or CTF exercise with it",
    "Testing": "write a test plan and automate 10 test cases with it",
}


def _transferability(skill: str) -> int:
    """How many roles list this skill (core or secondary): higher = more broadly useful."""
    return sum(skill in r["core"] or skill in r["secondary"] for r in ROLES.values())


@dataclass
class SkillGap:
    role: str
    readiness: float
    core_have: list = field(default_factory=list)
    core_missing: list = field(default_factory=list)
    secondary_have: list = field(default_factory=list)
    secondary_missing: list = field(default_factory=list)
    priorities: list = field(default_factory=list)  # dicts: skill, category, tier, suggestion


def analyze_gap(role: str, skills: SkillResult) -> SkillGap:
    if role not in ROLES:
        raise ValueError(f"Unknown role: {role}")
    have = set(skills.technical)
    core, sec = ROLES[role]["core"], ROLES[role]["secondary"]
    core_have, core_missing = [s for s in core if s in have], [s for s in core if s not in have]
    sec_have, sec_missing = [s for s in sec if s in have], [s for s in sec if s not in have]
    readiness = round(100 * (0.7 * len(core_have) / len(core) + 0.3 * len(sec_have) / len(sec)), 1)

    def entry(skill, tier):
        cat = SKILLS[skill][0]
        return {"skill": skill, "category": cat, "tier": tier, "transferability": _transferability(skill),
                "suggestion": f"Learn {display_name(skill)}: {PROJECT_IDEAS.get(cat, 'build a small project with it')}."}

    ordered = (sorted(core_missing, key=lambda s: (-_transferability(s), s)),
               sorted(sec_missing, key=lambda s: (-_transferability(s), s)))
    priorities = [entry(s, "core") for s in ordered[0]] + [entry(s, "nice-to-have") for s in ordered[1]]
    return SkillGap(role, readiness, core_have, core_missing, sec_have, sec_missing, priorities)
