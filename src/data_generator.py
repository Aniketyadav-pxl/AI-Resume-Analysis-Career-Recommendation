"""Synthetic resume dataset generator (no real personal data).

Dataset schema (data/synthetic_resumes.csv)
-------------------------------------------
resume_id         str   e.g. R0001
target_role       str   label to predict (10 classes)
experience_years  int   0-9
education         str   degree string
skills_truth      str   '|' separated canonical skills (ground truth for evaluating the skill extractor)
resume_text       str   full plain-text resume (what the ML model sees)

Realism knobs (an earlier, cleaner version was trivially separable: 100% accuracy from the
skills section alone, and from project nouns alone - a data-leakage red flag):
  * each core skill is present with p = 0.60-0.82 depending on experience level
  * 18% career switchers (mostly a neighbouring role's skills), 22% hybrids
  * 50% of resumes get 2-5 random "noise" skills from unrelated areas
  * project themes: only 20% role-specific, 25% from a neighbouring role, 55% generic
  * 20% of skill mentions use an alias spelling (e.g. 'sklearn', 'k8s')

Run:  python -m src.data_generator --n 1500
"""
import argparse
import random

import pandas as pd

import config
from src.knowledge_base import (
    GENERIC_NOUNS, NEIGHBOURS, ROLES, SKILLS, SOFT_SKILLS, display_name,
)

LEVELS = {  # label: (years range, P(core skill present))
    "fresher": ((0, 0), 0.60),
    "junior": ((1, 2), 0.68),
    "mid": ((3, 5), 0.75),
    "senior": ((6, 9), 0.82),
}
LEVEL_WEIGHTS = [0.40, 0.30, 0.20, 0.10]
DEGREES = ["B.Tech in Computer Science", "BCA", "B.Sc in Information Technology",
           "M.Tech in Computer Science", "MCA", "B.E. in Information Technology"]
BULLETS = [
    "Developed {noun} using {s1} and {s2}.",
    "Implemented {noun} leveraging {s1}.",
    "Improved the performance of {noun} by applying {s1}.",
    "Collaborated with a team of {n} to deliver {noun} with {s1} and {s2}.",
    "Maintained and documented {noun} built on {s1}.",
]
SUMMARY = [
    "Motivated {lvl} professional with hands-on experience in {s1} and {s2}.",
    "Detail-oriented {lvl} developer skilled in {s1}, {s2} and {s3}.",
    "{lvl_cap} technologist passionate about building reliable solutions with {s1}.",
]


def _pick_skills(role: str, level: str, rng: random.Random) -> list:
    spec, p_core = ROLES[role], LEVELS[level][1]
    neighbour = rng.choice(NEIGHBOURS[role])
    r = rng.random()
    if r < 0.18:    # career switcher: mostly the neighbouring role's skills, a few of the labelled role's
        skills = {s for s in ROLES[neighbour]["core"] if rng.random() < p_core}
        skills |= {s for s in spec["core"] if rng.random() < 0.35}
    elif r < 0.40:  # hybrid: own core skills plus a good share of the neighbour's
        skills = {s for s in spec["core"] if rng.random() < p_core}
        skills |= {s for s in ROLES[neighbour]["core"] if rng.random() < 0.50}
    else:           # focused profile
        skills = {s for s in spec["core"] if rng.random() < p_core}
    while len(skills) < 4:  # guarantee a minimally informative resume
        skills.add(rng.choice(spec["core"] + ROLES[neighbour]["core"]))
    skills |= {s for s in spec["secondary"] if rng.random() < 0.30}
    if rng.random() < 0.50:  # unrelated noise skills (hobby projects, courses)
        skills |= set(rng.sample(list(SKILLS), rng.randint(2, 5)))
    return sorted(skills)


def _surface(skill: str, rng: random.Random) -> str:
    aliases = SKILLS[skill][1]
    if aliases and rng.random() < 0.20:
        return rng.choice(aliases).title() if len(aliases[0]) > 3 else rng.choice(aliases)
    return display_name(skill)


def build_resume(idx: int, role: str, rng: random.Random) -> dict:
    level = rng.choices(list(LEVELS), weights=LEVEL_WEIGHTS)[0]
    years = rng.randint(*LEVELS[level][0])
    skills = _pick_skills(role, level, rng)
    tech_pool = skills[:]

    def two():
        a, b = rng.sample(tech_pool, 2)
        return display_name(a), display_name(b)

    def noun():  # project themes are shared across related roles, so they are only weakly informative
        r = rng.random()
        if r < 0.20:
            return rng.choice(ROLES[role]["nouns"])
        if r < 0.45:
            return rng.choice(ROLES[rng.choice(NEIGHBOURS[role])]["nouns"])
        return rng.choice(GENERIC_NOUNS)

    top = rng.sample(tech_pool, 3)
    summary = rng.choice(SUMMARY).format(
        lvl=level, lvl_cap=level.capitalize(),
        s1=display_name(top[0]), s2=display_name(top[1]), s3=display_name(top[2]))

    # Skills section grouped by category
    by_cat = {}
    for s in skills:
        by_cat.setdefault(SKILLS[s][0], []).append(_surface(s, rng))
    soft = rng.sample(SOFT_SKILLS, rng.randint(1, 3))
    skill_lines = [f"{cat}: {', '.join(v)}" for cat, v in sorted(by_cat.items())]
    skill_lines.append("Soft Skills: " + ", ".join(w.title() for w in soft))

    def bullet():
        s1, s2 = two()
        return "- " + rng.choice(BULLETS).format(noun=noun(), s1=s1, s2=s2, n=rng.randint(3, 8))

    exp_lines = []
    n_jobs = 1 if years == 0 else min(3, 1 + years // 3)
    for j in range(n_jobs):
        title = "Intern" if years == 0 else rng.choice(["Associate", "Engineer", "Analyst", "Specialist"])
        exp_lines.append(f"{title}, Sample Company {rng.randint(1, 99)}")
        exp_lines += [bullet() for _ in range(rng.randint(2, 4))]

    proj_lines = []
    for _ in range(rng.randint(1, 3)):
        s1, s2 = two()
        proj_lines.append(f"- {noun().capitalize()} built with {s1} and {s2}.")

    text = "\n".join([
        f"CANDIDATE R{idx:04d}",
        f"Email: candidate{idx:04d}@example.com",
        "", "SUMMARY", summary,
        "", "SKILLS", *skill_lines,
        "", "EXPERIENCE", *exp_lines,
        "", "PROJECTS", *proj_lines,
        "", "EDUCATION", f"{rng.choice(DEGREES)}, Sample University, {2026 - years - rng.randint(0, 1)}",
    ])
    return {
        "resume_id": f"R{idx:04d}", "target_role": role, "experience_years": years,
        "education": text.rsplit("\n", 1)[-1].split(",")[0],
        "skills_truth": "|".join(skills), "resume_text": text,
    }


def generate_dataset(n: int = 1500, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = random.Random(seed)
    roles = list(ROLES)
    rows = [build_resume(i + 1, roles[i % len(roles)], rng) for i in range(n)]  # balanced classes
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = ap.parse_args()
    out = generate_dataset(args.n, args.seed)
    config.DATA_DIR.mkdir(exist_ok=True)
    out.to_csv(config.SYNTHETIC_RESUMES_CSV, index=False)
    print(f"Saved {len(out)} synthetic resumes -> {config.SYNTHETIC_RESUMES_CSV}")
    print(out["target_role"].value_counts().to_string())
