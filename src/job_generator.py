"""Synthetic job postings (fictional companies) for the job matcher.  Run: python -m src.job_generator"""
import random

import pandas as pd

import config
from src.knowledge_base import ROLES, display_name

COMPANIES = ["Northwind Labs", "Acme Analytics", "Zenith Systems", "Bluepeak Tech", "Orbit Software",
             "Lumen Cloud", "Quark Digital", "Helix Works", "Nimbus Apps", "Vertex Solutions"]
LOCATIONS = ["Noida", "Gurugram", "Bengaluru", "Hyderabad", "Pune", "Remote"]
LEVELS = [("{r} Trainee", 0, 0), ("Junior {r}", 0, 1), ("Associate {r}", 1, 2),
          ("{r}", 2, 4), ("Senior {r}", 5, 7), ("{r}", 1, 3)]


def generate_jobs(per_role: int = 6, seed: int = 321) -> pd.DataFrame:
    rng = random.Random(seed)
    rows, n = [], 0
    for role, spec in ROLES.items():
        for pattern, lo, hi in LEVELS[:per_role]:
            n += 1
            core = spec["core"]
            req = sorted(rng.sample(core, max(4, round(len(core) * rng.uniform(0.6, 0.9)))))
            nice = sorted(rng.sample(spec["secondary"], min(len(spec["secondary"]), rng.randint(2, 4))))
            title, company, loc = pattern.format(r=role), rng.choice(COMPANIES), rng.choice(LOCATIONS)
            desc = (f"{company} is hiring a {title} in {loc}. You will work on {rng.choice(spec['nouns'])}. "
                    f"Required skills: {', '.join(display_name(s) for s in req)}. "
                    f"Nice to have: {', '.join(display_name(s) for s in nice)}.")
            rows.append({"job_id": f"J{n:03d}", "title": title, "company": company, "location": loc,
                         "role_family": role, "min_experience": rng.randint(lo, hi),
                         "required_skills": "|".join(req), "nice_to_have": "|".join(nice), "description": desc})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_jobs()
    config.DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(config.JOBS_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(df)} synthetic jobs -> {config.JOBS_CSV}")
