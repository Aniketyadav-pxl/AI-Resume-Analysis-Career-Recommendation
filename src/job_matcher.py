"""M6 - Job matching with TF-IDF cosine similarity + weighted skill coverage.

match score = 100 * (0.4 * normalised cosine similarity + 0.6 * weighted skill coverage)
  cosine   : TF-IDF vectors (fit on the job corpus) of the preprocessed resume vs each job description
  coverage : (2 * required skills matched + nice-to-have matched) / (2 * required + nice-to-have)
Cosine is divided by the best cosine in the result set so scores are comparable across jobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.preprocessing import prepare_many, prepare_model_text
from src.skill_extractor import SkillResult

COSINE_WEIGHT, COVERAGE_WEIGHT = 0.4, 0.6


@dataclass
class JobMatch:
    job_id: str
    title: str
    company: str
    location: str
    role_family: str
    min_experience: int
    match_score: float
    cosine: float
    coverage: float
    matched_required: list = field(default_factory=list)
    missing_required: list = field(default_factory=list)
    matched_nice: list = field(default_factory=list)
    missing_nice: list = field(default_factory=list)


def load_jobs(path=None) -> pd.DataFrame:
    path = path or config.JOBS_CSV
    if not path.exists():
        from src.job_generator import generate_jobs
        return generate_jobs()
    return pd.read_csv(path, encoding="utf-8")


class JobMatcher:
    def __init__(self, jobs: pd.DataFrame):
        self.jobs = jobs.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 1))
        self.job_matrix = self.vectorizer.fit_transform(prepare_many(self.jobs["description"].tolist()))

    def match(self, resume_text: str, skills: SkillResult, top_n: int = 10, location: str | None = None) -> list:
        have = set(skills.technical)
        cos = cosine_similarity(self.vectorizer.transform([prepare_model_text(resume_text)]), self.job_matrix)[0]
        norm = cos / cos.max() if cos.max() > 0 else cos
        out = []
        for i, row in self.jobs.iterrows():
            if location and location != "Any" and row["location"] != location:
                continue
            req, nice = row["required_skills"].split("|"), (row["nice_to_have"].split("|") if isinstance(row["nice_to_have"], str) else [])
            req_hit, nice_hit = [s for s in req if s in have], [s for s in nice if s in have]
            denom = 2 * len(req) + len(nice)
            coverage = (2 * len(req_hit) + len(nice_hit)) / denom if denom else 0.0
            score = 100 * (COSINE_WEIGHT * float(norm[i]) + COVERAGE_WEIGHT * coverage)
            out.append(JobMatch(
                job_id=row["job_id"], title=row["title"], company=row["company"], location=row["location"],
                role_family=row["role_family"], min_experience=int(row["min_experience"]),
                match_score=round(score, 1), cosine=round(float(cos[i]), 3), coverage=round(coverage, 3),
                matched_required=req_hit, missing_required=[s for s in req if s not in have],
                matched_nice=nice_hit, missing_nice=[s for s in nice if s not in have]))
        out.sort(key=lambda m: m.match_score, reverse=True)
        return out[:top_n]
