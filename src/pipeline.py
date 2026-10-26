"""End-to-end orchestration: text -> skills -> quality -> recommendations -> jobs -> skill gap."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.analyzer import QualityReport, analyze_quality
from src.job_matcher import JobMatch, JobMatcher
from src.recommender import Recommendation, recommend
from src.skill_extractor import SkillResult, extract_skills
from src.skill_gap import SkillGap, analyze_gap
from src.text_extractor import ExtractionError, extract_text

MIN_SKILLS_FOR_PREDICTION = 2


@dataclass
class AnalysisResult:
    filename: str
    file_type: str
    n_pages: int
    word_count: int
    skills: SkillResult
    quality: QualityReport
    recommendations: list
    probabilities: dict
    job_matches: list
    gap: SkillGap
    model_name: str = ""
    warnings: list = field(default_factory=list)

    def to_record(self, top_jobs: int = 5) -> dict:
        """Flat dict for database.save_analysis (no resume text is included)."""
        top = self.recommendations[0]
        return {
            "filename": self.filename, "file_type": self.file_type, "word_count": self.word_count,
            "quality_score": self.quality.score, "top_role": top.role, "top_probability": round(top.probability, 4),
            "recommendations": [{"role": r.role, "probability": round(r.probability, 4)} for r in self.recommendations],
            "skills": {"technical": self.skills.technical, "soft": self.skills.soft},
            "top_jobs": [{"title": j.title, "company": j.company, "score": j.match_score} for j in self.job_matches[:top_jobs]],
            "gap": {"role": self.gap.role, "readiness": self.gap.readiness,
                    "core_missing": self.gap.core_missing, "secondary_missing": self.gap.secondary_missing},
            "model_name": self.model_name,
        }


def analyze_resume(text: str, bundle: dict, matcher: JobMatcher, filename: str = "pasted_text.txt",
                   file_type: str = "txt", n_pages: int = 1, warnings: list | None = None) -> AnalysisResult:
    skills = extract_skills(text)
    warnings = list(warnings or [])
    if len(skills.technical) < MIN_SKILLS_FOR_PREDICTION:
        warnings.append("Very few technical skills were detected, so the career prediction is unreliable. "
                        "Add a clear 'Skills' section and try again.")
    quality = analyze_quality(text, skills)
    recs, probs = recommend(text, bundle, skills)
    return AnalysisResult(
        filename=filename, file_type=file_type, n_pages=n_pages, word_count=len(text.split()), skills=skills,
        quality=quality, recommendations=recs, probabilities=probs, job_matches=matcher.match(text, skills, top_n=20),
        gap=analyze_gap(recs[0].role, skills), model_name=bundle.get("meta", {}).get("model_name", ""), warnings=warnings)


def analyze_file(data: bytes, filename: str, bundle: dict, matcher: JobMatcher) -> AnalysisResult:
    """Raises ExtractionError with a user-friendly message on bad files."""
    ex = extract_text(data, filename)
    return analyze_resume(ex.text, bundle, matcher, ex.filename, ex.file_type, ex.n_pages, ex.warnings)
