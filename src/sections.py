"""Resume section detection (shared by skill extraction and quality analysis)."""
from __future__ import annotations

import re
from dataclasses import dataclass

HEADER_MAP = {
    "summary": "summary", "professional summary": "summary", "objective": "summary",
    "career objective": "summary", "profile": "summary", "about me": "summary",
    "skills": "skills", "technical skills": "skills", "key skills": "skills",
    "core competencies": "skills", "skill set": "skills", "technologies": "skills",
    "tools and technologies": "skills",
    "experience": "experience", "work experience": "experience", "professional experience": "experience",
    "employment history": "experience", "work history": "experience", "internship": "experience",
    "internships": "experience", "internship experience": "experience",
    "projects": "projects", "academic projects": "projects", "personal projects": "projects",
    "key projects": "projects",
    "education": "education", "academic background": "education", "qualifications": "education",
    "certifications": "certifications", "certification": "certifications", "courses": "certifications",
    "licenses": "certifications",
    "achievements": "achievements", "awards": "achievements", "accomplishments": "achievements",
}
_INLINE = re.compile(r"^\s*([A-Za-z &]{3,30}?)\s*[:\-]\s*\S")


@dataclass
class Section:
    name: str  # summary | skills | experience | projects | education | certifications | achievements | header
    start: int
    end: int
    text: str


def _header_name(line: str):
    """Return (section, is_inline) if the line is a section header."""
    stripped = line.strip().rstrip(":").strip()
    key = re.sub(r"\s+", " ", stripped.lower().replace("&", "and"))
    if 0 < len(stripped) <= 40 and key in HEADER_MAP:
        return HEADER_MAP[key], False
    m = _INLINE.match(line)
    if m:
        key = re.sub(r"\s+", " ", m.group(1).lower().replace("&", "and"))
        if key in HEADER_MAP:
            return HEADER_MAP[key], True
    return None, False


def split_sections(text: str) -> list:
    """Split text into Section objects with offsets into the original string."""
    marks, pos = [], 0
    for line in text.splitlines(keepends=True):
        name, inline = _header_name(line)
        if name:
            marks.append((pos, name))
        pos += len(line)
    if not marks:
        return [Section("header", 0, len(text), text)]
    sections = []
    if marks[0][0] > 0:
        sections.append(Section("header", 0, marks[0][0], text[: marks[0][0]]))
    for i, (start, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        sections.append(Section(name, start, end, text[start:end]))
    return sections


def section_names(text: str) -> set:
    return {s.name for s in split_sections(text)} - {"header"}
