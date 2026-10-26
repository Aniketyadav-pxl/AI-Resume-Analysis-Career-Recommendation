import json
import sqlite3

import pytest

import config
from src import database as db
from src.job_matcher import JobMatcher, load_jobs
from src.pipeline import analyze_file, analyze_resume
from src.text_extractor import ExtractionError


@pytest.fixture(scope="module")
def matcher():
    return JobMatcher(load_jobs())


@pytest.fixture
def path(tmp_path):
    return tmp_path / "t.db"


@pytest.fixture(scope="module")
def result(bundle, matcher):
    p = config.SAMPLE_DIR / "handwritten_devops.txt"
    return analyze_file(p.read_bytes(), p.name, bundle, matcher)


def test_end_to_end_pipeline(result):
    assert result.recommendations[0].role == "DevOps Engineer"
    assert 0 < result.quality.score <= 100 and result.job_matches and result.gap.role == "DevOps Engineer"
    assert len(result.recommendations) == 3 and not result.warnings


def test_pipeline_warns_on_unreadable_skills(bundle, matcher):
    r = analyze_resume("I like long walks and cooking dinner for my family every weekend evening.", bundle, matcher)
    assert any("Very few technical skills" in w for w in r.warnings)


def test_pipeline_bad_file_raises_friendly_error(bundle, matcher):
    with pytest.raises(ExtractionError):
        analyze_file(b"", "x.pdf", bundle, matcher)


def test_save_get_list_delete_roundtrip(result, path):
    rec = result.to_record()
    assert "resume_text" not in rec and "text" not in rec  # privacy: raw text never stored
    new_id = db.save_analysis(rec, path)
    got = db.get_analysis(new_id, path)
    assert got["top_role"] == "DevOps Engineer" and got["skills"]["technical"] == result.skills.technical
    assert got["recommendations"][0]["role"] == "DevOps Engineer" and got["created_at"].endswith("+00:00")
    assert [a["id"] for a in db.list_analyses(db_path=path)] == [new_id]
    assert db.delete_analysis(new_id, path) is True and db.delete_analysis(new_id, path) is False
    assert db.get_analysis(new_id, path) is None


def test_list_order_limit_and_clear(result, path):
    ids = [db.save_analysis(result.to_record(), path) for _ in range(5)]
    listed = db.list_analyses(limit=3, db_path=path)
    assert [a["id"] for a in listed] == ids[::-1][:3]
    assert db.clear_all(path) == 5 and db.list_analyses(db_path=path) == []


def test_sql_injection_is_inert(result, path):
    rec = result.to_record() | {"filename": "x'); DROP TABLE analyses;--"}
    new_id = db.save_analysis(rec, path)
    assert db.get_analysis(new_id, path)["filename"].startswith("x'); DROP")
    assert db.list_analyses(db_path=path)


def test_schema_and_env_override(monkeypatch, tmp_path, result):
    target = tmp_path / "env.db"
    monkeypatch.setenv("RESUME_AI_DB", str(target))
    db.save_analysis(result.to_record())
    assert target.exists()
    with sqlite3.connect(target) as c:
        assert c.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
        assert {r[1] for r in c.execute("PRAGMA table_info(analyses)")} >= {"quality_score", "top_role", "gap"}


def test_unwritable_location_falls_back_to_temp(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, so nothing can be created beneath me")  # portable on Windows/Linux
    p = db.resolve_path(blocker / "sub" / "x.db")
    assert p.name == "resume_ai_history.db"
