"""M8 - SQLite history of analyses.

Privacy: the resume TEXT is never stored, only derived results (scores, skills, recommendations).
The path can be overridden with the RESUME_AI_DB environment variable. If the default location
is read-only (some hosts), a temp-directory database is used instead.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import config

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    quality_score REAL NOT NULL,
    top_role TEXT NOT NULL,
    top_probability REAL NOT NULL,
    recommendations TEXT NOT NULL,
    skills TEXT NOT NULL,
    top_jobs TEXT NOT NULL,
    gap TEXT NOT NULL,
    model_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
"""
JSON_COLUMNS = ("recommendations", "skills", "top_jobs", "gap")


def resolve_path(db_path=None) -> Path:
    path = Path(db_path or os.environ.get("RESUME_AI_DB") or config.DB_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)):
            pass
        return path
    except (OSError, sqlite3.Error):
        return Path(tempfile.gettempdir()) / "resume_ai_history.db"


def _connect(db_path=None) -> sqlite3.Connection:
    conn = sqlite3.connect(resolve_path(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return conn


def save_analysis(record: dict, db_path=None) -> int:
    """Insert one analysis record (see pipeline.AnalysisResult.to_record) and return its id."""
    values = {k: (json.dumps(v) if k in JSON_COLUMNS else v) for k, v in record.items()}
    values["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cols = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute(f"INSERT INTO analyses ({cols}) VALUES ({marks})", list(values.values()))
        return int(cur.lastrowid)


def _row(row: sqlite3.Row) -> dict:
    d = dict(row)
    for c in JSON_COLUMNS:
        d[c] = json.loads(d[c])
    return d


def list_analyses(limit: int = 100, db_path=None) -> list:
    with closing(_connect(db_path)) as conn:
        rows = conn.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def get_analysis(analysis_id: int, db_path=None):
    with closing(_connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    return _row(row) if row else None


def delete_analysis(analysis_id: int, db_path=None) -> bool:
    with closing(_connect(db_path)) as conn, conn:
        return conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,)).rowcount > 0


def clear_all(db_path=None) -> int:
    with closing(_connect(db_path)) as conn, conn:
        return conn.execute("DELETE FROM analyses").rowcount
