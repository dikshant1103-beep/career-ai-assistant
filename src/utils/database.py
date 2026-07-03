"""Lightweight SQLite tracker for job applications & analysis history.

Schema is intentionally simple - it stores results of JD analyses and
application status so the user can revisit them from the CLI.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
@dataclass
class JobRecord:
    """A row in the `jobs` table."""

    id: Optional[int] = None
    title: str = ""
    company: str = ""
    jd_text: str = ""
    status: str = "saved"  # saved | analyzed | applied | interview | offer | rejected
    score: Optional[float] = None
    ats_score: Optional[float] = None
    keywords: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_row(self) -> tuple:
        return (
            self.title,
            self.company,
            self.jd_text,
            self.status,
            self.score,
            self.ats_score,
            json.dumps(self.keywords),
            json.dumps(self.missing_skills),
            self.notes,
            self.created_at,
            self.updated_at,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "JobRecord":
        return cls(
            id=row["id"],
            title=row["title"],
            company=row["company"],
            jd_text=row["jd_text"],
            status=row["status"],
            score=row["score"],
            ats_score=row["ats_score"],
            keywords=json.loads(row["keywords"] or "[]"),
            missing_skills=json.loads(row["missing_skills"] or "[]"),
            notes=row["notes"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# --------------------------------------------------------------------------- #
# Database wrapper
# --------------------------------------------------------------------------- #
class JobTracker:
    """SQLite-backed application tracker."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path: Path = Path(db_path) if db_path else get_settings().db_full_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # --------- low-level connection ---------
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    title           TEXT NOT NULL,
                    company         TEXT,
                    jd_text         TEXT,
                    status          TEXT DEFAULT 'saved',
                    score           REAL,
                    ats_score       REAL,
                    keywords        TEXT,
                    missing_skills  TEXT,
                    notes           TEXT,
                    created_at      TEXT,
                    updated_at      TEXT
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    role       TEXT,
                    content    TEXT,
                    feature    TEXT,
                    created_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_conv_feature ON conversations(feature);
                """
            )

    # --------- CRUD: jobs ---------
    def add_job(self, job: JobRecord) -> int:
        with self._conn() as con:
            cur = con.execute(
                """
                INSERT INTO jobs
                    (title, company, jd_text, status, score, ats_score,
                     keywords, missing_skills, notes, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                job.to_row(),
            )
            return cur.lastrowid

    def update_job(self, job: JobRecord) -> None:
        if job.id is None:
            raise ValueError("Cannot update a job without an id")
        job.updated_at = datetime.utcnow().isoformat()
        with self._conn() as con:
            con.execute(
                """
                UPDATE jobs
                   SET title=?, company=?, jd_text=?, status=?, score=?,
                       ats_score=?, keywords=?, missing_skills=?, notes=?,
                       updated_at=?
                 WHERE id=?
                """,
                (*job.to_row()[:-1], job.updated_at, job.id),
            )

    def list_jobs(self, status: Optional[str] = None) -> List[JobRecord]:
        q = "SELECT * FROM jobs"
        params: tuple = ()
        if status:
            q += " WHERE status = ?"
            params = (status,)
        q += " ORDER BY updated_at DESC"
        with self._conn() as con:
            rows = con.execute(q, params).fetchall()
        return [JobRecord.from_row(r) for r in rows]

    def get_job(self, job_id: int) -> Optional[JobRecord]:
        with self._conn() as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return JobRecord.from_row(row) if row else None

    def delete_job(self, job_id: int) -> None:
        with self._conn() as con:
            con.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    # --------- Conversations / memory ---------
    def log_message(self, role: str, content: str, feature: str) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT INTO conversations (role, content, feature, created_at) VALUES (?,?,?,?)",
                (role, content, feature, datetime.utcnow().isoformat()),
            )

    def recent_messages(self, feature: str, limit: int = 10) -> List[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT role, content, created_at FROM conversations "
                "WHERE feature=? ORDER BY id DESC LIMIT ?",
                (feature, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]
