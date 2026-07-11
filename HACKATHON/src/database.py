"""
database.py
------------
Yeh module SQLite database ko manage karta hai. Isme candidates ka data,
unke AI predictions, aur Human-in-the-Loop (HITL) decisions (approve/
reject/modify) store hote hain.
"""

import sqlite3
import json
from contextlib import contextmanager

from config import DATABASE_PATH
from src.utils import current_timestamp


@contextmanager
def get_connection():
    """Database connection ko context manager ke zariye safely handle karta hai."""
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Database aur tables create karta hai agar exist nahi karte."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY,
                candidate_name TEXT,
                file_name TEXT,
                resume_text TEXT,
                jd_snapshot TEXT,
                features_json TEXT,
                matched_skills_json TEXT,
                missing_skills_json TEXT,
                ai_probability REAL,
                ai_confidence REAL,
                ai_recommendation TEXT,
                llm_summary TEXT,
                llm_explanation TEXT,
                hitl_decision TEXT DEFAULT 'Pending',
                hitl_notes TEXT DEFAULT '',
                hitl_modified_score REAL,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        conn.commit()


def insert_candidate(record: dict):
    """Naye candidate ka pura record database mein insert karta hai."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO candidates (
                candidate_id, candidate_name, file_name, resume_text, jd_snapshot,
                features_json, matched_skills_json, missing_skills_json,
                ai_probability, ai_confidence, ai_recommendation,
                llm_summary, llm_explanation, hitl_decision, hitl_notes,
                hitl_modified_score, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["candidate_id"],
                record["candidate_name"],
                record["file_name"],
                record["resume_text"],
                record["jd_snapshot"],
                json.dumps(record["features"]),
                json.dumps(record["matched_skills"]),
                json.dumps(record["missing_skills"]),
                record["ai_probability"],
                record["ai_confidence"],
                record["ai_recommendation"],
                record["llm_summary"],
                record["llm_explanation"],
                record.get("hitl_decision", "Pending"),
                record.get("hitl_notes", ""),
                record.get("hitl_modified_score"),
                current_timestamp(),
                current_timestamp(),
            ),
        )


def update_hitl_decision(candidate_id: str, decision: str, notes: str = "", modified_score: float = None):
    """HR reviewer ka HITL decision (Approve/Reject/Modify) update karta hai."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE candidates
            SET hitl_decision = ?, hitl_notes = ?, hitl_modified_score = ?, updated_at = ?
            WHERE candidate_id = ?
            """,
            (decision, notes, modified_score, current_timestamp(), candidate_id),
        )


def get_all_candidates() -> list:
    """Sab candidates ka data list of dicts ki soorat mein return karta hai."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM candidates ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_candidate_by_id(candidate_id: str) -> dict:
    """Ek specific candidate ka record nikalta hai."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_candidate(candidate_id: str):
    """Ek candidate record ko database se delete karta hai."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM candidates WHERE candidate_id = ?", (candidate_id,))


def clear_all_candidates():
    """Poora candidates table clear karta hai (naya batch shuru karne ke liye)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM candidates")
