"""
analysis_db.py
================
A SECOND, separate SQLite database (Task 6) logging every /analyze call —
distinct from app/db/database.py (enrolled-user metadata) and
app/services/ai_models/embedding_store.py (speaker embeddings). Kept
separate rather than folded into the existing users table because it has
an entirely different shape (one row per CALL, not per user) and a
different lifecycle (append-only history vs. mutable user records).

Table: call_logs, fields exactly as specified: call_id, timestamp,
speaker_name, transcript, amount, urgency, spoof_score, similarity, risk.

NOTE: an earlier version of this module used a table named "analysis"
(same columns except speaker_name). init_db() migrates any existing rows
from that table into call_logs once, so demo data recorded before this
change isn't silently lost — see _migrate_legacy_table().
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from app.config import ANALYSIS_DB_PATH, DATA_DIR
import os


def get_connection() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(ANALYSIS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _migrate_legacy_table(conn: sqlite3.Connection) -> None:
    """One-time migration from the old "analysis" table name/shape (no
    speaker_name column) into the new call_logs table, if the old table
    exists and still has data. Safe to call every startup — a no-op once
    migrated (checks whether the old table exists at all)."""
    if not _table_exists(conn, "analysis"):
        return
    try:
        old_rows = conn.execute("SELECT * FROM analysis").fetchall()
    except sqlite3.OperationalError:
        return
    for row in old_rows:
        row_dict = dict(row)
        conn.execute(
            """
            INSERT OR IGNORE INTO call_logs
                (call_id, timestamp, speaker_name, transcript, amount, urgency, spoof_score, similarity, risk)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row_dict.get("call_id"),
                row_dict.get("timestamp"),
                None,  # legacy rows never had a speaker_name
                row_dict.get("transcript"),
                row_dict.get("amount"),
                row_dict.get("urgency"),
                row_dict.get("spoof_score"),
                row_dict.get("similarity"),
                row_dict.get("risk"),
            ),
        )
    conn.execute("DROP TABLE analysis")


def init_db() -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS call_logs (
                call_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                speaker_name TEXT,
                transcript TEXT,
                amount REAL,
                urgency TEXT,
                spoof_score REAL,
                similarity REAL,
                risk TEXT
            )
            """
        )
        _migrate_legacy_table(conn)
        conn.commit()
    finally:
        conn.close()


def save_analysis(
    transcript: Optional[str],
    spoof_score: float,
    similarity: Optional[float],
    amount: Optional[float],
    urgency: str,
    risk: str,
    speaker_name: Optional[str] = None,
) -> str:
    """Inserts a new call record and returns its generated call_id."""
    call_id = uuid.uuid4().hex
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO call_logs
                (call_id, timestamp, speaker_name, transcript, amount, urgency, spoof_score, similarity, risk)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                call_id,
                datetime.now(timezone.utc).isoformat(),
                speaker_name,
                transcript,
                amount,
                urgency,
                spoof_score,
                similarity,
                risk,
            ),
        )
        conn.commit()
        return call_id
    finally:
        conn.close()


def list_recent_analyses(limit: int = 10) -> List[dict]:
    """Task 6: "Recent Analyses" dashboard section shows the latest 10 —
    default limit matches that."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM call_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
