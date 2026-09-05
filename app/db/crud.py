"""
CRUD helpers for enrolled users/speakers.

NOTE ON THE user_id CONTRACT:
This local database is the authoritative source of `user_id` for now
(auto-incrementing primary key). When Member 1's real speaker-enrollment
pipeline is integrated, `user_id` should stay driven by this table, and
Member 1's `enroll_speaker()` should be treated as "register/return an
embedding reference for this user_id" rather than as the ID generator.
See services/mock_ai_service.py for the corresponding note.
"""

from datetime import datetime, timezone
from typing import Optional, List
from app.db.database import get_connection


def create_user(name: str, role: str, audio_path: str) -> int:
    conn = get_connection()
    try:
        enrolled_at = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            "INSERT INTO users (name, role, audio_path, enrolled_at) VALUES (?, ?, ?, ?)",
            (name, role, audio_path, enrolled_at),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_user(user_id: int) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT user_id, name, role, audio_path, enrolled_at FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_user(user_id: int) -> None:
    """
    Used to roll back an enrollment if a later step (currently: registering
    the voiceprint with the AI service) fails after the user row was
    already created — see app/routers/enroll.py. Without this, a failed
    enrollment would leave a "ghost user": a row in this table with no
    corresponding embedding, which would then 404 or error confusingly the
    next time someone tried to verify against it.
    """
    conn = get_connection()
    try:
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def set_embedding_status(user_id: int, status: str) -> None:
    """Record whether this profile is safe to offer for verification."""
    if status not in {"ready", "incomplete"}:
        raise ValueError("embedding status must be 'ready' or 'incomplete'")
    conn = get_connection()
    try:
        conn.execute("UPDATE users SET embedding_status = ? WHERE user_id = ?", (status, user_id))
        conn.commit()
    finally:
        conn.close()


def list_users() -> List[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT user_id, name, role, enrolled_at FROM users ORDER BY user_id ASC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
