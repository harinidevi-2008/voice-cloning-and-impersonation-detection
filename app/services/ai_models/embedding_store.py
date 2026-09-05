"""
Adapted from Member 1's utils/database.py (voice-cloning-and-impersonation-
detection repo). Stores ECAPA-TDNN speaker embeddings, keyed by user_id.

CHANGE FROM ORIGINAL: DB_PATH was a relative path ("database/voice.db"),
which only worked from one specific working directory and put runtime data
outside this project's existing data/ convention. Moved under
member2_backend/data/ (already gitignored) alongside the main app database.

This is intentionally a SEPARATE SQLite file from app/db/database.py's
voice_integrity.db. That database owns application-level user metadata
(name, role, enrolled_at) for the dashboard and API; this one owns the raw
embedding vectors, which the main app schema has no column for. The two are
kept in sync by user_id — see app/routers/enroll.py and the note in
app/services/real_ai_service.py for exactly how.
"""

import os
import sqlite3
import numpy as np

from app.config import DATA_DIR

EMBEDDING_DB_PATH = os.path.join(DATA_DIR, "voice_embeddings.db")
EMBEDDING_DIMENSION = 192


def init_db() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(EMBEDDING_DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users(
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                role TEXT,
                embedding BLOB
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_embedding(name: str, role: str, embedding: np.ndarray) -> int:
    """Inserts a new embedding and returns the auto-generated user_id."""
    conn = sqlite3.connect(EMBEDDING_DB_PATH)
    try:
        blob = embedding.astype(np.float32).tobytes()
        cursor = conn.execute(
            "INSERT INTO users(name, role, embedding) VALUES (?, ?, ?)",
            (name, role, blob),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def save_embedding_with_id(user_id: int, name: str, role: str, embedding: np.ndarray) -> int:
    """
    Inserts (or overwrites) an embedding at an EXPLICIT user_id.

    Not part of Member 1's original interface — added so this database's
    user_id can be kept in lockstep with app/db/database.py's, which is the
    ID the dashboard and API actually expose. Without this, a fresh restart
    of one database but not the other could desync the two auto-increment
    counters and cause get_similarity() to look up the wrong (or a
    nonexistent) embedding.
    """
    conn = sqlite3.connect(EMBEDDING_DB_PATH)
    try:
        blob = embedding.astype(np.float32).tobytes()
        conn.execute(
            "INSERT OR REPLACE INTO users(user_id, name, role, embedding) VALUES (?, ?, ?, ?)",
            (user_id, name, role, blob),
        )
        conn.commit()
        return user_id
    finally:
        conn.close()


def load_embedding(user_id: int):
    """Load a usable float32 embedding, or ``None`` for absent/bad legacy rows."""
    conn = sqlite3.connect(EMBEDDING_DB_PATH)
    try:
        row = conn.execute(
            "SELECT embedding FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
    finally:
        conn.close()

    if row is None or row[0] is None:
        return None
    try:
        embedding = np.frombuffer(row[0], dtype=np.float32)
    except (TypeError, ValueError):
        return None
    return embedding if embedding.size else None


def has_valid_embedding(user_id: int) -> bool:
    """True only for a finite ECAPA-sized vector usable for verification."""
    embedding = load_embedding(user_id)
    return bool(
        embedding is not None
        and embedding.ndim == 1
        and embedding.size == EMBEDDING_DIMENSION
        and np.isfinite(embedding).all()
    )


def delete_embedding(user_id: int) -> None:
    """
    Rollback helper — see app/routers/enroll.py and app/db/crud.py's
    delete_user(). Included for defensiveness/symmetry: in the current
    enrollment flow the app database row is created before the embedding
    is written, so a failure normally means nothing was written here yet.
    But if that ordering ever changes, this keeps both databases
    consistent rather than leaving an orphaned embedding behind.
    """
    conn = sqlite3.connect(EMBEDDING_DB_PATH)
    try:
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()
