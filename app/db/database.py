"""
Minimal SQLite setup. No ORM — a hackathon backend does not need one, and
plain sqlite3 keeps the dependency list (and the mental model) small.
"""

import sqlite3
import os
from app.config import DB_PATH, DATA_DIR


def get_connection() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                audio_path TEXT,
                enrolled_at TEXT NOT NULL
            )
            """
        )
        # Older builds stored the temporary enrollment WAV path in a NOT NULL
        # column. Preserve all non-audio metadata while removing that path.
        columns = {row[1]: row for row in conn.execute("PRAGMA table_info(users)")}
        if columns.get("audio_path") and columns["audio_path"][3]:
            has_status = "embedding_status" in columns
            conn.execute("ALTER TABLE users RENAME TO users_legacy_audio_path")
            conn.execute(
                """
                CREATE TABLE users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    audio_path TEXT,
                    enrolled_at TEXT NOT NULL,
                    embedding_status TEXT NOT NULL DEFAULT 'incomplete'
                )
                """
            )
            status_sql = "embedding_status" if has_status else "'incomplete'"
            conn.execute(
                "INSERT INTO users (user_id, name, role, audio_path, enrolled_at, embedding_status) "
                f"SELECT user_id, name, role, NULL, enrolled_at, {status_sql} "
                "FROM users_legacy_audio_path"
            )
            conn.execute("DROP TABLE users_legacy_audio_path")
        # SQLite does not add new columns when CREATE TABLE IF NOT EXISTS is
        # run against an existing database, so migrate the original schema.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
        if "embedding_status" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN embedding_status TEXT NOT NULL DEFAULT 'incomplete'"
            )
        conn.commit()
    finally:
        conn.close()
