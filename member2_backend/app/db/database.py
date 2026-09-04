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
                audio_path TEXT NOT NULL,
                enrolled_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
