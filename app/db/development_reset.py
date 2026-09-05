"""Development-only reset of runtime data; schemas are deliberately retained."""

import os
import sqlite3

from app.config import ANALYSIS_DB_PATH, AUDIO_UPLOAD_DIR, DB_PATH
from app.services.ai_models.embedding_store import EMBEDDING_DB_PATH


def _clear_table_and_sequence(database_path: str, table: str) -> None:
    """Delete data from one SQLite table and reset its AUTOINCREMENT counter."""
    conn = sqlite3.connect(database_path)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            return
        conn.execute(f'DELETE FROM "{table}"')
        # sqlite_sequence only exists after an AUTOINCREMENT table has been used.
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
        ).fetchone():
            conn.execute("DELETE FROM sqlite_sequence WHERE name=?", (table,))
        conn.commit()
    finally:
        conn.close()


def reset_development_data() -> None:
    """Clear runtime records and uploads without dropping or changing any table."""
    _clear_table_and_sequence(DB_PATH, "users")
    _clear_table_and_sequence(EMBEDDING_DB_PATH, "users")
    _clear_table_and_sequence(ANALYSIS_DB_PATH, "call_logs")
    # A legacy table can exist before analysis_db's startup migration.
    _clear_table_and_sequence(ANALYSIS_DB_PATH, "analysis")

    if not os.path.isdir(AUDIO_UPLOAD_DIR):
        return
    for root, directories, filenames in os.walk(AUDIO_UPLOAD_DIR, topdown=False):
        for filename in filenames:
            if filename != ".gitkeep":
                os.unlink(os.path.join(root, filename))
        for directory in directories:
            os.rmdir(os.path.join(root, directory))
