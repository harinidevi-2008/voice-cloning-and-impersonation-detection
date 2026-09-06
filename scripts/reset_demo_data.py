"""Explicitly reset demo records without touching model weights or caches.

Run once before a clean demo.  This script is intentionally never imported
by application startup.
"""

import os
import sqlite3
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.config import ANALYSIS_DB_PATH, AUDIO_UPLOAD_DIR, DB_PATH
from app.services.ai_models.embedding_store import EMBEDDING_DB_PATH


def _delete_rows(database_path: str, table: str, reset_sequence: bool) -> int:
    if not os.path.exists(database_path):
        return 0
    conn = sqlite3.connect(database_path)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            return 0
        count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        conn.execute(f'DELETE FROM "{table}"')
        if reset_sequence and conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
        ).fetchone():
            conn.execute("DELETE FROM sqlite_sequence WHERE name=?", (table,))
        conn.commit()
        return count
    finally:
        conn.close()


def main() -> None:
    main_users = _delete_rows(DB_PATH, "users", True)
    embeddings = _delete_rows(EMBEDDING_DB_PATH, "users", True)
    analyses = _delete_rows(ANALYSIS_DB_PATH, "call_logs", False)
    # Clear pre-migration history too, if a legacy database still has it.
    analyses += _delete_rows(ANALYSIS_DB_PATH, "analysis", False)
    removed_uploads = 0
    if os.path.isdir(AUDIO_UPLOAD_DIR):
        for filename in os.listdir(AUDIO_UPLOAD_DIR):
            path = os.path.join(AUDIO_UPLOAD_DIR, filename)
            if filename != ".gitkeep" and os.path.isfile(path):
                os.remove(path)
                removed_uploads += 1
    print(f"Main users removed: {main_users}")
    print(f"Embeddings removed: {embeddings}")
    print(f"Analysis records removed: {analyses}")
    print(f"Transient uploads removed: {removed_uploads}")
    print("Database IDs reset. Next enrollment will use user_id=1.")


if __name__ == "__main__":
    main()
