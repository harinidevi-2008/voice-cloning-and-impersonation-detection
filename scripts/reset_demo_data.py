"""Explicitly reset demo records without touching model weights or caches.

Run once before a clean demo. This script is intentionally never imported by
application startup.
"""

import os
import sqlite3
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.config import ANALYSIS_DB_PATH, DB_PATH  # noqa: E402
from app.services.ai_models.embedding_store import EMBEDDING_DB_PATH  # noqa: E402


def _delete_rows(database_path: str, table: str, *, reset_sequence: bool) -> int:
    if not os.path.exists(database_path):
        return 0
    conn = sqlite3.connect(database_path)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            return 0
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.execute(f"DELETE FROM {table}")
        if reset_sequence:
            conn.execute("DELETE FROM sqlite_sequence WHERE name=?", (table,))
        conn.commit()
        return count
    finally:
        conn.close()


def main() -> None:
    main_users = _delete_rows(DB_PATH, "users", reset_sequence=True)
    embeddings = _delete_rows(EMBEDDING_DB_PATH, "users", reset_sequence=True)
    analyses = _delete_rows(ANALYSIS_DB_PATH, "call_logs", reset_sequence=False)

    print(f"Main users removed: {main_users}")
    print(f"Embeddings removed: {embeddings}")
    print(f"Analysis records removed: {analyses}")
    print("Database IDs reset.")
    print("Next enrollment will use user_id=1.")


if __name__ == "__main__":
    main()
