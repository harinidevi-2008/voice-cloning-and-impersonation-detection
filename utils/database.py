import sqlite3
import numpy as np

DB_PATH = "database/voice.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users(
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            role TEXT,
            embedding BLOB
        )
    """)

    conn.commit()
    conn.close()


def save_embedding(name, role, embedding):

    conn = sqlite3.connect(DB_PATH)

    blob = embedding.astype(np.float32).tobytes()

    cursor = conn.execute(
        "INSERT INTO users(name, role, embedding) VALUES (?, ?, ?)",
        (name, role, blob)
    )

    conn.commit()

    uid = cursor.lastrowid

    conn.close()

    return uid


def load_embedding(user_id):

    conn = sqlite3.connect(DB_PATH)

    row = conn.execute(
        "SELECT embedding FROM users WHERE user_id=?",
        (user_id,)
    ).fetchone()

    conn.close()

    if row is None:
        return None

    return np.frombuffer(row[0], dtype=np.float32)