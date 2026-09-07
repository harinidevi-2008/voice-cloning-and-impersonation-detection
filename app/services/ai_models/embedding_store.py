"""Encrypted ECAPA-TDNN biometric speaker-template storage.

The embedding database is intentionally separate from the application user
database. It stores authenticated-encrypted templates only: no raw enrollment
audio, feature files, or plaintext float vectors are persisted.
"""

import os
import sqlite3

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

from app.config import DATA_DIR, EMBEDDING_DB_PATH, EMBEDDING_ENCRYPTION_KEY_ENV
from app.services.ai_models.exceptions import (
    EmbeddingEncryptionKeyError,
    EncryptedEmbeddingError,
)

EMBEDDING_DIMENSION = 192
_ENCRYPTED_TEMPLATE_VERSION = 1


def _get_fernet() -> Fernet:
    """Read the external Fernet key and fail closed if it is unusable."""
    raw_key = os.environ.get(EMBEDDING_ENCRYPTION_KEY_ENV)
    if not raw_key:
        raise EmbeddingEncryptionKeyError(
            f"{EMBEDDING_ENCRYPTION_KEY_ENV} must be configured to use speaker verification"
        )
    try:
        return Fernet(raw_key.encode("ascii"))
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EmbeddingEncryptionKeyError("Speaker-template encryption key is malformed") from exc


def _serialize_embedding(embedding: np.ndarray) -> bytes:
    vector = np.asarray(embedding, dtype=np.float32)
    if vector.ndim != 1 or vector.size != EMBEDDING_DIMENSION or not np.isfinite(vector).all():
        raise ValueError("Invalid ECAPA embedding")
    return vector.tobytes()


def _deserialize_embedding(payload: bytes) -> np.ndarray:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise EncryptedEmbeddingError("Invalid protected speaker template")
    try:
        embedding = np.frombuffer(bytes(payload), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise EncryptedEmbeddingError("Invalid protected speaker template") from exc
    if embedding.ndim != 1 or embedding.size != EMBEDDING_DIMENSION or not np.isfinite(embedding).all():
        raise EncryptedEmbeddingError("Invalid protected speaker template")
    return embedding.copy()


def _encrypt_embedding(embedding: np.ndarray) -> bytes:
    return _get_fernet().encrypt(_serialize_embedding(embedding))


def _decrypt_embedding(token: bytes) -> np.ndarray:
    try:
        plaintext = _get_fernet().decrypt(bytes(token))
    except (InvalidToken, TypeError, ValueError) as exc:
        raise EncryptedEmbeddingError("Speaker template decryption failed") from exc
    try:
        return _deserialize_embedding(plaintext)
    finally:
        # Python cannot guarantee a secure wipe; avoid retaining another
        # plaintext reference longer than the required conversion.
        plaintext = None


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
                embedding BLOB,
                template_version INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
        if "template_version" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN template_version INTEGER NOT NULL DEFAULT 0")

        # Migrate valid legacy float32 vectors when a deployment key is
        # present. Without a key, invalidate legacy bytes rather than leave
        # plaintext biometric templates on disk; user metadata is preserved.
        legacy_rows = conn.execute(
            "SELECT user_id, embedding FROM users WHERE template_version = 0 AND embedding IS NOT NULL"
        ).fetchall()
        if legacy_rows:
            try:
                fernet = _get_fernet()
            except EmbeddingEncryptionKeyError:
                fernet = None
            for user_id, blob in legacy_rows:
                plaintext = None
                try:
                    if fernet is None:
                        raise ValueError("no encryption key available for migration")
                    plaintext = _serialize_embedding(np.frombuffer(blob, dtype=np.float32))
                    encrypted = fernet.encrypt(plaintext)
                except (TypeError, ValueError):
                    conn.execute(
                        "UPDATE users SET embedding = ?, template_version = ? WHERE user_id = ?",
                        (None, _ENCRYPTED_TEMPLATE_VERSION, user_id),
                    )
                else:
                    conn.execute(
                        "UPDATE users SET embedding = ?, template_version = ? WHERE user_id = ?",
                        (encrypted, _ENCRYPTED_TEMPLATE_VERSION, user_id),
                    )
                finally:
                    plaintext = None
        conn.commit()
    finally:
        conn.close()


def save_embedding(name: str, role: str, embedding: np.ndarray) -> int:
    """Encrypt and insert an embedding, returning its generated user ID."""
    init_db()
    conn = sqlite3.connect(EMBEDDING_DB_PATH)
    try:
        cursor = conn.execute(
            "INSERT INTO users(name, role, embedding, template_version) VALUES (?, ?, ?, ?)",
            (name, role, _encrypt_embedding(embedding), _ENCRYPTED_TEMPLATE_VERSION),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def save_embedding_with_id(user_id: int, name: str, role: str, embedding: np.ndarray) -> int:
    """Encrypt and persist an embedding at the application-owned user ID."""
    init_db()
    conn = sqlite3.connect(EMBEDDING_DB_PATH)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO users(user_id, name, role, embedding, template_version) VALUES (?, ?, ?, ?, ?)",
            (user_id, name, role, _encrypt_embedding(embedding), _ENCRYPTED_TEMPLATE_VERSION),
        )
        conn.commit()
        return user_id
    finally:
        conn.close()


def load_embedding(user_id: int):
    """Decrypt a protected float32 template only for an active comparison."""
    conn = sqlite3.connect(EMBEDDING_DB_PATH)
    try:
        row = conn.execute(
            "SELECT embedding, template_version FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
    finally:
        conn.close()

    if row is None or row[0] is None:
        return None
    if row[1] != _ENCRYPTED_TEMPLATE_VERSION:
        raise EncryptedEmbeddingError("Speaker template is not an encrypted current-format template")
    return _decrypt_embedding(row[0])


def has_valid_embedding(user_id: int) -> bool:
    """True only for a finite, decryptable ECAPA-sized protected template."""
    try:
        embedding = load_embedding(user_id)
    except (EmbeddingEncryptionKeyError, EncryptedEmbeddingError):
        return False
    return bool(
        embedding is not None
        and embedding.ndim == 1
        and embedding.size == EMBEDDING_DIMENSION
        and np.isfinite(embedding).all()
    )


def delete_embedding(user_id: int) -> None:
    """Remove a protected template when its enrollment is rolled back."""
    conn = sqlite3.connect(EMBEDDING_DB_PATH)
    try:
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()
