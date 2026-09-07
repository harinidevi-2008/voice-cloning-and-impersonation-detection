import sqlite3

import numpy as np
import pytest
from cryptography.fernet import Fernet

from app.config import EMBEDDING_DB_PATH, EMBEDDING_ENCRYPTION_KEY_ENV
from app.services.ai_models import embedding_store
from app.services.ai_models.exceptions import (
    EmbeddingEncryptionKeyError,
    EncryptedEmbeddingError,
)


def _embedding() -> np.ndarray:
    return np.linspace(-1.0, 1.0, embedding_store.EMBEDDING_DIMENSION, dtype=np.float32)


def _stored_blob(user_id: int) -> bytes:
    conn = sqlite3.connect(EMBEDDING_DB_PATH)
    try:
        return conn.execute("SELECT embedding FROM users WHERE user_id = ?", (user_id,)).fetchone()[0]
    finally:
        conn.close()


def test_template_is_encrypted_in_sqlite_and_round_trips():
    user_id = embedding_store.save_embedding("Template Test", "customer", _embedding())
    blob = _stored_blob(user_id)

    assert isinstance(blob, bytes)
    assert blob.startswith(b"gAAAA")  # Fernet token, not a raw NumPy vector.
    assert blob != _embedding().tobytes()
    assert len(blob) > _embedding().nbytes
    np.testing.assert_array_equal(embedding_store.load_embedding(user_id), _embedding())


def test_missing_or_wrong_key_fails_closed(monkeypatch):
    user_id = embedding_store.save_embedding("Key Test", "customer", _embedding())
    monkeypatch.delenv(EMBEDDING_ENCRYPTION_KEY_ENV)
    with pytest.raises(EmbeddingEncryptionKeyError):
        embedding_store.load_embedding(user_id)
    assert not embedding_store.has_valid_embedding(user_id)

    monkeypatch.setenv(EMBEDDING_ENCRYPTION_KEY_ENV, Fernet.generate_key().decode("ascii"))
    with pytest.raises(EncryptedEmbeddingError):
        embedding_store.load_embedding(user_id)

    monkeypatch.setenv(EMBEDDING_ENCRYPTION_KEY_ENV, "not-a-valid-fernet-key")
    with pytest.raises(EmbeddingEncryptionKeyError):
        embedding_store.save_embedding("Malformed Key", "customer", _embedding())


def test_corrupted_template_and_bad_dimensions_fail_safely():
    user_id = embedding_store.save_embedding("Corrupt Test", "customer", _embedding())
    conn = sqlite3.connect(EMBEDDING_DB_PATH)
    try:
        conn.execute("UPDATE users SET embedding = ? WHERE user_id = ?", (b"not-a-fernet-token", user_id))
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(EncryptedEmbeddingError):
        embedding_store.load_embedding(user_id)
    assert not embedding_store.has_valid_embedding(user_id)
    with pytest.raises(ValueError):
        embedding_store.save_embedding("Bad Vector", "customer", np.ones(4, dtype=np.float32))


def test_legacy_plaintext_template_is_migrated_or_invalidated_without_a_key(monkeypatch):
    embedding_store.init_db()
    conn = sqlite3.connect(EMBEDDING_DB_PATH)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO users(user_id, name, role, embedding, template_version) VALUES (?, ?, ?, ?, 0)",
            (8811, "Legacy Key", "customer", _embedding().tobytes()),
        )
        conn.commit()
    finally:
        conn.close()

    embedding_store.init_db()
    assert _stored_blob(8811).startswith(b"gAAAA")
    np.testing.assert_array_equal(embedding_store.load_embedding(8811), _embedding())

    conn = sqlite3.connect(EMBEDDING_DB_PATH)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO users(user_id, name, role, embedding, template_version) VALUES (?, ?, ?, ?, 0)",
            (8812, "Legacy No Key", "customer", _embedding().tobytes()),
        )
        conn.commit()
    finally:
        conn.close()
    monkeypatch.delenv(EMBEDDING_ENCRYPTION_KEY_ENV)
    embedding_store.init_db()
    assert _stored_blob(8812) is None
    assert not embedding_store.has_valid_embedding(8812)


def test_verification_uses_decrypted_original_embedding(monkeypatch):
    pytest.importorskip("speechbrain")
    from app.services.ai_models import speaker_verifier

    reference = _embedding()
    embedding_store.save_embedding_with_id(7861, "Verification Test", "customer", reference)
    monkeypatch.setattr(speaker_verifier, "extract_embedding", lambda _path: reference.copy())

    assert speaker_verifier.get_similarity("temporary.wav", 7861) == pytest.approx(1.0)
