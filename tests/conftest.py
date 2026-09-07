"""Test-only protected-template configuration, established before app imports."""

import os
import tempfile

from cryptography.fernet import Fernet


_embedding_test_dir = tempfile.TemporaryDirectory(prefix="visl_embedding_tests_")
os.environ.setdefault("VISL_EMBEDDING_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
os.environ.setdefault(
    "VISL_EMBEDDING_DB_PATH", os.path.join(_embedding_test_dir.name, "voice_embeddings.db")
)
