"""
Tests for _sanitize_filename() / save_upload_file() path-traversal
protections (app/storage/audio_store.py, Task 5). Unit-level: constructs
UploadFile-like objects directly rather than going through the full FastAPI
TestClient, so these run fast and don't depend on any heavy AI deps.
"""

import io
import os
import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from app.config import AUDIO_UPLOAD_DIR
from app.storage.audio_store import save_upload_file, _sanitize_filename


def _make_upload(filename: str, content: bytes = b"fake-audio-bytes") -> UploadFile:
    return UploadFile(
        filename=filename,
        file=io.BytesIO(content),
        headers=Headers({"content-type": "audio/wav"}),
    )


def test_sanitize_strips_unix_path_traversal():
    assert _sanitize_filename("../../etc/passwd.wav") == "passwd.wav"


def test_sanitize_strips_windows_path_traversal():
    assert _sanitize_filename("..\\..\\Windows\\System32\\evil.wav") == "evil.wav"


def test_sanitize_strips_absolute_path():
    assert _sanitize_filename("/etc/passwd.wav") == "passwd.wav"
    assert _sanitize_filename("C:\\secrets\\keys.wav") == "keys.wav"


def test_sanitize_replaces_unsafe_characters():
    result = _sanitize_filename("gen uine;rm -rf.wav")
    assert " " not in result
    assert ";" not in result
    assert result.endswith(".wav")


def test_sanitize_preserves_normal_filenames_for_mock_hints():
    # The mock AI service reads filename hints like "genuine_"/"clone_" —
    # make sure ordinary, well-behaved filenames pass through unchanged.
    assert _sanitize_filename("genuine_alice.wav") == "genuine_alice.wav"
    assert _sanitize_filename("clone_bob-01.wav") == "clone_bob-01.wav"


def test_save_upload_file_with_malicious_filename_stays_inside_upload_dir():
    malicious = _make_upload("../../../../etc/passwd.wav")
    saved_path = save_upload_file(malicious, prefix="test")
    try:
        real_upload_dir = os.path.realpath(AUDIO_UPLOAD_DIR)
        real_saved_path = os.path.realpath(saved_path)
        assert real_saved_path.startswith(real_upload_dir + os.sep)
        assert os.path.exists(saved_path)
    finally:
        if os.path.exists(saved_path):
            os.remove(saved_path)


def test_save_upload_file_rejects_extension_regardless_of_sanitization():
    # Sanitization must not accidentally rescue a disallowed extension.
    from fastapi import HTTPException
    bad = _make_upload("../../evil.exe")
    with pytest.raises(HTTPException) as exc_info:
        save_upload_file(bad, prefix="test")
    assert exc_info.value.status_code == 400
