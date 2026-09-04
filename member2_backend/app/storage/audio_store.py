"""
Handles saving uploaded audio files to disk and basic validation.
Both /enroll and /analyze funnel their uploads through here so file-handling
logic (naming, extension checks, size limits) lives in exactly one place.
"""

import os
import re
import uuid
from fastapi import UploadFile, HTTPException

from app.config import (
    AUDIO_UPLOAD_DIR,
    ALLOWED_AUDIO_EXTENSIONS,
    MAX_AUDIO_SIZE_BYTES,
)

# Anything outside this set gets replaced with "_". Deliberately permissive
# enough to keep the mock AI service's filename hints working (e.g.
# "genuine_alice.wav") while rejecting path separators, "..", null bytes,
# and anything else that could escape AUDIO_UPLOAD_DIR or confuse Windows.
_SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_filename(raw_filename: str) -> str:
    """
    Never trust upload_file.filename directly — it's client-controlled and
    can contain path separators (e.g. "../../etc/passwd", or Windows-style
    "..\\..\\evil.wav") intended to escape AUDIO_UPLOAD_DIR when joined into
    a path. os.path.basename() strips any directory components first (does
    this for both "/" and "\\" regardless of host OS), then the character
    allowlist strips anything else that isn't safe on all supported
    platforms. The UUID prefix added by the caller still guarantees
    uniqueness even after sanitization collapses distinct inputs.
    """
    basename = os.path.basename(raw_filename.replace("\\", "/"))
    safe = _SAFE_FILENAME_CHARS.sub("_", basename)
    safe = safe.lstrip(".")  # avoid dotfiles / bare "."/".." after stripping
    return safe or "audio"


def _validate_extension(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio file type '{ext}'. "
                f"Allowed types: {sorted(ALLOWED_AUDIO_EXTENSIONS)}"
            ),
        )
    return ext


def save_upload_file(upload_file: UploadFile, prefix: str = "audio") -> str:
    """
    Saves an UploadFile to AUDIO_UPLOAD_DIR with a unique, collision-free,
    sanitized name while preserving the (sanitized) original filename (mock
    AI functions use filename hints like 'genuine_'/'clone_' to produce
    plausible demo scores). Returns the absolute path to the saved file.
    """
    if upload_file is None or not upload_file.filename:
        raise HTTPException(status_code=400, detail="No audio file was provided.")

    ext = _validate_extension(upload_file.filename)
    safe_filename = _sanitize_filename(upload_file.filename)

    os.makedirs(AUDIO_UPLOAD_DIR, exist_ok=True)
    unique_name = f"{prefix}_{uuid.uuid4().hex[:10]}_{safe_filename}"
    dest_path = os.path.join(AUDIO_UPLOAD_DIR, unique_name)

    # Defense in depth: even after sanitization, confirm the resolved path
    # is actually inside AUDIO_UPLOAD_DIR before writing anything.
    if os.path.commonpath([os.path.abspath(dest_path), os.path.abspath(AUDIO_UPLOAD_DIR)]) != os.path.abspath(AUDIO_UPLOAD_DIR):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    contents = upload_file.file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(contents) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Audio file too large. Max size is {MAX_AUDIO_SIZE_BYTES // (1024*1024)} MB.",
        )

    with open(dest_path, "wb") as f:
        f.write(contents)

    return dest_path
