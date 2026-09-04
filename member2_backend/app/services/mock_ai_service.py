"""
mock_ai_service.py
===================

The DEFAULT AI backend for this project — deterministic, hash-based stand-
ins for Member 1's models. This is still the backend automated tests run
against (fast, no heavy dependencies, no downloads).

STATUS: Member 1 has delivered real models (XLS-R+AASIST spoof detection,
ECAPA-TDNN speaker verification). They're wired in at
app/services/real_ai_service.py and selected by setting the environment
variable VISL_AI_BACKEND=real (see app/config.py and
app/services/ai_service.py). This mock file is kept as-is and remains the
default so:
  - The test suite (tests/test_api.py) keeps running in milliseconds
    without needing torch/speechbrain installed or real audio files.
  - Anyone without the real models set up yet (or without a GPU/enough
    RAM) can still run the full backend + dashboard end-to-end.

This module implements exactly the three function signatures Member 1
committed to:

    get_spoof_score(audio_path: str) -> float
    enroll_speaker(name: str, role: str, audio_path: str) -> user_id
    get_similarity(audio_path: str, user_id: int) -> float

Determinism:
All "scores" here are derived from SHA-256 hashes of filenames/ids, not
random.random(), so re-running /analyze on the same file always produces
the same numbers — important for demoing and for automated tests.
As a demo convenience, filenames containing "genuine", "clone",
"impersonator", "synthetic", or "spoof" bias the mock scores in the
expected direction (see Member 3's demo audio naming convention).
"""

import hashlib
import os


def _hash_to_unit_float(seed: str) -> float:
    """Deterministically maps a string to a float in [0, 1)."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def get_spoof_score(audio_path: str) -> float:
    """
    MOCK for XLS-R + AASIST synthetic/spoof voice detection.
    Returns a float in [0, 1]: higher = more likely AI-generated/spoofed.
    """
    filename = os.path.basename(audio_path).lower()
    base = _hash_to_unit_float(filename)

    if any(tag in filename for tag in ("clone", "synthetic", "spoof")):
        return round(0.75 + base * 0.24, 4)          # clearly spoofed range
    if "impersonator" in filename:
        return round(0.55 + base * 0.30, 4)           # human impersonator, not AI clone
    if any(tag in filename for tag in ("genuine", "real")):
        return round(base * 0.25, 4)                  # clearly genuine range
    return round(base, 4)                              # unknown filename -> neutral


def enroll_speaker(name: str, role: str, audio_path: str) -> int:
    """
    MOCK for ECAPA-TDNN speaker embedding extraction + enrollment.
    Returns a deterministic pseudo-id (see integration note above: this
    value is NOT the canonical user_id used by the rest of the backend).
    """
    seed = f"{name}:{role}:{os.path.basename(audio_path)}"
    return int(_hash_to_unit_float(seed) * 1_000_000)


def get_similarity(audio_path: str, user_id: int) -> float:
    """
    MOCK for ECAPA-TDNN speaker verification (cosine similarity).
    Returns a float in [0, 1]: higher = more likely the claimed speaker.
    """
    filename = os.path.basename(audio_path).lower()
    seed = f"{filename}:{user_id}"
    base = _hash_to_unit_float(seed)

    if "genuine" in filename:
        return round(0.80 + base * 0.19, 4)            # high similarity
    if any(tag in filename for tag in ("clone", "impersonator", "synthetic", "spoof")):
        return round(0.20 + base * 0.35, 4)             # low similarity
    return round(0.40 + base * 0.30, 4)                 # unknown filename -> mid range
