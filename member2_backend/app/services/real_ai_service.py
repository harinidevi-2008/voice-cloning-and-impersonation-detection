"""
real_ai_service.py
===================

The REAL counterpart to mock_ai_service.py, wired to Member 1's actual
models (XLS-R+AASIST for spoof detection, ECAPA-TDNN for speaker
verification). Implements the exact same 3-function contract:

    get_spoof_score(audio_path: str) -> float
    enroll_speaker(name: str, role: str, audio_path: str) -> user_id
    get_similarity(audio_path: str, user_id: int) -> float

Selected automatically when the VISL_AI_BACKEND environment variable is set
to "real" (see app/config.py and app/services/ai_service.py). Requires the
heavy ML dependencies in requirements-real-ai.txt to be installed.

IMPORTANT — user_id synchronization:
Member 1's enroll_speaker() returns an ID from ITS OWN embeddings database
(app/services/ai_models/embedding_store.py), separate from the main app
database (app/db/database.py) that this backend's API/dashboard actually
expose as `user_id`. get_similarity() looks up the stored embedding by that
same ID, so the two databases MUST agree on IDs for every enrolled user, or
verification will silently look up the wrong (or a nonexistent) embedding.

enroll_speaker() below is kept for interface completeness (e.g. so this
module still "looks like" a drop-in replacement for mock_ai_service), but
app/routers/enroll.py does NOT call it directly. It calls enroll_speaker_at()
instead, which pins the embedding to the user_id that app/db/database.py
already assigned — that's what keeps the two databases in lockstep. This is
the resolution to the integration risk flagged back when mock_ai_service.py
was first written (see that file's own docstring).
"""

from app.services.ai_models.spoof_detector import get_spoof_score as _get_spoof_score
from app.services.ai_models import speaker_verifier as _speaker_verifier


def get_spoof_score(audio_path: str) -> float:
    return _get_spoof_score(audio_path)


def get_similarity(audio_path: str, user_id: int) -> float:
    return _speaker_verifier.get_similarity(audio_path, user_id)


def enroll_speaker(name: str, role: str, audio_path: str) -> int:
    """
    Matches the original 3-function contract exactly. NOT used by
    app/routers/enroll.py (which calls enroll_speaker_at instead) — kept so
    this module is a complete, honest drop-in for mock_ai_service.py for
    anyone importing it directly (e.g. ad-hoc scripts, tests).

    Returns an ID from the embeddings database's own autoincrement counter,
    which will only match the main app database's user_id if the two have
    never diverged (e.g. fresh databases, or every enrollment so far went
    through enroll_speaker_at).
    """
    return _speaker_verifier.enroll_speaker(name, role, audio_path)


def enroll_speaker_at(user_id: int, name: str, role: str, audio_path: str) -> int:
    """
    Real-backend-only helper. Stores the speaker embedding at an EXPLICIT
    user_id — the one app/db/database.py already assigned — instead of
    letting the embeddings database pick its own. This is what
    app/routers/enroll.py actually calls when AI_BACKEND=real.
    """
    return _speaker_verifier.enroll_speaker_with_id(user_id, name, role, audio_path)
