"""
ai_service.py
=============
Single import point for get_spoof_score / get_similarity, so routers (and
any future code) don't need to branch on the backend themselves. Picked
once, at import time, based on app.config.AI_BACKEND (env var
VISL_AI_BACKEND — defaults to "mock").

Enrollment is deliberately NOT re-exported here — see app/routers/enroll.py.
The real backend needs extra handling to keep its embeddings database's
user_id in sync with the main app database, so that router branches
explicitly between mock_ai_service.enroll_speaker and
real_ai_service.enroll_speaker_at rather than going through a generic
dispatcher here.
"""

from app.config import AI_BACKEND

if AI_BACKEND == "real":
    from app.services.real_ai_service import get_spoof_score, get_similarity
else:
    from app.services.mock_ai_service import get_spoof_score, get_similarity

__all__ = ["get_spoof_score", "get_similarity"]
