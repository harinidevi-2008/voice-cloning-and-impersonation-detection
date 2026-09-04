import logging
import os

from fastapi import APIRouter, Form, File, UploadFile, HTTPException

from app.schemas import EnrollResponse
from app.storage.audio_store import save_upload_file
from app.db import crud
from app.config import AI_BACKEND
from app.services import mock_ai_service
from app.services.ai_models.exceptions import AudioDecodeError

if AI_BACKEND == "real":
    from app.services import real_ai_service

router = APIRouter()
logger = logging.getLogger("visl.enroll")


@router.post("/enroll", response_model=EnrollResponse)
async def enroll_user(
    name: str = Form(..., min_length=1, description="Full name of the person being enrolled"),
    role: str = Form(..., min_length=1, description="e.g. 'customer', 'employee', 'executive'"),
    audio_file: UploadFile = File(..., description="Reference voice sample for enrollment"),
):
    """
    Enrolls a new speaker: saves their reference audio, stores their profile
    in the local database, and registers their voiceprint with the AI
    speaker-verification model (mock or real, per AI_BACKEND).

    ATOMICITY: if voiceprint registration fails after the user row has
    already been created, the user row is deleted and the saved audio file
    is removed, rather than leaving a "ghost user" — a row with no
    corresponding embedding, which would 404 or fail confusingly the next
    time someone tried to verify against it. See crud.delete_user().
    """
    name = name.strip()
    role = role.strip()
    if not name:
        raise HTTPException(status_code=400, detail="'name' must not be empty.")
    if not role:
        raise HTTPException(status_code=400, detail="'role' must not be empty.")

    saved_path = save_upload_file(audio_file, prefix="enroll")

    # Our local DB is authoritative for user_id (see crud.py note).
    user_id = crud.create_user(name=name, role=role, audio_path=saved_path)

    try:
        if AI_BACKEND == "real":
            # Pin the embedding to OUR user_id, not whatever the embeddings
            # DB would auto-assign — this is what keeps app/db/database.py
            # and app/services/ai_models/embedding_store.py in sync. See
            # real_ai_service.py's docstring for the full explanation.
            real_ai_service.enroll_speaker_at(
                user_id=user_id, name=name, role=role, audio_path=saved_path
            )
        else:
            # Exercise the mock interface for parity; result unused (mock's
            # pseudo-id was never the source of truth — see mock_ai_service.py).
            mock_ai_service.enroll_speaker(name=name, role=role, audio_path=saved_path)
    except AudioDecodeError as exc:
        _rollback(user_id, saved_path)
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the uploaded audio for enrollment: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 — any AI-service failure must not leave a ghost user
        _rollback(user_id, saved_path)
        logger.exception("Enrollment failed for user_id=%s during voiceprint registration", user_id)
        raise HTTPException(
            status_code=500,
            detail="Enrollment failed while registering the voiceprint. "
                   "No user was created — please try again.",
        )

    return EnrollResponse(user_id=user_id)


def _rollback(user_id: int, saved_path: str) -> None:
    """Best-effort cleanup: delete the user row and the uploaded file."""
    try:
        crud.delete_user(user_id)
    except Exception:  # noqa: BLE001 — rollback must never mask the original error
        logger.exception("Failed to roll back user_id=%s after enrollment failure", user_id)

    try:
        if saved_path and os.path.exists(saved_path):
            os.remove(saved_path)
    except OSError:
        logger.exception("Failed to remove audio file %s after enrollment failure", saved_path)
