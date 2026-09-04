import logging
import time
from typing import Optional
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Response

from app.schemas import AnalyzeResponse
from app.storage.audio_store import save_upload_file
from app.db import crud
from app.services import ai_service
from app.services import context_engine
from app.services import risk_engine
from app.services.ai_models.exceptions import AudioDecodeError
from app.config import AI_BACKEND, URGENCY_RISK_MAP

router = APIRouter()
logger = logging.getLogger("visl.analyze")


def _parse_bool(value: str, field_name: str) -> bool:
    truthy = {"true", "1", "yes", "y"}
    falsy = {"false", "0", "no", "n"}
    normalized = str(value).strip().lower()
    if normalized in truthy:
        return True
    if normalized in falsy:
        return False
    raise HTTPException(
        status_code=400,
        detail=f"'{field_name}' must be a boolean-like value (true/false), got '{value}'.",
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_call(
    response: Response,
    audio_file: UploadFile = File(..., description="Audio sample from the call/transaction"),
    claimed_user_id: Optional[int] = Form(
        None, description="Enrolled user_id the caller claims to be, or omit/null if unknown"
    ),
    transaction_value: float = Form(..., ge=0, description="Monetary value of the transaction"),
    urgency: str = Form(..., description=f"One of: {list(URGENCY_RISK_MAP.keys())}"),
    caller_known: str = Form(..., description="true/false — is the caller a known contact?"),
):
    """
    Runs the full pipeline for one call/transaction:
      1. Save uploaded audio.
      2. Get spoof_score from the AI detection model (mock or real,
         selected by AI_BACKEND — see app/services/ai_service.py).
      3. If claimed_user_id is given, verify it exists, then get
         speaker_similarity from the speaker verification model.
      4. Compute context_risk from transaction metadata.
      5. Fuse everything into impersonation_risk and a verdict.

    Latency instrumentation (Day 2 "measure end-to-end latency" task):
    the response body's shape is frozen by the Section 3 interface
    contract, so timing is NOT added as a JSON field. Instead it's logged
    server-side and returned as the X-Processing-Time-Ms response header —
    covers the "audio in -> risk score out" window judges tend to ask
    about, without touching the agreed contract at all.
    """
    pipeline_start = time.perf_counter()

    # --- validate caller_known ---
    caller_known_bool = _parse_bool(caller_known, "caller_known")

    # --- validate urgency (fail loudly rather than silently defaulting —
    #     required for the risk engine to stay transparent/explainable) ---
    if urgency.strip().lower() not in URGENCY_RISK_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"'urgency' must be one of {list(URGENCY_RISK_MAP.keys())}, got '{urgency}'.",
        )

    # --- validate claimed_user_id, if provided ---
    if claimed_user_id is not None:
        user = crud.get_user(claimed_user_id)
        if user is None:
            raise HTTPException(
                status_code=404,
                detail=f"claimed_user_id {claimed_user_id} is not enrolled.",
            )

    # --- save audio ---
    saved_path = save_upload_file(audio_file, prefix="analyze")

    # --- Member 1 interface calls (mock or real, per AI_BACKEND) ---
    ai_start = time.perf_counter()
    try:
        spoof_score = ai_service.get_spoof_score(saved_path)

        speaker_similarity = None
        if claimed_user_id is not None:
            speaker_similarity = ai_service.get_similarity(saved_path, claimed_user_id)
    except AudioDecodeError as exc:
        # The extension/size checks in audio_store.py only look at the
        # filename — a corrupt or non-audio file with a valid extension
        # only surfaces here, when the real AI pipeline actually tries to
        # decode it. Without this, that would otherwise be an unhandled
        # 500 instead of a clean client error.
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the uploaded audio: {exc}",
        )
    ai_elapsed_ms = (time.perf_counter() - ai_start) * 1000

    # --- context risk ---
    context_result = context_engine.compute_context_risk(
        caller_known=caller_known_bool,
        transaction_value=transaction_value,
        urgency=urgency,
    )
    context_risk = context_result["context_risk"]

    # --- fusion ---
    identity_mismatch_risk = risk_engine.compute_identity_mismatch_risk(speaker_similarity)
    impersonation_risk = risk_engine.compute_impersonation_risk(
        spoof_score=spoof_score,
        identity_mismatch_risk=identity_mismatch_risk,
        context_risk=context_risk,
    )
    verdict = risk_engine.get_verdict(impersonation_risk)

    total_elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
    response.headers["X-Processing-Time-Ms"] = f"{total_elapsed_ms:.1f}"
    logger.info(
        "analyze: backend=%s claimed_user_id=%s ai_ms=%.1f total_ms=%.1f verdict=%s",
        AI_BACKEND, claimed_user_id, ai_elapsed_ms, total_elapsed_ms, verdict,
    )

    return AnalyzeResponse(
        spoof_score=spoof_score,
        speaker_similarity=speaker_similarity,
        context_risk=context_risk,
        impersonation_risk=impersonation_risk,
        verdict=verdict,
    )
